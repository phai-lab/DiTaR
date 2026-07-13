"""WAN-TC latent-diffusion reward model for DrQ-v2-style RL.

Wraps `ditar_encoder`'s `WanTcLatentDiffusionLAM` (the reward model under
`TSEncoder/`) to expose a `calc_reward(imgs)` API for the RL loop:

    imgs: torch.Tensor of shape [1, T, H, W, 3], values in [-1, 1] OR [0, 1].
    returns: torch.Tensor of shape [T] — per-step reward to be scattered into
             the episode by the replay buffer.

The reward at step t (for t with at least 4 future frames available) is

    E_t = MC mean over (tau, eps) of  ||x0_hat(z_tau, c_t) - x0_t||^2,
    r_t = -E_t                                        (variant "neg_e")
        | exp(-E_t)                                   (variant "exp")
        | tau_td * (E_{t-1} - E_t)                    (variant "td_residual")
        | tau_td * (exp(-E_t) - exp(-E_{t-1}))        (variant "td_exp")

For t near the end of the episode (no 4 future frames), the source frame
is held and the missing future frames are repeated. See README.

The reward model **does not** apply any normalisation by demo statistics
(`use_std` in baseline). RL caller can scale via `expl_scale` and combine
with env reward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_pixel_01(imgs: torch.Tensor) -> torch.Tensor:
    """Accept either [-1, 1] or [0, 1] uint-mapped float; return [0, 1] float."""
    if imgs.dtype != torch.float32:
        imgs = imgs.float()
    mn, mx = float(imgs.min()), float(imgs.max())
    # [-1, 1] convention: any negative value indicates this range.
    if mn < -0.05:
        if mn < -1.05 or mx > 1.05:
            raise ValueError(f"imgs claim [-1,1] but got min={mn} max={mx}")
        imgs = (imgs + 1.0) * 0.5
    else:
        if mx > 1.05:
            raise ValueError(f"imgs claim [0,1] but max={mx}")
    return imgs.clamp_(0.0, 1.0)


class WanTcLatentDiffusionReward(nn.Module):
    """Reward model wrapping the WAN TC v1.1 flow-matching diffusion."""

    SUPPORTED_REWARD_TYPES = {"neg_e", "exp", "td_residual", "td_exp"}

    def __init__(self, cfg):
        super().__init__()

        import os
        import sys
        # The reward model ships as the `ditar_encoder` package (bundled under
        # <repo>/TSEncoder, or installed/checked out separately). It is an
        # OPTIONAL dependency: importing `ours.reward_models` does not require
        # it, only *instantiating* this class does. Default to the bundled copy;
        # override via cfg.taskvae_root or REWARD_MODEL_ROOT / TASKVAE_ROOT.
        reward_root = (
            getattr(cfg, "taskvae_root", "")
            or os.environ.get("REWARD_MODEL_ROOT", "")
            or os.environ.get("TASKVAE_ROOT", "")
            or str(Path(__file__).resolve().parents[2] / "TSEncoder")
        )
        if reward_root:
            reward_root = str(Path(reward_root).expanduser())
            if reward_root not in sys.path:
                sys.path.insert(0, reward_root)
        from ditar_encoder.diffusion.wan_tc_latent_diffusion_model import (  # noqa: E402
            WanTcLatentDiffusionLAM,
        )

        # Paths to the frozen sub-models the diffusion ckpt depends on. Defaults
        # are HuggingFace repo ids (auto-downloaded); all are overridable via the
        # reward yaml / env vars. See CHECKPOINTS.md.
        default_wan_vae = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        default_t5 = "t5-base"
        default_stats = os.environ.get("WAN_TC_STATS", "")

        ckpt_path = str(Path(cfg.ckpt_path).expanduser())
        wan_vae_path = str(getattr(cfg, "wan_vae_path", "") or default_wan_vae)
        t5_path = str(getattr(cfg, "t5_path", "") or default_t5)
        stats_path = str(Path(getattr(cfg, "stats_path", "") or default_stats).expanduser())
        source_stats_path = str(Path(getattr(cfg, "source_stats_path", stats_path)).expanduser())

        # The diffusion checkpoint may embed the trainer-side absolute path
        # for the frozen TC teacher under hparams.stage2_checkpoint. Override at
        # load time so we can reproduce in a different filesystem.
        load_kwargs = dict(
            map_location="cpu",
            strict=False,
            wan_vae_model_path=wan_vae_path,
            t5_model_path=t5_path,
            target_stats_path=stats_path,
            source_stats_path=source_stats_path,
        )
        stage2_ckpt_path = getattr(cfg, "stage2_ckpt_path", None)
        if stage2_ckpt_path:
            load_kwargs["stage2_checkpoint"] = str(Path(stage2_ckpt_path).expanduser())

        model = WanTcLatentDiffusionLAM.load_from_checkpoint(ckpt_path, **load_kwargs)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

        # Auto-detect window geometry from the loaded model so the wrapper can
        # serve all three variants (t1 first-frame / t2 history / t3 textjoint)
        # without per-variant code paths.
        self.source_mode = str(getattr(model, "source_mode", "first_frame_wan_l0"))
        target_window = int(model.tc_teacher.teacher.teacher_num_frames)
        if self.source_mode == "history_5frame_wan2":
            history_len = int(model.history_num_frames)
            self.window = history_len + target_window - 1
            self.anchor_offset = history_len - 1
        else:
            self.window = target_window
            self.anchor_offset = 0
        self.future_chunk = target_window - 1
        self.target_tokens = int(model.core.target_tokens)
        self.target_dim = int(model.core.target_dim)

        # Reward configuration.
        self.reward_type = str(getattr(cfg, "reward_type", "exp")).lower()
        if self.reward_type not in self.SUPPORTED_REWARD_TYPES:
            raise ValueError(f"reward_type {self.reward_type!r} not in {self.SUPPORTED_REWARD_TYPES}")
        self.K = int(getattr(cfg, "mc_samples", 4))
        self.tau_min = float(getattr(cfg, "tau_min", 1e-2))
        self.tau_max = float(getattr(cfg, "tau_max", 1.0))
        self.td_scale = float(getattr(cfg, "td_scale", 1.0))
        self.batch_size_internal = int(getattr(cfg, "internal_batch_size", 8))
        # Stride between adjacent frames inside one window. stride=1 keeps the
        # window contiguous in env time (the RL default). Larger strides match
        # how the diffusion was trained with temporal sub-sampling, but cost
        # more padding on both ends per step.
        self.stride = int(getattr(cfg, "stride", 1))
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        # TD differencing gap: r_t = E_{t-k} - E_t (or exp analog) with k =
        # td_step_gap. Decouples the differencing horizon from the window
        # stride so stride=1 stays in-distribution while the TD signal jumps
        # across a meaningful number of env steps.
        self.td_step_gap = int(getattr(cfg, "td_step_gap", 4))
        if self.td_step_gap <= 0:
            raise ValueError("td_step_gap must be a positive integer")

        # Multiplier on the (already-normalized) learned reward at the very
        # end of ``calc_reward``. Combined with ``expl_scale`` this lets the
        # agent's reward composition be written as
        #     r_t = learned_scale * r_learned + r_env + expl_scale * r_rnd
        # The baseline default is 1.0 (no scaling). With the user's
        # 0.95-learned / 0.05-rnd split, set learned_scale=0.95 in the yaml.
        self.learned_scale = float(getattr(cfg, "learned_scale", 1.0))

        # Demo-based z-score normalization (μ, σ per reward_type), stored in
        # the file pointed to by ``reward_stats_path``. Schema v2 (preferred):
        #     {variant: {"demo": {mean, std, ...}, "mixed": {...}}, ...}
        # Schema v1 (legacy, still readable): {variant: {mean, std, ...}}
        # The ``stats_mode`` config field picks which branch to read (demo or
        # mixed). Default is ``demo`` — calibration with expert demos only.
        self.normalize_reward = bool(getattr(cfg, "normalize_reward", True))
        # H1' tier-A: clip negative tail. Use when demo-z makes RL trajectories
        # systematically negative (penalising exploration). With reward_nonneg=true
        # the post-z reward is max(z, 0): expert-like trajectories still get >0,
        # non-expert get 0 instead of negative.
        self.reward_nonneg = bool(getattr(cfg, "reward_nonneg", False))
        self.stats_mode = str(getattr(cfg, "stats_mode", "demo")).lower()
        if self.stats_mode not in {"demo", "mixed"}:
            raise ValueError(f"stats_mode must be 'demo' or 'mixed', got {self.stats_mode!r}")
        self._reward_mu = 0.0
        self._reward_sigma = 1.0
        reward_stats_path = getattr(cfg, "reward_stats_path", None)
        if self.normalize_reward and reward_stats_path:
            self._load_reward_stats(str(Path(reward_stats_path).expanduser()))
        elif self.normalize_reward:
            print(
                "[WanTcReward] normalize_reward=True but no reward_stats_path "
                "provided; falling back to identity (no normalization). Run the "
                "calibration script to generate stats."
            )

        # Default text instruction; can be overridden per-call via set_instruction().
        self._instruction: Optional[str] = getattr(cfg, "default_instruction", None)
        if self._instruction is None:
            self._instruction = ""
        # If still empty AND cfg has task_name, try the registry.
        if not self._instruction:
            task_name = getattr(cfg, "task_name", None)
            if task_name:
                from .instructions import get_instruction
                self._instruction = get_instruction(str(task_name))
                if self._instruction:
                    print(f"[WanTcReward] auto-resolved instruction for task '{task_name}': {self._instruction!r}")

        # Pre-encoded shared MC noise per episode is set lazily so paired_sampling
        # across timesteps within an episode is enabled.
        self._mc_eps: Optional[torch.Tensor] = None
        self._mc_taus: Optional[torch.Tensor] = None

        # Required attribute consumed by replay_buffer integration.
        self.use_env_reward = bool(getattr(cfg, "use_env_reward", True))

        # Optional exploration-reward submodule (RND), instantiated through
        # Hydra exactly like baseline `RND.__init__`. Without this, the
        # DrQ-v2 agent's update step tries `self.rm.calc_expl_reward(...)`
        # against a non-existent module.
        self.use_expl_reward = bool(getattr(cfg, "use_expl_reward", False))
        self.expl_scale = float(getattr(cfg, "expl_scale", 0.0))
        if self.use_expl_reward and hasattr(cfg, "expl_reward"):
            try:
                import hydra
                cfg.expl_reward.obs_shape = getattr(cfg, "obs_shape", "???")
                cfg.expl_reward.action_shape = getattr(cfg, "action_shape", "???")
                self.expl_reward = hydra.utils.instantiate(cfg.expl_reward)
            except Exception as exc:  # pragma: no cover
                print(f"[WanTcReward] expl_reward instantiate failed ({exc}); disabling")
                self.use_expl_reward = False
                self.expl_scale = 0.0

        # Mirror the baseline `DiffusionReward.__init__` pattern: assign every
        # cfg key as a self.* attribute. This is what training scripts expect
        # (e.g. self.rm.expl_update_interval, self.rm.expl_scale). Without this
        # the first DrQ-v2 update step raises AttributeError.
        try:
            cfg_items = cfg.items()
        except AttributeError:
            cfg_items = []
            for k in dir(cfg):
                if not k.startswith("_"):
                    cfg_items.append((k, getattr(cfg, k)))
        for attr_name, attr_value in cfg_items:
            if not hasattr(self, attr_name):
                setattr(self, attr_name, attr_value)

    # ------------------------------------------------------------------ helpers
    def _load_reward_stats(self, stats_path: str) -> None:
        """Read (mu, sigma) for the active reward_type from a calibration JSON.

        Schema v2 (preferred):
            {variant: {"demo":  {"mean": ..., "std": ...},
                       "mixed": {"mean": ..., "std": ...}}, ...}
        Schema v1 (legacy fallback):
            {variant: {"mean": ..., "std": ...}, ...}

        ``stats_mode`` picks the demo / mixed branch. If a stats file lacks
        the requested branch, falls back to identity (mean=0, std=1).
        """
        import json
        p = Path(stats_path)
        if not p.exists():
            print(f"[WanTcReward] reward_stats_path={stats_path} not found; using identity normalization")
            return
        try:
            data = json.loads(p.read_text())
        except Exception as exc:
            print(f"[WanTcReward] failed to parse reward stats at {stats_path}: {exc}")
            return
        entry = data.get(self.reward_type)
        if not isinstance(entry, dict):
            print(f"[WanTcReward] reward_stats has no entry for reward_type={self.reward_type!r}; identity normalization")
            return
        # Detect schema: v2 has 'demo'/'mixed' sub-dicts; v1 has 'mean'/'std' at top
        if "demo" in entry or "mixed" in entry:
            branch = entry.get(self.stats_mode)
            if not isinstance(branch, dict) or not branch:
                print(
                    f"[WanTcReward] stats_mode={self.stats_mode!r} branch missing/empty for variant "
                    f"{self.reward_type!r}; identity normalization"
                )
                return
            mu = float(branch.get("mean", 0.0))
            sigma = float(branch.get("std", 1.0))
            schema = "v2"
        else:
            mu = float(entry.get("mean", 0.0))
            sigma = float(entry.get("std", 1.0))
            schema = "v1"
        if sigma < 1e-6:
            print(f"[WanTcReward] reward_stats std={sigma:.2e} too small; clamping to 1e-6")
            sigma = 1e-6
        self._reward_mu = mu
        self._reward_sigma = sigma
        print(
            f"[WanTcReward] loaded reward stats ({schema}, mode={self.stats_mode}) for "
            f"{self.reward_type!r}: mean={mu:.6f}, std={sigma:.6f}"
        )

    def set_instruction(self, instruction: str) -> None:
        """RL caller can pass the task instruction string per-episode."""
        self._instruction = str(instruction)

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _refresh_mc_noise(self, T_windows: int) -> None:
        """Sample K paired (tau, eps) shared across all windows in this episode."""
        device = self._device()
        self._mc_taus = torch.empty(self.K, device=device).uniform_(self.tau_min, self.tau_max)
        self._mc_eps = torch.randn(
            self.K, T_windows, self.target_tokens, self.target_dim, device=device,
        )

    @torch.no_grad()
    def _extract_window_latents(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """windows: [W, self.window, 3, 256, 256] in [0,1]. Returns (source, target_tc).

        source shape is [W, 256, 48] for first-frame mode and [W, 512, 48] for
        5-frame history mode; target_tc is always [W, target_tokens, target_dim].
        """
        return self.model._extract_online_latents(windows)

    @torch.no_grad()
    def _compute_E_for_batch(self, source: torch.Tensor, target: torch.Tensor,
                             eps_slice: torch.Tensor) -> torch.Tensor:
        """source [B,256,48], target [B,64,32], eps_slice [K,B,64,32]. Returns [B] mean over K."""
        device = self._device()
        source_n = self.model._normalize_source(source.to(device))
        x0_n = self.model._normalize_target(target.to(device))
        instructions = [self._instruction] * source.shape[0]
        text_tokens, text_mask = self.model.text_conditioner(instructions)
        text_tokens = text_tokens.float().to(device)
        if text_mask is not None:
            text_mask = text_mask.to(device)
        losses = []
        for k in range(self.K):
            t_scalar = self._mc_taus[k].to(device)
            t_batch = t_scalar.expand(x0_n.shape[0]).float()
            e = eps_slice[k]
            z_t = (1.0 - t_scalar) * x0_n + t_scalar * e
            x0_hat = self.model.core(z_t, source_n, text_tokens, text_mask, t_batch)
            losses.append((x0_hat - x0_n).pow(2).mean(dim=(1, 2)))
        return torch.stack(losses, dim=0).mean(dim=0)

    # ----------------------------------------------------------------- API
    @torch.no_grad()
    def compute_Et(self, imgs: torch.Tensor) -> torch.Tensor:
        """Same preprocessing as ``calc_reward`` but returns the raw paired-MC
        x0-MSE residual ``E_t`` array of shape [T] without any variant
        transform or normalization. Used by the calibration script to derive
        all four reward variants from a single forward pass.
        """
        device = self._device()
        if imgs.ndim != 5:
            raise ValueError(f"Expect 5-D imgs [1,T,H,W,3], got shape {tuple(imgs.shape)}")
        if imgs.shape[0] != 1:
            raise ValueError("compute_Et expects batch dim 1 (one episode)")
        imgs01 = _to_pixel_01(imgs.to(device))
        T = imgs01.shape[1]
        videos = imgs01.permute(0, 1, 4, 2, 3)
        H, W = videos.shape[-2], videos.shape[-1]
        if (H, W) != (256, 256):
            v = videos.flatten(0, 1)
            v = F.interpolate(v, size=(256, 256), mode="bilinear", align_corners=False)
            videos = v.unsqueeze(0)
        frames = videos[0]
        idx_offsets = (torch.arange(self.window, device=device, dtype=torch.long)
                       - self.anchor_offset) * self.stride
        t_grid = torch.arange(T, device=device, dtype=torch.long).unsqueeze(1)
        idx = (t_grid + idx_offsets.unsqueeze(0)).clamp_(0, T - 1)
        windows = frames[idx]

        bs = self.batch_size_internal
        src_list, tgt_list = [], []
        for i in range(0, T, bs):
            src, tgt = self._extract_window_latents(windows[i:i+bs])
            src_list.append(src)
            tgt_list.append(tgt)
        source_all = torch.cat(src_list, dim=0)
        target_all = torch.cat(tgt_list, dim=0)
        self._refresh_mc_noise(T)
        Et = torch.empty(T, device=device)
        for i in range(0, T, bs):
            j = min(i + bs, T)
            eps_slice = self._mc_eps[:, i:j]
            Et[i:j] = self._compute_E_for_batch(source_all[i:j], target_all[i:j], eps_slice)
        return Et

    def apply_variant(self, Et: torch.Tensor) -> torch.Tensor:
        """Apply the configured reward_type transform to a precomputed E_t
        array. Used by calibration to derive all variants without re-running
        the diffusion forward; calc_reward calls this internally too.
        """
        T = Et.shape[0]
        k = self.td_step_gap
        if self.reward_type == "neg_e":
            return -Et
        if self.reward_type == "exp":
            return torch.exp(-Et)
        if self.reward_type == "td_residual":
            r = torch.zeros_like(Et)
            if T > k:
                r[k:] = self.td_scale * (Et[:T - k] - Et[k:])
            return r
        if self.reward_type == "td_exp":
            s = torch.exp(-Et)
            r = torch.zeros_like(Et)
            if T > k:
                r[k:] = self.td_scale * (s[k:] - s[:T - k])
            return r
        raise RuntimeError(f"unreachable reward_type={self.reward_type!r}")

    @torch.no_grad()
    def calc_reward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: [1, T, H, W, 3] in [-1,1] or [0,1]. Returns [T] reward."""
        Et = self.compute_Et(imgs)
        reward = self.apply_variant(Et)
        # Demo-based z-score normalization (μ, σ from offline calibration).
        if self.normalize_reward and self._reward_sigma > 0:
            reward = (reward - self._reward_mu) / self._reward_sigma
        # Tier-A: clip negative tail so non-expert trajectories get 0 rather
        # than a penalty. Applied AFTER z-score, BEFORE learned_scale.
        if self.reward_nonneg:
            reward = torch.clamp(reward, min=0.0)
        # Final scaling so agent reward composition reads as
        # ``learned_scale * r_learned + r_env + expl_scale * r_rnd``.
        if self.learned_scale != 1.0:
            reward = reward * self.learned_scale
        return reward

    # Agent.update hook: delegate to RND if configured, else no-op.
    def update(self, batch):
        metrics = {}
        if self.use_expl_reward and hasattr(self, "expl_reward"):
            metrics.update(self.expl_reward.update(batch))
        return metrics

    # Agent reward composition: returns [B, 1] (matching env-reward shape) and
    # delegates to RND when expl_scale > 0; zeros otherwise.
    @torch.no_grad()
    def calc_expl_reward(self, obs, next_obs):
        if self.use_expl_reward and hasattr(self, "expl_reward") and self.expl_scale > 0:
            return self.expl_reward.calc_reward(obs, next_obs) * self.expl_scale
        return torch.zeros(obs.shape[0], 1, device=obs.device, dtype=torch.float32)
