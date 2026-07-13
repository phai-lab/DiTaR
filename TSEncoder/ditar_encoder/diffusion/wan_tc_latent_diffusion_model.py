from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer


OptimizerCallable = Callable[[Iterable], Optimizer]
# No baked-in default: the Stage-2 (task-relevant) encoder checkpoint must be
# supplied via config/CLI, e.g. `${DITAR_STAGE2_CKPT}` or an explicit path.
DEFAULT_STAGE2_CHECKPOINT = ""
CANONICAL_SOURCE_MODE = "history_5frame_wan2"


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ditar_encoder").is_dir() and (parent / "configs").is_dir():
            return parent
    return Path(__file__).resolve().parents[2]


def _resolve_repo_relative(path_like: str | Path) -> Path:
    path = Path(os.path.expandvars(str(path_like))).expanduser()
    if not path.is_absolute() and not path.exists():
        candidate = _repo_root() / path
        if candidate.exists():
            return candidate
    return path


def _resolve_checkpoint(path_like: str | Path) -> Path:
    path = _resolve_repo_relative(path_like)
    if path.is_file():
        return path
    checkpoint_dir = path / "checkpoints"
    if checkpoint_dir.is_dir():
        for name in ("last.ckpt", "last-v1.ckpt"):
            candidate = checkpoint_dir / name
            if candidate.exists():
                return candidate
        best = sorted(checkpoint_dir.glob("best-*.ckpt"))
        if best:
            return best[0]
    raise FileNotFoundError(f"Unable to resolve checkpoint from {path_like!r}")


def _normalize_source_mode(source_mode: str | None) -> str:
    normalized = str(source_mode or CANONICAL_SOURCE_MODE).strip().lower()
    if normalized not in {"first_frame_wan_l0", CANONICAL_SOURCE_MODE}:
        raise ValueError("source_mode must be 'first_frame_wan_l0' or 'history_5frame_wan2'")
    return normalized


def _has_diffusers_config(path: Path, subfolder: str) -> bool:
    try:
        return path.exists() and (path / subfolder / "config.json").exists()
    except OSError:
        return False


def _candidate_hf_cache_roots() -> Sequence[Path]:
    roots = []
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        hf_home = Path(os.environ["HF_HOME"]).expanduser()
        roots.extend([hf_home, hf_home / "hub"])
    roots.extend([Path.home() / ".cache" / "huggingface" / "hub", Path.home() / "cache" / "huggingface"])
    for parent in _repo_root().parents:
        roots.extend([parent / "cache" / "huggingface", parent / ".cache" / "huggingface" / "hub"])
    unique = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _resolve_model_snapshot(model_path: Optional[str], subfolder: str) -> Optional[str]:
    if not model_path:
        return None
    path = Path(os.path.expandvars(model_path)).expanduser()
    if _has_diffusers_config(path, subfolder):
        return str(path)
    parts = list(path.parts)
    model_index = next((idx for idx, part in enumerate(parts) if part.startswith("models--")), None)
    if model_index is None:
        return str(path)
    tail = Path(*parts[model_index:])
    for root in _candidate_hf_cache_roots():
        candidate = root / tail
        if _has_diffusers_config(candidate, subfolder):
            return str(candidate)
    return str(path)


def _make_sincos_timestep_embedding(t: Tensor, dim: int) -> Tensor:
    if t.ndim != 1:
        t = t.reshape(-1)
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _zero_last_linear(module: nn.Sequential) -> None:
    last = module[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError(f"Expected last module to be nn.Linear, got {type(last).__name__}")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.0, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class WanTcDiTBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        text_gate_init: float = 0.0,
        text_mode: str = "cross_attn_target_only",
    ) -> None:
        super().__init__()
        self.text_mode = str(text_mode).strip().lower()
        if self.text_mode not in {"cross_attn_target_only", "in_joint_attn"}:
            raise ValueError(
                "text_mode must be 'cross_attn_target_only' or 'in_joint_attn'"
            )

        self.joint_norm_attn = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.joint_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.joint_mod_attn = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 3))

        self.joint_norm_mlp = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.joint_mlp = FeedForward(model_dim, dropout=dropout)
        self.joint_mod_mlp = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 3))

        if self.text_mode == "cross_attn_target_only":
            self.text_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
            self.text_cross_attn = nn.MultiheadAttention(
                model_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.text_mod = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 2))
            self.text_gate = nn.Parameter(torch.full((), float(text_gate_init)))

        self.target_norm_mlp = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.target_mlp = FeedForward(model_dim, dropout=dropout)
        self.target_mod_mlp = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 3))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _zero_last_linear(self.joint_mod_attn)
        _zero_last_linear(self.joint_mod_mlp)
        if self.text_mode == "cross_attn_target_only":
            _zero_last_linear(self.text_mod)
        _zero_last_linear(self.target_mod_mlp)

    def forward(
        self,
        source_tokens: Tensor,
        target_tokens: Tensor,
        text_tokens: Optional[Tensor],
        text_mask: Optional[Tensor],
        cond: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        num_source = int(source_tokens.shape[1])

        if self.text_mode == "in_joint_attn" and text_tokens is not None:
            num_text = int(text_tokens.shape[1])
            joint_tokens = torch.cat(
                [
                    source_tokens,
                    text_tokens.to(device=source_tokens.device, dtype=source_tokens.dtype),
                    target_tokens,
                ],
                dim=1,
            )
            seq_len = joint_tokens.shape[1]
            joint_key_padding_mask: Optional[Tensor] = None
            if text_mask is not None:
                kpm = torch.zeros(
                    joint_tokens.shape[0],
                    seq_len,
                    device=joint_tokens.device,
                    dtype=torch.bool,
                )
                kpm[:, num_source : num_source + num_text] = ~text_mask.to(
                    device=joint_tokens.device, dtype=torch.bool
                )
                joint_key_padding_mask = kpm
        else:
            num_text = 0
            joint_tokens = torch.cat([source_tokens, target_tokens], dim=1)
            joint_key_padding_mask = None

        shift, scale, gate = self.joint_mod_attn(cond).chunk(3, dim=-1)
        joint_norm = _modulate(self.joint_norm_attn(joint_tokens), shift, scale)
        joint_tokens = joint_tokens + gate.unsqueeze(1) * self.joint_attn(
            joint_norm,
            joint_norm,
            joint_norm,
            key_padding_mask=joint_key_padding_mask,
            need_weights=False,
        )[0]

        shift, scale, gate = self.joint_mod_mlp(cond).chunk(3, dim=-1)
        joint_norm = _modulate(self.joint_norm_mlp(joint_tokens), shift, scale)
        joint_tokens = joint_tokens + gate.unsqueeze(1) * self.joint_mlp(joint_norm)

        if self.text_mode == "in_joint_attn" and text_tokens is not None:
            source_tokens = joint_tokens[:, :num_source]
            target_tokens = joint_tokens[:, num_source + num_text :]
        else:
            source_tokens, target_tokens = joint_tokens[:, :num_source], joint_tokens[:, num_source:]

            if text_tokens is not None and self.text_mode == "cross_attn_target_only":
                shift, scale = self.text_mod(cond).chunk(2, dim=-1)
                target_norm = _modulate(self.text_norm(target_tokens), shift, scale)
                key_padding_mask = None
                if text_mask is not None:
                    key_padding_mask = ~text_mask.to(device=target_tokens.device, dtype=torch.bool)
                cross = self.text_cross_attn(
                    target_norm,
                    text_tokens.to(device=target_tokens.device, dtype=target_tokens.dtype),
                    text_tokens.to(device=target_tokens.device, dtype=target_tokens.dtype),
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0]
                target_tokens = target_tokens + self.text_gate * cross

        shift, scale, gate = self.target_mod_mlp(cond).chunk(3, dim=-1)
        target_norm = _modulate(self.target_norm_mlp(target_tokens), shift, scale)
        target_tokens = target_tokens + gate.unsqueeze(1) * self.target_mlp(target_norm)
        return source_tokens, target_tokens


class WanTcLatentDiffusionCore(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        num_heads: int = 8,
        dit_blocks: int = 12,
        source_tokens: int = 512,
        target_tokens: int = 4,
        source_dim: int = 48,
        target_dim: int = 32,
        text_dim: int = 768,
        dropout: float = 0.0,
        text_gate_init: float = 0.0,
        text_mode: str = "in_joint_attn",
        t5_max_length: int = 64,
    ) -> None:
        super().__init__()
        normalized_text_mode = str(text_mode).strip().lower()
        if normalized_text_mode not in {"cross_attn_target_only", "in_joint_attn"}:
            raise ValueError(
                "text_mode must be 'cross_attn_target_only' or 'in_joint_attn'"
            )
        self.text_mode = normalized_text_mode
        self.t5_max_length = int(t5_max_length)
        self.model_dim = int(model_dim)
        self.source_tokens = int(source_tokens)
        self.target_tokens = int(target_tokens)
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.text_dim = int(text_dim)

        self.source_proj = nn.Linear(self.source_dim, self.model_dim)
        self.target_in_proj = nn.Linear(self.target_dim, self.model_dim)
        self.text_proj = nn.Sequential(nn.LayerNorm(self.text_dim), nn.Linear(self.text_dim, self.model_dim))
        self.target_out_norm = nn.LayerNorm(self.model_dim)
        self.target_out_proj = nn.Linear(self.model_dim, self.target_dim)

        self.source_pos_embed = nn.Parameter(torch.zeros(1, self.source_tokens, self.model_dim))
        self.target_query_embed = nn.Parameter(torch.zeros(1, self.target_tokens, self.model_dim))
        num_type_rows = 3 if self.text_mode == "in_joint_attn" else 2
        self.type_embed = nn.Parameter(torch.zeros(num_type_rows, self.model_dim))
        if self.text_mode == "in_joint_attn":
            self.text_pos_embed = nn.Parameter(torch.zeros(1, self.t5_max_length, self.model_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim * 4),
            nn.SiLU(),
            nn.Linear(self.model_dim * 4, self.model_dim),
        )
        self.blocks = nn.ModuleList(
            [
                WanTcDiTBlock(
                    model_dim=self.model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    text_gate_init=text_gate_init,
                    text_mode=self.text_mode,
                )
                for _ in range(dit_blocks)
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.source_pos_embed, std=0.02)
        nn.init.normal_(self.target_query_embed, std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)
        if self.text_mode == "in_joint_attn":
            nn.init.normal_(self.text_pos_embed, std=0.02)
        nn.init.zeros_(self.target_out_proj.weight)
        nn.init.zeros_(self.target_out_proj.bias)

    def _check_shapes(self, noisy_target: Tensor, source: Tensor) -> None:
        if source.shape[1:] != (self.source_tokens, self.source_dim):
            raise ValueError(f"Expected source [B,{self.source_tokens},{self.source_dim}], got {tuple(source.shape)}")
        if noisy_target.shape[1:] != (self.target_tokens, self.target_dim):
            raise ValueError(
                f"Expected noisy_target [B,{self.target_tokens},{self.target_dim}], got {tuple(noisy_target.shape)}"
            )

    def forward(
        self,
        noisy_target: Tensor,
        source: Tensor,
        text_tokens: Optional[Tensor],
        text_mask: Optional[Tensor],
        timesteps: Tensor,
    ) -> Tensor:
        self._check_shapes(noisy_target, source)
        cond = self.time_mlp(_make_sincos_timestep_embedding(timesteps, self.model_dim).to(dtype=source.dtype))
        source_tokens = self.source_proj(source) + self.source_pos_embed + self.type_embed[0].view(1, 1, -1)
        target_tokens = (
            self.target_in_proj(noisy_target) + self.target_query_embed + self.type_embed[1].view(1, 1, -1)
        )
        if text_tokens is not None:
            text_tokens = self.text_proj(text_tokens.to(device=source.device, dtype=self.text_proj[1].weight.dtype))
            text_tokens = text_tokens.to(dtype=source_tokens.dtype)
            if self.text_mode == "in_joint_attn":
                num_text = int(text_tokens.shape[1])
                if num_text > self.t5_max_length:
                    raise ValueError(
                        f"text token length {num_text} exceeds t5_max_length={self.t5_max_length}"
                    )
                text_tokens = (
                    text_tokens
                    + self.text_pos_embed[:, :num_text].to(dtype=text_tokens.dtype)
                    + self.type_embed[2].view(1, 1, -1).to(dtype=text_tokens.dtype)
                )
        for block in self.blocks:
            source_tokens, target_tokens = block(source_tokens, target_tokens, text_tokens, text_mask, cond)
        return self.target_out_proj(self.target_out_norm(target_tokens))


def deterministic_ddim_flow_sample(
    model_fn: Callable[[Tensor, Tensor], Tensor],
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    sampling_steps: int = 16,
    t_min: float = 1e-4,
    eps_min: float = 1e-6,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    z = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    schedule = torch.linspace(1.0, float(t_min), sampling_steps + 1, device=device, dtype=dtype)
    x0_hat = z
    for step_idx in range(sampling_steps):
        t_value = schedule[step_idx]
        t_next = schedule[step_idx + 1]
        t_batch = t_value.expand(shape[0]).to(dtype=torch.float32)
        x0_hat = model_fn(z, t_batch)
        eps_hat = (z - (1.0 - t_value) * x0_hat) / torch.clamp(t_value, min=eps_min)
        z = (1.0 - t_next) * x0_hat + t_next * eps_hat
    return x0_hat


def _load_stats(path: str, shape_last_dim: int, *, key_prefix: str) -> Tuple[Tensor, Tensor]:
    if not path:
        mean = torch.zeros(1, 1, shape_last_dim, dtype=torch.float32)
        std = torch.ones(1, 1, shape_last_dim, dtype=torch.float32)
        return mean, std
    resolved = _resolve_repo_relative(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Diffusion latent-stats file not found: {path!r} (resolved to {resolved}). "
            "Generate it from a trained Stage-2 encoder with `python scripts/compute_diffusion_stats.py "
            "--stage2-ckpt <ckpt> --config configs/diffusion_q4.yaml --out configs/stats/q4_stats.json`, "
            "or set target_stats_path/source_stats_path to '' for identity normalization "
            "(development/smoke only — does not reproduce the paper recipe)."
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    mean = torch.as_tensor(payload[f"{key_prefix}_mean"], dtype=torch.float32).reshape(1, 1, shape_last_dim)
    std = torch.as_tensor(payload[f"{key_prefix}_std"], dtype=torch.float32).reshape(1, 1, shape_last_dim)
    return mean, std.clamp_min(1e-6)


def _normalize_latent(latent: Tensor, mean: Tensor, std: Tensor, scale_divisor: float) -> Tensor:
    return ((latent - mean.to(device=latent.device, dtype=latent.dtype)) / std.to(device=latent.device, dtype=latent.dtype)) / float(
        scale_divisor
    )


def _unnormalize_latent(latent: Tensor, mean: Tensor, std: Tensor, scale_divisor: float) -> Tensor:
    return latent * float(scale_divisor) * std.to(device=latent.device, dtype=latent.dtype) + mean.to(
        device=latent.device,
        dtype=latent.dtype,
    )


class WanTcLatentDiffusionLAM(LightningModule):
    _CHECKPOINT_EXCLUDED_PREFIXES = ("tc_teacher.", "text_conditioner.")

    def __init__(
        self,
        model_dim: int = 512,
        num_heads: int = 8,
        dit_blocks: int = 12,
        source_tokens: int = 512,
        target_tokens: int = 4,
        source_dim: int = 48,
        target_dim: int = 32,
        text_dim: int = 768,
        dropout: float = 0.0,
        text_gate_init: float = 0.0,
        t_min: float = 1e-4,
        text_drop_p: float = 0.15,
        source_token_drop_p: float = 0.0,
        sampling_steps: int = 16,
        guidance_scale: float = 1.0,
        tc_target_mode: str = "first",
        source_mode: str = "history_5frame_wan2",
        history_num_frames: int = 5,
        text_mode: str = "in_joint_attn",
        stage2_checkpoint: str = DEFAULT_STAGE2_CHECKPOINT,
        teacher_kind: str = "stage2",
        wan_vae_model_path: str = "",
        t5_model_id: str = "t5-base",
        t5_model_path: str = "",
        t5_max_length: int = 64,
        text_cache_size: int = 4096,
        target_stats_path: str = "",
        source_stats_path: str = "",
        latent_scale_divisor: float = 1.0,
        log_interval: int = 10,
        log_path: str = "ditar_encoder/logs",
        task_name: str = "wan_tc_latent_diffusion_v1_1",
        optimizer: OptimizerCallable = AdamW,
        use_wandb: bool = False,
    ) -> None:
        super().__init__()
        normalized_target_mode = str(tc_target_mode or "first").strip().lower()
        if normalized_target_mode not in {"first", "last"}:
            raise ValueError("tc_target_mode must be 'first' or 'last'")
        normalized_source_mode = _normalize_source_mode(source_mode)
        self.core = WanTcLatentDiffusionCore(
            model_dim=model_dim,
            num_heads=num_heads,
            dit_blocks=dit_blocks,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            source_dim=source_dim,
            target_dim=target_dim,
            text_dim=text_dim,
            dropout=dropout,
            text_gate_init=text_gate_init,
            text_mode=text_mode,
            t5_max_length=t5_max_length,
        )
        self.teacher_kind = str(teacher_kind or "stage2").strip().lower()
        if self.teacher_kind not in {"stage1", "stage2"}:
            raise ValueError(
                f"teacher_kind must be 'stage1' or 'stage2', got {teacher_kind!r}"
            )
        self.tc_teacher = self._load_tc_teacher(
            stage2_checkpoint=stage2_checkpoint,
            wan_vae_model_path=wan_vae_model_path or None,
        )
        self.tc_teacher.requires_grad_(False)
        self.tc_teacher.eval()

        from ditar_encoder.encoders.vjepa2_fullgrid_common import FrozenT5TextConditioner

        self.text_conditioner = FrozenT5TextConditioner(
            t5_model_id=t5_model_id,
            t5_model_path=t5_model_path or None,
            t5_max_length=t5_max_length,
            text_cache_size=text_cache_size,
        )
        self.text_conditioner.requires_grad_(False)
        self.text_conditioner.eval()

        target_mean, target_std = _load_stats(target_stats_path, target_dim, key_prefix="target")
        source_mean, source_std = _load_stats(source_stats_path, source_dim, key_prefix="source")
        self.register_buffer("target_mean", target_mean, persistent=True)
        self.register_buffer("target_std", target_std, persistent=True)
        self.register_buffer("source_mean", source_mean, persistent=True)
        self.register_buffer("source_std", source_std, persistent=True)
        self.latent_scale_divisor = float(latent_scale_divisor)
        if self.latent_scale_divisor <= 0:
            raise ValueError("latent_scale_divisor must be positive")

        self.t_min = float(t_min)
        self.text_drop_p = float(text_drop_p)
        self.source_token_drop_p = float(source_token_drop_p)
        self.sampling_steps = int(sampling_steps)
        self.guidance_scale = float(guidance_scale)
        self.tc_target_mode = normalized_target_mode
        self.source_mode = normalized_source_mode
        self.history_num_frames = int(history_num_frames)
        if self.history_num_frames <= 0:
            raise ValueError("history_num_frames must be positive")
        self.stage2_checkpoint = str(stage2_checkpoint)
        self.log_interval = int(log_interval)
        self.log_path = str(log_path)
        self.task_name = str(task_name)
        self.optimizer = optimizer
        self.use_wandb = bool(use_wandb)
        self.save_hyperparameters(ignore=["optimizer"])

    @staticmethod
    def _checkpoint_hparams(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        return checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, Mapping) else {}

    @staticmethod
    def _checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            return checkpoint["state_dict"]  # type: ignore[return-value]
        return checkpoint  # type: ignore[return-value]

    @staticmethod
    def _load_teacher_state(model: nn.Module, core_state: Mapping[str, Tensor], frozen_prefixes: Tuple[str, ...]):
        """Load the encoder teacher with key validation.

        The frozen frontends (`frozen_prefixes`, e.g. the WAN-VAE `teacher.*` and the
        Stage-1 `t5_conditioner.*`) are intentionally absent from `core_state` and are
        expected to be missing. Any OTHER missing key, or any unexpected key, signals a
        topology/hparam mismatch (e.g. `separate_tc_encoder` or `query_num_tokens` not
        recorded in the checkpoint) that would otherwise leave the frozen teacher silently
        emitting garbage target latents (see audit A3). Unexpected keys are fatal; missing
        non-frozen keys are surfaced as a loud warning.
        """
        result = model.load_state_dict(core_state, strict=False)
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        missing = [k for k in (getattr(result, "missing_keys", []) or []) if not k.startswith(frozen_prefixes)]
        if unexpected:
            raise RuntimeError(
                "Diffusion teacher reload: checkpoint carries "
                f"{len(unexpected)} unexpected key(s) the constructed teacher does not have "
                "(likely a topology/hparam mismatch — e.g. separate_tc_encoder or query_num_tokens "
                f"missing from the checkpoint's hyper_parameters). First few: {unexpected[:8]}"
            )
        if missing:
            warnings.warn(
                f"Diffusion teacher reload: {len(missing)} teacher parameter(s) were absent from the "
                f"checkpoint and remain at initialization (frozen {frozen_prefixes} prefixes excluded). "
                f"First few: {missing[:8]}",
                RuntimeWarning,
                stacklevel=2,
            )
        return model

    def _load_tc_teacher(self, stage2_checkpoint: str, wan_vae_model_path: Optional[str]):
        checkpoint_path = _resolve_checkpoint(stage2_checkpoint)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        hparams = self._checkpoint_hparams(checkpoint)
        subfolder = hparams.get("wan_vae_subfolder", "vae")

        if self.teacher_kind == "stage1":
            from ditar_encoder.lam.wan_vae_queryset_predictive_t5_lam import WanVaeQuerysetPredictiveT5Stage1LatentActionModel
            model = WanVaeQuerysetPredictiveT5Stage1LatentActionModel(
                bottleneck_dim=hparams.get("bottleneck_dim", 32),
                bottleneck_hidden_dim=hparams.get("bottleneck_hidden_dim", 128),
                model_dim=hparams.get("model_dim", 768),
                num_heads=hparams.get("num_heads", 12),
                enc_blocks=hparams.get("enc_blocks", 4),
                dec_blocks=hparams.get("dec_blocks", 12),
                bottleneck_regularizer=hparams.get("bottleneck_regularizer", "kl"),
                bottleneck_post_norm=hparams.get("bottleneck_post_norm", True),
                lam_dropout=hparams.get("lam_dropout", 0.1),
                encoder_arch=hparams.get("encoder_arch", "queryset_cross_v1"),
                decoder_arch=hparams.get("decoder_arch", "source_query_self_attn_v1"),
                query_num_tokens=hparams.get("query_num_tokens", 4),
                query_blocks=hparams.get("query_blocks", 8),
                kl_scope=hparams.get("kl_scope", "target"),
                wan_vae_model_id=hparams.get("wan_vae_model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
                wan_vae_model_path=_resolve_model_snapshot(wan_vae_model_path or hparams.get("wan_vae_model_path"), subfolder),
                wan_vae_subfolder=subfolder,
                wan_vae_dtype=hparams.get("wan_vae_dtype", "float32"),
                wan_vae_sample_mode=hparams.get("wan_vae_sample_mode", "mode"),
                wan_vae_normalize_latents=hparams.get("wan_vae_normalize_latents", True),
                teacher_input_resolution=hparams.get("teacher_input_resolution", 256),
                num_frames=hparams.get("num_frames", 5),
                t5_model_id=hparams.get("t5_model_id", "t5-base"),
                t5_model_path=hparams.get("t5_model_path", None),
                t5_max_length=hparams.get("t5_max_length", 64),
                text_cache_size=hparams.get("text_cache_size", 0),
                max_decoder_text_tokens=hparams.get("max_decoder_text_tokens", 64),
            )
            core_state: Dict[str, Tensor] = {}
            for key, value in self._checkpoint_state(checkpoint).items():
                if key.startswith("lam."):
                    stripped = key[len("lam.") :]
                    if not stripped.startswith("teacher.") and not stripped.startswith("t5_conditioner."):
                        core_state[stripped] = value
                elif not key.startswith("teacher.") and not key.startswith("t5_conditioner."):
                    core_state[key] = value
            return self._load_teacher_state(model, core_state, ("teacher.", "t5_conditioner."))

        from ditar_encoder.lam.wan_vae_queryset_predictive_lam import WanVaeQuerysetPredictiveStage2LatentActionModel
        model = WanVaeQuerysetPredictiveStage2LatentActionModel(
            bottleneck_dim=hparams.get("bottleneck_dim", 32),
            bottleneck_hidden_dim=hparams.get("bottleneck_hidden_dim", 128),
            model_dim=hparams.get("model_dim", 768),
            num_heads=hparams.get("num_heads", 12),
            enc_blocks=hparams.get("enc_blocks", 4),
            dec_blocks=hparams.get("dec_blocks", 12),
            bottleneck_regularizer=hparams.get("bottleneck_regularizer", "kl"),
            bottleneck_post_norm=hparams.get("bottleneck_post_norm", True),
            lam_dropout=hparams.get("lam_dropout", 0.1),
            encoder_arch=hparams.get("encoder_arch", "queryset_cross_v1"),
            decoder_arch=hparams.get("decoder_arch", "source_query_self_attn_v1"),
            query_num_tokens=hparams.get("query_num_tokens", 64),
            query_blocks=hparams.get("query_blocks", 8),
            kl_scope=hparams.get("kl_scope", "target"),
            wan_vae_model_id=hparams.get("wan_vae_model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
            wan_vae_model_path=_resolve_model_snapshot(wan_vae_model_path or hparams.get("wan_vae_model_path"), subfolder),
            wan_vae_subfolder=subfolder,
            wan_vae_dtype=hparams.get("wan_vae_dtype", "float32"),
            wan_vae_sample_mode=hparams.get("wan_vae_sample_mode", "mode"),
            wan_vae_normalize_latents=hparams.get("wan_vae_normalize_latents", True),
            teacher_input_resolution=hparams.get("teacher_input_resolution", 256),
            num_frames=hparams.get("num_frames", 17),
            max_decoder_text_tokens=hparams.get("max_decoder_text_tokens", 256),
            separate_tc_encoder=hparams.get("separate_tc_encoder", False),
            init_tc_from_ti=hparams.get("init_tc_from_ti", False),
            tc_bootstrap_freeze=hparams.get("tc_bootstrap_freeze", False),
            freeze_ti_branch=hparams.get("freeze_ti_branch", True),
            ti_decoder_dropout_p=hparams.get("ti_decoder_dropout_p", 0.0),
        )
        core_state: Dict[str, Tensor] = {}
        for key, value in self._checkpoint_state(checkpoint).items():
            if key.startswith("lam."):
                stripped = key[len("lam.") :]
                if not stripped.startswith("teacher."):
                    core_state[stripped] = value
            elif not key.startswith("teacher."):
                core_state[key] = value
        return self._load_teacher_state(model, core_state, ("teacher.",))

    def _normalize_source(self, source: Tensor) -> Tensor:
        return _normalize_latent(source, self.source_mean, self.source_std, self.latent_scale_divisor)

    def _normalize_target(self, target: Tensor) -> Tensor:
        return _normalize_latent(target, self.target_mean, self.target_std, self.latent_scale_divisor)

    def unnormalize_target(self, target: Tensor) -> Tensor:
        return _unnormalize_latent(target, self.target_mean, self.target_std, self.latent_scale_divisor)

    def _select_target_tc_latent(self, tc_latents: Tensor) -> Tensor:
        future = tc_latents[:, 1:]
        if self.tc_target_mode == "first":
            return future[:, 0].contiguous()
        return future[:, -1].contiguous()

    def _split_history_target_videos(self, videos: Tensor) -> Tuple[Tensor, Tensor]:
        """For history mode: input videos is the full [history + shared + target]
        window of length (history_num_frames + target_window - 1). The shared
        anchor frame s sits at index (history_num_frames - 1). The future-target
        window starts at that anchor.
        """
        history_len = int(self.history_num_frames)
        target_window = int(self.tc_teacher.teacher.teacher_num_frames)
        expected_total = history_len + target_window - 1
        if videos.shape[1] != expected_total:
            raise ValueError(
                f"source_mode={self.source_mode!r} expects num_frames={expected_total} "
                f"(history={history_len} + target={target_window} - 1 shared), "
                f"got videos with T={videos.shape[1]}"
            )
        history = videos[:, :history_len].contiguous()
        target_videos = videos[:, history_len - 1 : history_len - 1 + target_window].contiguous()
        return history, target_videos

    @torch.no_grad()
    def _extract_online_latents(self, videos: Tensor) -> Tuple[Tensor, Tensor]:
        self.tc_teacher.eval()
        if self.source_mode == "first_frame_wan_l0":
            target_videos = videos
            teacher_latents_full = self.tc_teacher.teacher.encode(target_videos)
            source_wan_l0 = self.tc_teacher.teacher.encode(target_videos[:, :1])[:, 0].contiguous()
            source = source_wan_l0
        else:
            history_videos, target_videos = self._split_history_target_videos(videos)
            teacher_latents_full = self.tc_teacher.teacher.encode(target_videos)
            history_latents = self.tc_teacher.teacher.encode(history_videos)
            b, t_lat, n_tok, z_dim = history_latents.shape
            source = history_latents.reshape(b, t_lat * n_tok, z_dim).contiguous()
        reduced_context = self.tc_teacher._encode_teacher_context(teacher_latents_full)
        expected_steps = int(getattr(self.tc_teacher, "reduced_num_steps", 0) or 0)
        if expected_steps and int(reduced_context.shape[1]) != expected_steps:
            raise ValueError(
                f"source_mode={self.source_mode!r}: the teacher target window produced "
                f"{int(reduced_context.shape[1])} latent context step(s) but the Stage-2 teacher expects "
                f"{expected_steps} (it was trained with a different num_frames). The shipped data uses "
                f"num_frames {videos.shape[1]}; for the canonical model use source_mode='history_5frame_wan2' "
                f"with data.num_frames = history_num_frames + teacher_num_frames - 1."
            )
        if self.teacher_kind == "stage1":
            ti_encoded = self.tc_teacher.ti_encoder(reduced_context, text_tokens=None)
            tc_latent = self.tc_teacher.ti_bottleneck(ti_encoded)["latent"]
        else:
            # Single source of truth on the encoder (audit B2) — no private reach-through.
            tc_latent = self.tc_teacher.task_relevant_from_context(reduced_context)
        target_tc_latent = self._select_target_tc_latent(tc_latent)
        return source.float(), target_tc_latent.float()

    @torch.no_grad()
    def encode_task_relevant(self, videos: Tensor) -> Tensor:
        """Public encoder API: a clip -> the task-relevant latent ``z_rel`` the DiT trains on.

        ``videos`` is the same window training consumes (``history_num_frames`` history frames +
        the teacher target window, sharing one anchor frame, for the history source modes).
        Returns ``z_rel`` of shape ``[B, target_tokens, target_dim]`` (4 x 32 for q4) — the exact
        tensor the downstream reward / RL repo consumes. Uses the frozen-teacher extraction so the
        encoding matches training; no trained diffusion weights are required.
        """
        self.eval()
        _, z_rel = self._extract_online_latents(videos.to(self.device))
        return z_rel

    def _drop_instructions(self, instructions: Sequence[str]) -> Sequence[str]:
        if not self.training or self.text_drop_p <= 0:
            return instructions
        dropped = []
        for instruction in instructions:
            if torch.rand((), device=self.device).item() < self.text_drop_p:
                dropped.append("NULL")
            else:
                dropped.append(instruction)
        return dropped

    def _encode_text(self, instructions: Sequence[str]) -> Tuple[Tensor, Tensor]:
        dropped = self._drop_instructions(instructions)
        with torch.no_grad():
            text_tokens, text_mask = self.text_conditioner(dropped)
        return text_tokens.float(), text_mask

    def _maybe_drop_source_tokens(self, source: Tensor) -> Tensor:
        if not self.training or self.source_token_drop_p <= 0:
            return source
        keep = torch.rand(source.shape[:2], device=source.device, dtype=source.dtype) >= self.source_token_drop_p
        return source * keep.unsqueeze(-1)

    def shared_step(self, batch: Mapping[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]:
        videos = batch["videos"].to(self.device)
        source_wan_l0, target_tc_latent = self._extract_online_latents(videos)
        source = self._maybe_drop_source_tokens(self._normalize_source(source_wan_l0))
        x0 = self._normalize_target(target_tc_latent)
        eps = torch.randn_like(x0)
        timesteps = torch.empty(x0.shape[0], device=x0.device).uniform_(self.t_min, 1.0)
        z_t = (1.0 - timesteps.view(-1, 1, 1)) * x0 + timesteps.view(-1, 1, 1) * eps
        text_tokens, text_mask = self._encode_text(batch["task_instruction"])
        x0_hat = self.core(z_t, source, text_tokens, text_mask, timesteps)
        error = x0_hat.float() - x0.float()
        loss = error.pow(2).mean()
        target_rms = x0.float().pow(2).mean().sqrt().clamp_min(1e-8)
        logs = {
            "loss": loss,
            "rmse": loss.sqrt(),
            "relative_rmse": loss.sqrt() / target_rms,
            "mae": error.abs().mean(),
            "cosine": F.cosine_similarity(x0_hat.flatten(1).float(), x0.flatten(1).float(), dim=-1).mean(),
            "target_rms": target_rms,
            "pred_rms": x0_hat.float().pow(2).mean().sqrt(),
            "source_rms": source.float().pow(2).mean().sqrt(),
            "t_mean": timesteps.mean(),
        }
        return loss, logs

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        loss, logs = self.shared_step(batch)
        batch_size = int(batch["videos"].shape[0])
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(
            {f"train/{key}": value for key, value in logs.items()},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        loss, logs = self.shared_step(batch)
        batch_size = int(batch["videos"].shape[0])
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(
            {f"val/{key}": value for key, value in logs.items()},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    @torch.no_grad()
    def sample(
        self,
        source_wan_l0: Tensor,
        instructions: Sequence[str],
        sampling_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        self.eval()
        source = self._normalize_source(source_wan_l0.to(self.device).float())
        text_tokens, text_mask = self.text_conditioner(instructions)
        text_tokens = text_tokens.float()
        null_tokens = null_mask = None
        scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        if scale != 1.0:
            null_tokens, null_mask = self.text_conditioner(["NULL"] * source.shape[0])
            null_tokens = null_tokens.float()

        def model_fn(z: Tensor, t_batch: Tensor) -> Tensor:
            cond = self.core(z, source, text_tokens, text_mask, t_batch)
            if scale == 1.0:
                return cond
            uncond = self.core(z, source, null_tokens, null_mask, t_batch)
            return uncond + scale * (cond - uncond)

        x0 = deterministic_ddim_flow_sample(
            model_fn,
            (source.shape[0], self.core.target_tokens, self.core.target_dim),
            device=source.device,
            dtype=source.dtype,
            sampling_steps=self.sampling_steps if sampling_steps is None else int(sampling_steps),
            t_min=self.t_min,
            generator=generator,
        )
        return self.unnormalize_target(x0)

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(param for param in self.parameters() if param.requires_grad)

    @classmethod
    def _is_excluded_checkpoint_key(cls, key: str) -> bool:
        return key.startswith(cls._CHECKPOINT_EXCLUDED_PREFIXES)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if self._is_excluded_checkpoint_key(key):
                del state[key]
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        if strict:
            missing = [key for key in incompatible.missing_keys if not self._is_excluded_checkpoint_key(key)]
            unexpected = [key for key in incompatible.unexpected_keys if not self._is_excluded_checkpoint_key(key)]
            if missing or unexpected:
                raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")
        return incompatible
