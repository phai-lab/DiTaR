"""Calibrate (mean, std) of the WAN TC diffusion reward on expert demos.

Runs ``WanTcLatentDiffusionReward.compute_Et`` on a random sample of expert
trajectories per benchmark, derives all four reward variants from each
trajectory's E_t, and writes a per-benchmark JSON with z-score stats:

    {
      "exp":         {"mean": ..., "std": ..., "n_steps": N},
      "neg_e":       {"mean": ..., "std": ..., "n_steps": N},
      "td_residual": {"mean": ..., "std": ..., "n_steps": N},
      "td_exp":      {"mean": ..., "std": ..., "n_steps": N},
      "metadata": {...}
    }

At training time the wrapper reads this file and outputs ``(r - μ) / σ`` so
the reward scale stays fixed regardless of how exp(-E_t) compresses.

Demo sources (all mp4 + per-episode instructions):
  - maniskill:  <repo>/data/maniskill/episode_instruction_multiview_v2.jsonl
  - metaworld:  data/metaworldv3/episode_instructions.jsonl

Usage:
    python ours/scripts/calibrate_reward_stats.py --benchmark maniskill --n_demos 64 --out ours/reward_stats/maniskill.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

OURS_ROOT = Path(__file__).resolve().parents[1]    # <repo>/ours
REPO_ROOT = OURS_ROOT.parent                        # <repo>
for p in (str(REPO_ROOT), str(REPO_ROOT / "TSEncoder")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("_CHECK_PEFT", "0")

from ours.reward_models import WanTcLatentDiffusionReward  # noqa: E402

# WAN-TC reward checkpoints / sub-models. Provided via env vars (see
# CHECKPOINTS.md); HF repo ids are auto-downloaded for the VAE / T5.
DEFAULT_CKPT = os.environ.get("WAN_TC_CKPT", "")
DEFAULT_STAGE2 = os.environ.get("WAN_TC_STAGE2_CKPT", "")
DEFAULT_STATS = os.environ.get("WAN_TC_STATS", "")
DEFAULT_WAN_VAE = os.environ.get("WAN_VAE_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DEFAULT_T5 = os.environ.get("T5_PATH", "t5-base")

# Expert-demo manifests. These datasets are NOT shipped with the repo; place
# them under <repo>/data/ (or override the paths) to re-run calibration. The
# precomputed stats under ours/reward_stats/ let you train without this step.
MANISKILL_JSONL = REPO_ROOT / "data/maniskill/episode_instruction_multiview_v2.jsonl"
METAWORLD_JSONL = REPO_ROOT / "data/metaworldv3/episode_instructions.jsonl"
METAWORLD_ROOT = REPO_ROOT / "data/metaworldv3"

# Roots scanned for random-policy rollouts (eval_video/0.mp4 from existing RL workspaces).
RL_OUTPUTS_ROOT = OURS_ROOT / "outputs/rl"

# Per-benchmark task-dir tokens that identify candidate workspaces.
BENCH_TASK_TOKENS = {
    "maniskill": ["PickCube-v1","PushCube-v1","PullCube-v1","PokeCube-v1","RollBall-v1",
                  "StackCube-v1","LiftPegUpright-v1","PushT-v1","PegInsertionSide-v1"],
    "metaworld": ["button-press-v3","coffee-pull-v3","coffee-push-v3","drawer-open-v3","faucet-open-v3",
                  "handle-pull-v3","lever-pull-v3","window-close-v3","window-open-v3","door-open-v3",
                  "door-close-v3","drawer-close-v3","faucet-close-v3","handle-press-v3","dial-turn-v2"],
}

VARIANTS = ("exp", "neg_e", "td_residual", "td_exp")


def collect_demo_records(benchmark: str, n: int, seed: int) -> list[dict]:
    """Return up to n records: each {"video": absolute-path, "instruction": str}."""
    rng = random.Random(seed)
    records: list[dict] = []
    if benchmark == "maniskill":
        recs = [json.loads(l) for l in MANISKILL_JSONL.open()]
        rng.shuffle(recs)
        for r in recs:
            video = r.get("video_left_camera")
            instr = r.get("instruction", "")
            if not video or not Path(video).exists():
                continue
            records.append({"video": video, "instruction": instr})
            if len(records) >= n:
                break
    elif benchmark == "metaworld":
        recs = [json.loads(l) for l in METAWORLD_JSONL.open()]
        rng.shuffle(recs)
        for r in recs:
            rel = r.get("video")
            instr = r.get("instruction", "")
            if not rel:
                continue
            video = str(METAWORLD_ROOT / rel)
            if not Path(video).exists():
                continue
            records.append({"video": video, "instruction": instr})
            if len(records) >= n:
                break
    else:
        raise ValueError(f"unknown benchmark={benchmark!r}")
    return records


def collect_random_rollout_records(benchmark: str, n: int, seed: int) -> list[dict]:
    """Find eval_video/0.mp4 files from existing RL workspaces (= random-policy
    rollouts, since 0.mp4 is recorded BEFORE the first gradient step). Used to
    build the "mixed" stat branch where μ/σ anchor on random-policy trajectory
    distribution rather than demo distribution.

    For MS/MW we have many such files across existing workspaces. Returns
    up to n records.
    """
    rng = random.Random(seed + 7919)
    tokens = BENCH_TASK_TOKENS.get(benchmark, [])
    candidates: list[str] = []
    for tok in tokens:
        # search the RL workspace tree for eval_video/0.mp4 (random-rollout source)
        for root in [RL_OUTPUTS_ROOT]:
            if not root.exists(): continue
            for mp4 in root.glob(f"*/{tok}/*/*/eval_video/0.mp4"):
                if mp4.exists() and mp4.stat().st_size > 1000:
                    candidates.append(str(mp4))
    rng.shuffle(candidates)
    records = [{"video": c, "instruction": ""} for c in candidates[:n]]
    return records


def decode_mp4(path: str, max_frames: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames, axis=0) if frames else None


def to_input_tensor(frames: np.ndarray, size: int = 64) -> torch.Tensor:
    """Downsample demos to ``size`` (default 64×64) to **match the RL pipeline**:
    DrQ-v2 env wrappers always feed 64×64 obs, and our reward wrapper internally
    upsamples 64 → 256 (bilinear) before calling compute_Et. Calibration at the
    same 64×64 input level removes the resolution domain gap that was making
    (μ, σ) shift away from the RL trajectory distribution.
    """
    if frames.shape[1] != size or frames.shape[2] != size:
        h, w = frames.shape[1], frames.shape[2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        frames = frames[:, top:top + side, left:left + side]
        if frames.shape[1] != size:
            frames = np.stack(
                [cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames],
                axis=0,
            )
    arr = frames.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, T, size, size, 3]


def build_reward_model(args) -> WanTcLatentDiffusionReward:
    cfg = SimpleNamespace(
        ckpt_path=args.ckpt,
        stage2_ckpt_path=args.stage2,
        stats_path=args.stats,
        source_stats_path=args.stats,
        taskvae_root=os.environ.get("TASKVAE_ROOT", ""),
        hf_hub_root=os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        wan_vae_path=DEFAULT_WAN_VAE,
        t5_path=DEFAULT_T5,
        reward_type="exp",            # arbitrary; apply_variant handles all four
        mc_samples=args.mc_samples,
        tau_min=1e-2,
        tau_max=1.0,
        td_scale=1.0,
        td_step_gap=args.td_step_gap,
        internal_batch_size=args.batch_size,
        stride=args.stride,
        default_instruction="",
        use_expl_reward=False,
        expl_scale=0.0,
        use_env_reward=True,
        rm_model="wan_tc_diffusion",
        task_name="calibration",
        normalize_reward=False,        # CRITICAL: raw rewards for calibration
        reward_stats_path=None,
    )
    rm = WanTcLatentDiffusionReward(cfg)
    rm.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return rm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["maniskill", "metaworld"], required=True)
    parser.add_argument("--n_demos", type=int, default=64)
    parser.add_argument("--max_frames", type=int, default=200, help="hard cap per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mc_samples", type=int, default=2)
    parser.add_argument("--td_step_gap", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1, help="Window stride between adjacent reward windows in env steps")
    parser.add_argument("--input_size", type=int, default=64, help="Resize demo videos to this size before feeding reward model (must match RL pipeline)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--stage2", default=DEFAULT_STAGE2)
    parser.add_argument("--stats", default=DEFAULT_STATS, help="diffusion source/target stats JSON")
    parser.add_argument("--out", required=True, help="output JSON for reward stats")
    parser.add_argument("--include_mixed", action="store_true",
                        help="Also compute mixed (demo+random) stats; random rollouts auto-discovered from eval_video/0.mp4")
    parser.add_argument("--n_random", type=int, default=64, help="# random-rollout episodes for mixed branch")
    args = parser.parse_args()

    print(f"[calib] benchmark={args.benchmark}  n_demos={args.n_demos}  td_step_gap={args.td_step_gap}")

    t0 = time.monotonic()
    rm = build_reward_model(args)
    print(f"[calib] reward model loaded in {time.monotonic() - t0:.1f}s")
    print(f"[calib] geometry: source_mode={rm.source_mode}  window={rm.window}  anchor_offset={rm.anchor_offset}  "
          f"target_tokens={rm.target_tokens}  td_step_gap={rm.td_step_gap}")

    records = collect_demo_records(args.benchmark, args.n_demos, args.seed)
    print(f"[calib] sampled {len(records)} demo records")

    random_records = []
    if args.include_mixed:
        random_records = collect_random_rollout_records(args.benchmark, args.n_random, args.seed)
        print(f"[calib] sampled {len(random_records)} random-rollout records (from existing eval_video/0.mp4)")

    accum_demo = {v: [] for v in VARIANTS}
    accum_random = {v: [] for v in VARIANTS}
    raw_E_demo, raw_E_random = [], []
    n_demo_used = 0; n_random_used = 0

    def process_records(records_in, accum_dict, raw_E_list, tag):
        used = 0
        for i, rec in enumerate(records_in):
            t0 = time.monotonic()
            frames = decode_mp4(rec["video"], args.max_frames)
            if frames is None or frames.shape[0] < (rm.window + rm.td_step_gap):
                continue
            imgs = to_input_tensor(frames, size=args.input_size)
            rm.set_instruction(rec["instruction"] or "")
            Et = rm.compute_Et(imgs)
            raw_E_list.append(Et.detach().cpu().numpy())
            for v in VARIANTS:
                rm.reward_type = v
                r = rm.apply_variant(Et)
                accum_dict[v].append(r.detach().cpu().numpy())
            used += 1
            if (i + 1) % 5 == 0 or i == len(records_in) - 1:
                print(f"[calib] {tag} ep {i + 1}/{len(records_in)}  T={frames.shape[0]}  "
                      f"E_t mean={Et.mean().item():.4f} std={Et.std().item():.4f}  "
                      f"({time.monotonic() - t0:.1f}s)")
        return used

    n_demo_used = process_records(records, accum_demo, raw_E_demo, "demo  ")
    if args.include_mixed:
        n_random_used = process_records(random_records, accum_random, raw_E_random, "random")

    # Build nested JSON: {variant: {demo: {...}, mixed: {...}}, raw_E: {...}, metadata: {...}}
    def stats_of(arr):
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std() + 1e-12),
            "n_steps": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    out: dict = {}
    for v in VARIANTS:
        if not accum_demo[v]:
            out[v] = {"demo": {"mean": 0.0, "std": 1.0, "n_steps": 0}, "mixed": {}}
            continue
        demo_arr = np.concatenate(accum_demo[v], axis=0)
        out[v] = {"demo": stats_of(demo_arr), "mixed": {}}
        if accum_random[v]:
            rand_arr = np.concatenate(accum_random[v], axis=0)
            # mixed = concat of demo + random (so μ, σ across union)
            mix_arr = np.concatenate([demo_arr, rand_arr], axis=0)
            out[v]["mixed"] = stats_of(mix_arr)
            out[v]["random_only"] = stats_of(rand_arr)
        d = out[v]["demo"]
        mline = ""
        if out[v]["mixed"]:
            m = out[v]["mixed"]; r = out[v]["random_only"]
            mline = (f"  | random: μ={r['mean']:+.4f} σ={r['std']:.4f} N={r['n_steps']}"
                     f"  | mixed: μ={m['mean']:+.4f} σ={m['std']:.4f} N={m['n_steps']}")
        print(f"[calib] {v:>12s} demo: μ={d['mean']:+.6f} σ={d['std']:.6f} "
              f"range=[{d['min']:+.4f}, {d['max']:+.4f}] N={d['n_steps']}{mline}")

    if raw_E_demo:
        demo_E = np.concatenate(raw_E_demo, axis=0)
        out["raw_E"] = {"demo": stats_of(demo_E), "mixed": {}}
        if raw_E_random:
            rand_E = np.concatenate(raw_E_random, axis=0)
            mix_E = np.concatenate([demo_E, rand_E], axis=0)
            out["raw_E"]["mixed"] = stats_of(mix_E)
            out["raw_E"]["random_only"] = stats_of(rand_E)
        rE = out["raw_E"]["demo"]
        print(f"[calib] {'raw_E':>12s} demo: μ={rE['mean']:+.6f} σ={rE['std']:.6f} "
              f"range=[{rE['min']:+.4f}, {rE['max']:+.4f}] N={rE['n_steps']}")

    out["metadata"] = {
        "benchmark": args.benchmark,
        "n_demo_episodes": n_demo_used,
        "n_random_episodes": n_random_used,
        "include_mixed": args.include_mixed,
        "seed": args.seed,
        "ckpt": args.ckpt,
        "stage2": args.stage2,
        "td_step_gap": args.td_step_gap,
        "mc_samples": args.mc_samples,
        "source_mode": rm.source_mode,
        "window": rm.window,
        "anchor_offset": rm.anchor_offset,
        "target_tokens": rm.target_tokens,
        "target_dim": rm.target_dim,
        "input_resolution": 64,
        "stats_schema_version": 2,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[calib] wrote {out_path}")


if __name__ == "__main__":
    main()
