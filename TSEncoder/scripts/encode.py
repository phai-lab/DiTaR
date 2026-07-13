"""Encode a video clip into the task-relevant latent z_rel using a trained
Stage-2 (task-relevant) encoder checkpoint.

z_rel is the paper's task-relevant latent (Sec 3.2): `q` query tokens x
bottleneck dim (e.g. 4 x 32 for the canonical q4 model). It is the latent the
diffusion model (Sec 3.3) is trained on and the quantity a downstream reward /
RL repo consumes.

This reuses the frozen-teacher extraction path of the diffusion model
(`_extract_online_latents`) so the encoding exactly matches training. No trained
diffusion checkpoint is needed — only the Stage-2 encoder checkpoint.

Usage:
    python scripts/encode.py --stage2-ckpt /path/to/stage2/last.ckpt
    python scripts/encode.py --stage2-ckpt $DITAR_STAGE2_CKPT --videos clip.npy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make `ditar_encoder` importable when run as `python scripts/encode.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2-ckpt", required=True, help="Stage-2 encoder checkpoint (or ${DITAR_STAGE2_CKPT})")
    ap.add_argument("--videos", default="", help="optional .npy [B,T,3,H,W] in [-1,1]; random clip if omitted")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--frames", type=int, default=9, help="history_num_frames + target_window - 1 (5+5-1=9)")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--source-mode", default="history_5frame_wan2")
    ap.add_argument("--history-num-frames", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from ditar_encoder.diffusion.wan_tc_latent_diffusion_model import WanTcLatentDiffusionLAM

    lam = WanTcLatentDiffusionLAM(
        stage2_checkpoint=os.path.expandvars(args.stage2_ckpt),
        teacher_kind="stage2",
        source_mode=args.source_mode,
        tc_target_mode="first",
        history_num_frames=args.history_num_frames,
        target_stats_path="",
        source_stats_path="",
    )
    lam = lam.to(args.device).eval()

    if args.videos:
        videos = torch.from_numpy(np.load(args.videos)).float().to(args.device)
    else:
        videos = (torch.rand(args.batch, args.frames, 3, args.res, args.res, device=args.device) * 2.0 - 1.0)

    z_rel = lam.encode_task_relevant(videos)

    finite = bool(torch.isfinite(z_rel).all().item())
    print(f"[encode] z_rel  shape={tuple(z_rel.shape)} finite={finite} "
          f"mean={z_rel.mean().item():.4f} std={z_rel.std().item():.4f}")
    if not finite:
        raise SystemExit("[encode] z_rel contains non-finite values")
    print("[encode] OK")


if __name__ == "__main__":
    main()
