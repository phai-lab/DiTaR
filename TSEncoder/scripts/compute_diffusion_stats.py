"""Compute the diffusion latent-normalization stats (`q4_stats.json`).

The TC-latent diffusion model (Sec 3.3) normalizes its WAN-source and TC-target latents
with per-channel mean/std computed over the training data. Those stats are **data-derived**
and therefore not shipped; this script reproduces them from a trained Stage-2 encoder and the
diffusion config, using the *exact* frozen-teacher extraction path the model trains on
(`WanTcLatentDiffusionLAM._extract_online_latents`).

Output JSON (consumed by `_load_stats` in the diffusion model) contains per-channel vectors:

    {"target_mean": [..32..], "target_std": [..32..],
     "source_mean": [..48..], "source_std": [..48..]}

Usage:
    python scripts/compute_diffusion_stats.py \
        --config configs/diffusion_q4.yaml \
        --stage2-ckpt $DITAR_STAGE2_CKPT \
        --out configs/stats/q4_stats.json \
        --max-batches 200

Set DITAR_DATA_ROOT (and any other ${...} the config references) before running.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import yaml


def _expandvars(obj: Any) -> Any:
    """Recursively expand ${ENV} / ~ in every string in a loaded YAML config."""
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, dict):
        return {key: _expandvars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_expandvars(value) for value in obj]
    return obj


class _ChannelStats:
    """Streaming per-channel (last-dim) mean/std over all leading dims, in float64."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.count = 0
        self.sum = torch.zeros(self.dim, dtype=torch.float64)
        self.sumsq = torch.zeros(self.dim, dtype=torch.float64)

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.reshape(-1, tensor.shape[-1]).to(dtype=torch.float64, device="cpu")
        self.count += int(flat.shape[0])
        self.sum += flat.sum(dim=0)
        self.sumsq += flat.pow(2).sum(dim=0)

    def finalize(self) -> tuple[list[float], list[float]]:
        if self.count == 0:
            raise RuntimeError("no samples accumulated — check the data root / dataloader")
        mean = self.sum / self.count
        var = (self.sumsq / self.count) - mean.pow(2)
        std = var.clamp_min(0.0).sqrt().clamp_min(1e-6)
        return mean.tolist(), std.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="diffusion config (model + data sections)")
    ap.add_argument("--stage2-ckpt", default="", help="override model.stage2_checkpoint (the encoder teacher)")
    ap.add_argument("--out", required=True, help="output JSON path (e.g. configs/stats/q4_stats.json)")
    ap.add_argument("--max-batches", type=int, default=200, help="number of train batches to accumulate over")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Import after arg-parsing so `--help` works without the heavy deps installed.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ditar_encoder.diffusion.wan_tc_latent_diffusion_model import WanTcLatentDiffusionLAM
    from ditar_encoder.data.weighted_multi_manifest_datamodule import WeightedMultiManifestDataModule

    cfg = _expandvars(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    model_cfg = dict(cfg.get("model", {}))
    data_cfg = dict(cfg.get("data", {}))

    # The model takes an optimizer *callable*; the YAML stores a class_path dict for the CLI.
    # Strip it (we never optimize here) and force identity stats so we measure raw latents.
    model_cfg.pop("optimizer", None)
    model_cfg["target_stats_path"] = ""
    model_cfg["source_stats_path"] = ""
    if args.stage2_ckpt:
        model_cfg["stage2_checkpoint"] = os.path.expanduser(os.path.expandvars(args.stage2_ckpt))

    model = WanTcLatentDiffusionLAM(**model_cfg).to(args.device).eval()
    datamodule = WeightedMultiManifestDataModule(**data_cfg)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    source_stats = _ChannelStats(int(model.core.source_dim))
    target_stats = _ChannelStats(int(model.core.target_dim))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            source, target = model._extract_online_latents(batch["videos"].to(args.device))
            source_stats.update(source)
            target_stats.update(target)
            if (batch_idx + 1) % 25 == 0:
                print(f"[compute_diffusion_stats] {batch_idx + 1}/{args.max_batches} batches", flush=True)

    source_mean, source_std = source_stats.finalize()
    target_mean, target_std = target_stats.finalize()
    payload = {
        "target_mean": target_mean,
        "target_std": target_std,
        "source_mean": source_mean,
        "source_std": source_std,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[compute_diffusion_stats] wrote {out_path} "
        f"(source_dim={source_stats.dim} over {source_stats.count} rows, "
        f"target_dim={target_stats.dim} over {target_stats.count} rows)"
    )


if __name__ == "__main__":
    main()
