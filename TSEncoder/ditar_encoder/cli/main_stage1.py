"""Stage-1 entry point for the v1 multi-dataset run (AgiBot + DROID + Bridge).

Differences vs the v0 main:

  * Uses `WeightedMultiManifestDataModule` (per-source target ratios + AgiBot
    inverse-frequency text sampling + per-source chunk_stride).
  * Adds an `--init_ckpt` CLI argument. When set AND `--ckpt_path` is empty,
    the LightningCLI subclass loads the checkpoint's `state_dict` into the
    model with `strict=False` before fit. Optimizer / scheduler / step
    counter remain freshly initialized — this is a warm-start, not a resume.
    On chain-tail jobs `--ckpt_path` is set (full Lightning resume) and
    `init_ckpt` is intentionally ignored.

The model class itself (`WanVaeQuerysetPredictiveT5Stage1LAM`) is reused
unchanged from v0 — the v1 run does not need any new constructor args.
"""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ditar_encoder.cli.checkpoint_init import STAGE1_REMAPPERS
    from ditar_encoder.cli.checkpoint_init import WeightInitCLI as _WeightInitCLI
    from ditar_encoder.data.weighted_multi_manifest_datamodule import WeightedMultiManifestDataModule
    from ditar_encoder.models.wan_vae_queryset_predictive_t5_stage1_model import WanVaeQuerysetPredictiveT5Stage1LAM
else:
    from ditar_encoder.cli.checkpoint_init import STAGE1_REMAPPERS
    from ditar_encoder.cli.checkpoint_init import WeightInitCLI as _WeightInitCLI
    from ditar_encoder.data.weighted_multi_manifest_datamodule import WeightedMultiManifestDataModule
    from ditar_encoder.models.wan_vae_queryset_predictive_t5_stage1_model import WanVaeQuerysetPredictiveT5Stage1LAM


class WeightInitCLI(_WeightInitCLI):
    """Stage-1 warm-start: cross-architecture init with endpoint/query remapping."""

    warm_start_remappers = STAGE1_REMAPPERS


def main() -> None:
    WeightInitCLI(
        WanVaeQuerysetPredictiveT5Stage1LAM,
        WeightedMultiManifestDataModule,
        seed_everything_default=42,
    )


if __name__ == '__main__':
    main()
