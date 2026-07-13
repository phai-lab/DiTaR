from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ditar_encoder.cli.checkpoint_init import WeightInitCLI as _WeightInitCLI
    from ditar_encoder.data.weighted_multi_manifest_datamodule import WeightedMultiManifestDataModule
    from ditar_encoder.models.wan_vae_queryset_predictive_stage2_model import WanVaeQuerysetPredictiveStage2LAM
else:
    from ditar_encoder.cli.checkpoint_init import WeightInitCLI as _WeightInitCLI
    from ditar_encoder.data.weighted_multi_manifest_datamodule import WeightedMultiManifestDataModule
    from ditar_encoder.models.wan_vae_queryset_predictive_stage2_model import WanVaeQuerysetPredictiveStage2LAM


class WeightInitCLI(_WeightInitCLI):
    """Stage-2 warm-start: same-architecture Phase-A -> Phase-B transition.

    A shape mismatch on a key present in both checkpoint and model is fatal
    here (the topologies must match), instead of being silently skipped.
    """

    warm_start_require_present_keys = True


def main() -> None:
    WeightInitCLI(
        WanVaeQuerysetPredictiveStage2LAM,
        WeightedMultiManifestDataModule,
        seed_everything_default=42,
    )


if __name__ == '__main__':
    main()
