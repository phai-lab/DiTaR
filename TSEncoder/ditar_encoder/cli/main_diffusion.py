from lightning.pytorch.cli import LightningCLI

if __package__ in (None, ''):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ditar_encoder.diffusion.wan_tc_latent_diffusion_model import WanTcLatentDiffusionLAM
    from ditar_encoder.data.weighted_multi_manifest_datamodule import WeightedMultiManifestDataModule
else:
    from ditar_encoder.diffusion.wan_tc_latent_diffusion_model import WanTcLatentDiffusionLAM
    from ditar_encoder.data.weighted_multi_manifest_datamodule import WeightedMultiManifestDataModule


def main() -> None:
    LightningCLI(
        WanTcLatentDiffusionLAM,
        WeightedMultiManifestDataModule,
        seed_everything_default=42,
    )


if __name__ == '__main__':
    main()
