"""B2 / V0 — the public encoder API exists and the entrypoint modules import cleanly.

Importing the encoder modules pulls transformers (T5) at module load, and the entrypoints pull
the full training stack; each dependency is `importorskip`-guarded so the suite still collects in
a minimal environment and runs in full in a GPU/deps environment."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")


def test_diffusion_lam_exposes_encode_task_relevant():
    from ditar_encoder.diffusion.wan_tc_latent_diffusion_model import WanTcLatentDiffusionLAM

    assert callable(getattr(WanTcLatentDiffusionLAM, "encode_task_relevant", None))


def test_stage2_lam_exposes_encode_api():
    pytest.importorskip("transformers")  # encoder module imports T5 at load time
    from ditar_encoder.lam.wan_vae_queryset_predictive_lam import WanVaeQuerysetPredictiveStage2LatentActionModel as Stage2LAM

    assert callable(getattr(Stage2LAM, "encode_task_relevant", None))
    assert callable(getattr(Stage2LAM, "task_relevant_from_context", None))


def test_stage1_forward_globals_are_resolved():
    pytest.importorskip("transformers")
    from ditar_encoder.lam.wan_vae_queryset_predictive_t5_lam import WanVaeQuerysetPredictiveT5Stage1LatentActionModel as Stage1LAM

    assert "gather_step_tokens" in Stage1LAM.forward.__globals__


def test_entrypoints_import():
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    pytest.importorskip("h5py")
    pytest.importorskip("einops")
    import importlib

    for module in (
        "ditar_encoder.cli.main_stage1",
        "ditar_encoder.cli.main_stage2",
        "ditar_encoder.cli.main_diffusion",
    ):
        assert importlib.import_module(module) is not None
