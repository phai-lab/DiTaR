"""A2 — Stage-2 cannot be constructed without a Stage-1 checkpoint.

The guard fires in __init__ before any submodule (and before any HF download), so these run on
CPU with no network. They pin the fix for the previously-broken documented `stage2-warmup` command.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from ditar_encoder.models.wan_vae_queryset_predictive_stage2_model import WanVaeQuerysetPredictiveStage2LAM


def test_empty_stage_one_ckpt_raises():
    with pytest.raises(ValueError, match="stage_one_ckpt"):
        WanVaeQuerysetPredictiveStage2LAM(stage_one_ckpt="")


def test_unset_env_var_is_rejected():
    # An unset ${VAR} stays literal after expandvars, so the still-unresolved value is rejected
    # with the same actionable message rather than failing later with a cryptic file error.
    with pytest.raises(ValueError, match="stage_one_ckpt"):
        WanVaeQuerysetPredictiveStage2LAM(stage_one_ckpt="${DITAR_DEFINITELY_UNSET_XYZ}")
