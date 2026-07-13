"""A3 — the diffusion teacher reload validates keys instead of silently partial-loading.

`_load_teacher_state` raises on unexpected keys (a topology/hparam mismatch) and warns on
non-frozen missing keys, while excluding the intentionally-absent frozen prefixes. Tested in
isolation with toy modules (torch only, no HF downloads)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
import torch.nn as nn

from ditar_encoder.diffusion.wan_tc_latent_diffusion_model import WanTcLatentDiffusionLAM


def test_unexpected_key_raises():
    model = nn.Linear(2, 2)
    state = dict(model.state_dict())
    state["ghost.weight"] = torch.zeros(1)  # a key the model does not have
    with pytest.raises(RuntimeError, match="unexpected"):
        WanTcLatentDiffusionLAM._load_teacher_state(model, state, ("teacher.",))


def test_frozen_prefix_missing_is_allowed():
    model = nn.ModuleDict({"teacher": nn.Linear(2, 2), "core": nn.Linear(2, 2)})
    # Provide only core.*; teacher.* is intentionally absent (loaded separately) -> excluded.
    state = {k: v for k, v in model.state_dict().items() if k.startswith("core.")}
    assert WanTcLatentDiffusionLAM._load_teacher_state(model, state, ("teacher.",)) is model


def test_nonfrozen_missing_warns():
    model = nn.ModuleDict({"teacher": nn.Linear(2, 2), "core": nn.Linear(2, 2)})
    # Drop a non-frozen key (core.bias); teacher.* still excluded -> a warning, not a raise.
    state = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("teacher.") and k != "core.bias"
    }
    with pytest.warns(RuntimeWarning, match="remain at initialization"):
        WanTcLatentDiffusionLAM._load_teacher_state(model, state, ("teacher.",))
