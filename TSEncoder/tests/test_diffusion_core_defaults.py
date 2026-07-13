"""A7 — the diffusion DiT defaults are the canonical q4 values, and the core runs a CPU
forward in `in_joint_attn` mode. No HF downloads (the DiT core has no frozen frontends)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from ditar_encoder.diffusion.wan_tc_latent_diffusion_model import (
    WanTcLatentDiffusionCore,
    _normalize_source_mode,
    _resolve_repo_relative,
    deterministic_ddim_flow_sample,
)


def test_core_defaults_are_canonical_q4():
    core = WanTcLatentDiffusionCore()
    assert core.source_tokens == 512
    assert core.target_tokens == 4
    assert core.source_dim == 48
    assert core.target_dim == 32
    assert core.text_mode == "in_joint_attn"
    # in_joint_attn => 3 token-type rows (source/target/text) + a text position table
    assert core.type_embed.shape[0] == 3
    assert hasattr(core, "text_pos_embed")


def test_core_forward_shapes():
    core = WanTcLatentDiffusionCore().eval()
    batch, text_len = 2, 5
    noisy_target = torch.randn(batch, 4, 32)
    source = torch.randn(batch, 512, 48)
    text_tokens = torch.randn(batch, text_len, 768)
    text_mask = torch.ones(batch, text_len, dtype=torch.bool)
    timesteps = torch.rand(batch)
    with torch.no_grad():
        out = core(noisy_target, source, text_tokens, text_mask, timesteps)
    assert out.shape == (batch, 4, 32)
    assert torch.isfinite(out).all()


def test_repo_relative_stats_resolution_uses_tse_root():
    resolved = _resolve_repo_relative("configs/stats/README.md")
    assert resolved.exists()
    assert resolved.parts[-3:] == ("configs", "stats", "README.md")


def test_source_mode_empty_value_falls_back_to_canonical_history_mode():
    assert _normalize_source_mode(None) == "history_5frame_wan2"
    assert _normalize_source_mode("") == "history_5frame_wan2"


def test_flow_sampler_accepts_generator_for_reproducibility():
    def model_fn(z, t_batch):
        return z + t_batch.reshape(-1, 1, 1) * 0.01

    shape = (2, 3, 4)
    gen_a = torch.Generator(device="cpu").manual_seed(123)
    gen_b = torch.Generator(device="cpu").manual_seed(123)
    gen_c = torch.Generator(device="cpu").manual_seed(124)

    out_a = deterministic_ddim_flow_sample(
        model_fn,
        shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampling_steps=4,
        generator=gen_a,
    )
    out_b = deterministic_ddim_flow_sample(
        model_fn,
        shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampling_steps=4,
        generator=gen_b,
    )
    out_c = deterministic_ddim_flow_sample(
        model_fn,
        shape,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampling_steps=4,
        generator=gen_c,
    )

    assert torch.allclose(out_a, out_b)
    assert not torch.allclose(out_a, out_c)
