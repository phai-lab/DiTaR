"""B3 — the deduplicated `--init_ckpt` warm-start core.

`build_compatible_state` is the single key-matching/remapping routine both
entrypoints now share. These exercise its three branches (exact match, remap,
shape-skip) and the Stage-1 remappers, plus the Stage-2 same-architecture
policy (a shape mismatch on a present key must be detectable as fatal). Needs
torch (tensor copies/resampling); no lightning/HF and no downloads."""
import pytest

torch = pytest.importorskip("torch")

from ditar_encoder.cli.checkpoint_init import STAGE1_REMAPPERS, build_compatible_state, remap_query_count_embedding, remap_temporal_endpoint_embedding


def test_exact_match_kept_and_unknown_key_dropped():
    current = {"a": torch.zeros(2, 3), "b": torch.zeros(4)}
    incoming = {"a": torch.ones(2, 3), "ghost": torch.ones(9)}
    compatible, report = build_compatible_state(incoming, current)
    assert set(compatible) == {"a"}  # ghost is not in the model -> dropped
    assert torch.equal(compatible["a"], torch.ones(2, 3))
    assert report["skipped_shape"] == []


def test_shape_mismatch_without_remapper_is_skipped():
    current = {"w": torch.zeros(4, 4)}
    incoming = {"w": torch.zeros(8, 4)}  # present in both, wrong shape, no remapper
    compatible, report = build_compatible_state(incoming, current)
    assert compatible == {}
    assert report["skipped_shape"] == ["w"]  # Stage-2 would treat this as fatal


def test_query_count_remap_resamples_to_target():
    key = "lam.decoder.reduced_patch_embed"
    source = torch.arange(8 * 5, dtype=torch.float32).reshape(1, 8, 5)
    target = torch.zeros(1, 4, 5)
    compatible, report = build_compatible_state(
        {key: source}, {key: target}, remappers=STAGE1_REMAPPERS
    )
    assert key in compatible
    assert tuple(compatible[key].shape) == (1, 4, 5)
    assert report["remapped"]["query"] == [key]
    assert report["remapped"]["temporal"] == []


def test_temporal_endpoint_remap_takes_first_and_last_step():
    key = "lam.decoder.target_step_embed.weight"
    source = torch.stack([torch.full((3,), float(i)) for i in range(5)])  # (5, 3)
    target = torch.zeros(2, 3)
    out = remap_temporal_endpoint_embedding(key, source, target)
    assert out is not None
    assert torch.equal(out[0], torch.zeros(3))   # source step 0
    assert torch.equal(out[1], torch.full((3,), 4.0))  # source step -1


def test_remappers_return_none_for_unrelated_key():
    assert remap_temporal_endpoint_embedding("unrelated", torch.zeros(2), torch.zeros(3)) is None
    assert remap_query_count_embedding("unrelated", torch.zeros(2), torch.zeros(3)) is None
