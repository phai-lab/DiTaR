"""Canonical model-dimension defaults + a group-norm helper, shared by the
Stage-1/Stage-2 wrappers, the inner LAMs, and the predictive building blocks.

These were previously carried in a since-removed multiview LAM module; this is
their dedicated, paper-facing home.
"""

DEFAULT_MODEL_DIM = 768
DEFAULT_NUM_HEADS = 12
DEFAULT_CONV_CHANNELS = 768
DEFAULT_ENC_BLOCKS = 4
DEFAULT_DEC_BLOCKS = 12
DEFAULT_BOTTLENECK_DIM = 32
DEFAULT_BOTTLENECK_HIDDEN_DIM = 128


def _group_norm_groups(num_channels: int, max_groups: int = 32) -> int:
    """Largest divisor of ``num_channels`` that is <= ``max_groups`` (for GroupNorm)."""
    group_count = min(int(max_groups), int(num_channels))
    while group_count > 1 and num_channels % group_count != 0:
        group_count -= 1
    return group_count
