from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


def _as_sequence(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        stripped = stripped.strip('[]()')
        if not stripped:
            return None
        return [part.strip() for part in stripped.split(',') if part.strip()]
    if isinstance(value, Sequence):
        return list(value)
    raise TypeError(f'Expected a sequence or comma-separated string, got {type(value).__name__}.')


@dataclass(frozen=True)
class StrideSchedule:
    choices: tuple[int, ...]
    probs: tuple[float, ...]
    eval_stride: int
    configured: bool = False

    @property
    def max_stride(self) -> int:
        return max((*self.choices, self.eval_stride))

    def span(self, num_frames: int, stride: int | None = None) -> int:
        return 1 + (int(num_frames) - 1) * int(self.eval_stride if stride is None else stride)

    def max_span(self, num_frames: int) -> int:
        return self.span(num_frames, self.max_stride)


def normalize_stride_schedule(
    *,
    chunk_stride: int,
    stride_choices: Any = None,
    stride_probs: Any = None,
    eval_chunk_stride: int | None = None,
) -> StrideSchedule:
    base_stride = int(chunk_stride)
    if base_stride <= 0:
        raise ValueError(f'chunk_stride must be positive, got {chunk_stride}.')

    raw_choices = _as_sequence(stride_choices)
    configured = raw_choices is not None or eval_chunk_stride is not None or _as_sequence(stride_probs) is not None
    if raw_choices is None:
        choices = (base_stride,)
    else:
        choices = tuple(int(value) for value in raw_choices)
        if not choices:
            raise ValueError('stride_choices must not be empty when provided.')
    if any(choice <= 0 for choice in choices):
        raise ValueError(f'stride_choices must all be positive, got {choices}.')

    raw_probs = _as_sequence(stride_probs)
    if raw_probs is None:
        probs = tuple(1.0 / len(choices) for _ in choices)
    else:
        probs = tuple(float(value) for value in raw_probs)
        if len(probs) != len(choices):
            raise ValueError(
                f'stride_probs length {len(probs)} must match stride_choices length {len(choices)}.'
            )
        if any(prob < 0.0 for prob in probs):
            raise ValueError(f'stride_probs must be non-negative, got {probs}.')
        total = float(sum(probs))
        if total <= 0.0:
            raise ValueError('stride_probs must contain at least one positive entry.')
        probs = tuple(prob / total for prob in probs)

    eval_stride = int(base_stride if eval_chunk_stride is None else eval_chunk_stride)
    if eval_stride <= 0:
        raise ValueError(f'eval_chunk_stride must be positive, got {eval_chunk_stride}.')
    return StrideSchedule(choices=choices, probs=probs, eval_stride=eval_stride, configured=configured)


def sample_chunk_stride(split: str, schedule: StrideSchedule) -> int:
    if split == 'train':
        if len(schedule.choices) == 1:
            return int(schedule.choices[0])
        probs = torch.tensor(schedule.probs, dtype=torch.float64)
        sampled = int(torch.multinomial(probs, num_samples=1).item())
        return int(schedule.choices[sampled])
    return int(schedule.eval_stride)
