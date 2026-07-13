"""Shared `--init_ckpt` warm-start support for the training entrypoints.

`--init_ckpt` loads a checkpoint's ``state_dict`` into the model *before* fit
WITHOUT restoring optimizer / scheduler / global-step state. It is a warm-start,
not a resume. It is ignored when ``--ckpt_path`` is set, so a chain-tail full
Lightning resume always takes precedence.

Two warm-start modes share this code (B3 in the cleanup plan):

* Stage 1 (``warm_start_remappers = STAGE1_REMAPPERS``): cross-architecture
  initialization. Shape-mismatched endpoint / query embeddings are remapped
  (temporal endpoints, query-count resampling); any remaining mismatch is
  skipped and reported.
* Stage 2 (``warm_start_require_present_keys = True``): the same-architecture
  Phase-A -> Phase-B transition. A shape mismatch on a key that exists in both
  the checkpoint and the model can only mean the topology disagrees, so it is
  fatal rather than silently skipped.

Both entrypoints previously carried near-identical copies of this logic; this
module is the single source of truth.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from lightning.pytorch.cli import LightningCLI

# A remapper turns a shape-mismatched source tensor into one matching the target
# shape, or returns None if it does not apply to this key.
Remapper = Callable[[str, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]


def remap_temporal_endpoint_embedding(
    key: str, source_value: torch.Tensor, target_value: torch.Tensor
) -> Optional[torch.Tensor]:
    """Map a multi-step temporal embedding onto a 2-step (first/last) target."""
    if key.endswith('lam.ti_encoder.step_embed'):
        if source_value.ndim != 4 or target_value.ndim != 4:
            return None
        if source_value.shape[0] != target_value.shape[0] or source_value.shape[2:] != target_value.shape[2:]:
            return None
        if source_value.shape[1] < 2 or target_value.shape[1] != 2:
            return None
        remapped = target_value.detach().clone()
        remapped[:, 0].copy_(source_value[:, 0].to(device=remapped.device, dtype=remapped.dtype))
        remapped[:, 1].copy_(source_value[:, -1].to(device=remapped.device, dtype=remapped.dtype))
        return remapped

    if key.endswith('lam.decoder.target_step_embed.weight'):
        if source_value.ndim != 2 or target_value.ndim != 2:
            return None
        if source_value.shape[1] != target_value.shape[1]:
            return None
        if source_value.shape[0] < 2 or target_value.shape[0] != 2:
            return None
        remapped = target_value.detach().clone()
        remapped[0].copy_(source_value[0].to(device=remapped.device, dtype=remapped.dtype))
        remapped[1].copy_(source_value[-1].to(device=remapped.device, dtype=remapped.dtype))
        return remapped

    return None


def remap_query_count_embedding(
    key: str, source_value: torch.Tensor, target_value: torch.Tensor
) -> Optional[torch.Tensor]:
    """Resample a query-token embedding to a different query count along its query axis."""
    query_axis = None
    if key.endswith('lam.ti_encoder.query_embed'):
        query_axis = 2
    elif key.endswith('lam.ti_encoder.query_step_embed'):
        query_axis = 2
    elif key.endswith('lam.decoder.reduced_patch_embed'):
        query_axis = 1
    elif key.endswith('lam.decoder.ti_patch_embed'):
        query_axis = 1
    elif key.endswith('lam.decoder.tc_patch_embed'):
        query_axis = 1
    else:
        return None

    if source_value.ndim != target_value.ndim:
        return None
    if query_axis >= source_value.ndim:
        return None
    source_shape = list(source_value.shape)
    target_shape = list(target_value.shape)
    if source_shape[:query_axis] != target_shape[:query_axis]:
        return None
    if source_shape[query_axis + 1 :] != target_shape[query_axis + 1 :]:
        return None

    source_query_count = int(source_shape[query_axis])
    target_query_count = int(target_shape[query_axis])
    if source_query_count <= 0 or target_query_count <= 0:
        return None

    remapped = target_value.detach().clone()
    indices = torch.linspace(0, source_query_count - 1, target_query_count).round().to(dtype=torch.long)
    expanded = source_value.index_select(query_axis, indices)
    remapped.copy_(expanded.to(device=remapped.device, dtype=remapped.dtype))
    return remapped


# Named remapper bundle for the Stage-1 cross-architecture warm-start.
STAGE1_REMAPPERS: Tuple[Tuple[str, Remapper], ...] = (
    ('temporal', remap_temporal_endpoint_embedding),
    ('query', remap_query_count_embedding),
)


def build_compatible_state(
    state_dict: Dict[str, torch.Tensor],
    current_state: Dict[str, torch.Tensor],
    *,
    remappers: Sequence[Tuple[str, Remapper]] = (),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Filter ``state_dict`` against the model's ``current_state``.

    Keeps keys present in both: exact shape matches verbatim, shape mismatches
    via the first applicable remapper, otherwise recorded in ``skipped_shape``.
    Returns ``(compatible_state, report)`` where ``report`` carries the
    ``remapped`` lists (keyed by remapper name) and ``skipped_shape``.
    """
    compatible_state: Dict[str, torch.Tensor] = {}
    remapped: Dict[str, List[str]] = {name: [] for name, _ in remappers}
    skipped_shape: List[str] = []
    for key, value in state_dict.items():
        if key not in current_state:
            continue
        target = current_state[key]
        if tuple(target.shape) == tuple(value.shape):
            compatible_state[key] = value
            continue
        for name, fn in remappers:
            out = fn(key, value, target)
            if out is not None:
                compatible_state[key] = out
                remapped[name].append(key)
                break
        else:
            skipped_shape.append(key)
    return compatible_state, {'remapped': remapped, 'skipped_shape': skipped_shape}


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('state_dict'), dict):
        return checkpoint['state_dict']
    return checkpoint  # type: ignore[return-value]


def _sample(keys: Sequence[str], n: int = 5) -> str:
    head = ','.join(keys[:n])
    return f'[{head}{"..." if len(keys) > n else ""}]'


class WeightInitCLI(LightningCLI):
    """LightningCLI variant supporting an ``--init_ckpt`` warm-start.

    Subclasses configure the mode via class attributes:

    * ``warm_start_remappers``: named remappers applied to shape-mismatched keys
      (Stage 1 cross-architecture init). Default: none.
    * ``warm_start_require_present_keys``: if True, a shape mismatch on a key
      present in both checkpoint and model is fatal (Stage 2 same-architecture
      transition). Default: False (skip and report).
    """

    warm_start_remappers: Sequence[Tuple[str, Remapper]] = ()
    warm_start_require_present_keys: bool = False

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument(
            '--init_ckpt',
            type=str,
            default='',
            help=(
                'Optional path to a Lightning checkpoint whose `state_dict` is '
                'loaded into the model with strict=False before fit. Optimizer, '
                'scheduler, and step counters are NOT restored (warm-start, not '
                'resume). Ignored when --ckpt_path is set (full resume takes '
                'precedence).'
            ),
        )

    def _resolve_subcommand_namespace(self):
        cfg = self.config
        if self.subcommand and hasattr(cfg, self.subcommand):
            return getattr(cfg, self.subcommand)
        return cfg

    def before_fit(self) -> None:
        cfg = self._resolve_subcommand_namespace()
        init_ckpt = str(getattr(cfg, 'init_ckpt', '') or '')
        ckpt_path_str = str(getattr(cfg, 'ckpt_path', None) or '')
        if not init_ckpt:
            return
        if ckpt_path_str:
            print(
                f'[init_ckpt] --ckpt_path={ckpt_path_str!r} is set; '
                f'skipping --init_ckpt warm-start (full resume takes precedence).',
                flush=True,
            )
            return

        print(f'[init_ckpt] loading state_dict from {init_ckpt!r} (shape-compatible strict=False)', flush=True)
        checkpoint = torch.load(init_ckpt, map_location='cpu', weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        current_state = self.model.state_dict()
        compatible_state, report = build_compatible_state(
            state_dict, current_state, remappers=self.warm_start_remappers
        )
        skipped_shape = report['skipped_shape']

        if self.warm_start_require_present_keys and skipped_shape:
            raise RuntimeError(
                f'[init_ckpt] same-architecture warm-start from {init_ckpt!r} has '
                f'{len(skipped_shape)} shape-mismatched key(s) {_sample(skipped_shape, 10)}. '
                f'In a same-model Phase-A -> Phase-B transition a shape mismatch means the '
                f'checkpoint topology disagrees with the constructed model. Refusing to '
                f'silently skip trainable weights; check that the model config matches the '
                f'checkpoint that produced --init_ckpt.'
            )

        result = self.model.load_state_dict(compatible_state, strict=False)
        missing = list(getattr(result, 'missing_keys', []) or [])
        unexpected = list(getattr(result, 'unexpected_keys', []) or [])
        remapped_summary = ' '.join(
            f'remapped_{name}={len(keys)}' for name, keys in report['remapped'].items()
        )
        print(
            f'[init_ckpt] loaded={len(compatible_state)} {remapped_summary} '
            f'skipped_shape={len(skipped_shape)} '
            f'missing={len(missing)} unexpected={len(unexpected)} '
            f'(sample_missing={_sample(missing)} '
            f'sample_unexpected={_sample(unexpected)} '
            f'sample_skipped_shape={_sample(skipped_shape)})',
            flush=True,
        )
