"""Weighted multi-manifest DataModule for stage-1 multi-dataset training (v1).

Differences vs `MultiManifestDROIDDataModule`:

  * Per-manifest `sample_weight` (target source ratio, e.g. `0.4`). The
    DataModule computes per-sample weights so the global sum per source
    matches the target ratio, regardless of the source's row count.
  * Per-manifest `chunk_stride` / `stride_choices` schedule. Falls back to the
    global `chunk_stride`.
  * AgiBot HDF5 manifest format (`agibot_hdf5`) — uses `AgiBotHDF5Dataset`,
    which flattens to per-segment rows and exposes inverse-frequency
    `sample_weights`. The DataModule honors those internal weights and
    rescales them to hit `sample_weight` for the source.
  * Train uses `DistributedWeightedRandomSampler` (per-rank seed,
    with-replacement); val uses `DistributedSampler(shuffle=False)`. The
    user (or the launcher) must set
    `--trainer.use_distributed_sampler=false` so Lightning does not wrap
    our train sampler.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Sampler

from ditar_encoder.data.agibot_hdf5_dataset import AgiBotHDF5Dataset
from ditar_encoder.data.chunked_dataset import ChunkedCollatorForLatentAction
from ditar_encoder.data.droid_hdf5_dataset import DROIDHDF5Dataset


def _temp_timing_enabled() -> bool:
    flag = os.environ.get('DITAR_TEMP_TIMING', os.environ.get('TASKVAE_TEMP_TIMING', '0')).strip().lower()
    return flag not in {'', '0', 'false', 'no'}


def _temp_timing_log(message: str) -> None:
    if _temp_timing_enabled():
        print(f'[TEMP TIMING][WeightedMultiManifestDataModule] {message}', flush=True)


def _coerce_str(value: Any) -> str:
    return '' if value is None else str(value)


def _expand_path(value: str) -> str:
    """Resolve ``~`` and ``$VAR`` / ``${VAR}`` in a path string.

    DiTaR config standard: dataset roots are given as env placeholders (e.g.
    ``${DITAR_DATA_ROOT}/droid``) so the public configs carry no absolute paths.
    An unset variable is left verbatim, which fails loudly at load time rather
    than silently resolving to the wrong place.
    """
    if not value:
        return ''
    return os.path.expanduser(os.path.expandvars(value))


def _coerce_int(value: Any, default: int) -> int:
    if value is None or value == '':
        return default
    return int(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None or value == '':
        return default
    return float(value)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None or value == '':
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


class _HarmonizedSourceDataset(Dataset):
    """Wraps a per-source dataset to standardize `action` shape and
    `dataset_name` literal so mixed-source batches collate cleanly."""

    def __init__(self, base: Dataset, source_name: str, num_frames: int) -> None:
        self.base = base
        self.source_name = source_name
        self.num_frames = int(num_frames)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        sample = dict(sample)
        sample['action'] = torch.zeros(self.num_frames, 1, dtype=torch.float32)
        sample['dataset_name'] = self.source_name
        return sample


class DistributedWeightedRandomSampler(Sampler[int]):
    """With-replacement weighted sampler with independent rank-local RNG streams.

    `weights` is the GLOBAL per-sample weight vector (length = total rows
    across the concat dataset). Each rank issues `num_samples_per_rank =
    num_samples // num_replicas` indices per epoch. Because sampling is with
    replacement, different ranks may draw the same global index. Lightning's
    `set_epoch` is called between epochs and updates the per-rank seed.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if num_replicas < 1:
            raise ValueError('num_replicas must be >= 1.')
        if not 0 <= rank < num_replicas:
            raise ValueError(f'rank={rank} out of range for num_replicas={num_replicas}.')

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if (weights_tensor < 0).any():
            raise ValueError('weights must be non-negative.')
        total_weight = float(weights_tensor.sum().item())
        if total_weight <= 0:
            raise ValueError('weights must contain at least one positive entry.')

        self.weights = weights_tensor
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples_per_rank = max(1, int(num_samples) // int(num_replicas))
        self.total_size = self.num_samples_per_rank * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 100003 + self.rank)
        indices = torch.multinomial(
            self.weights,
            self.num_samples_per_rank,
            replacement=True,
            generator=g,
        ).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class WeightedMultiManifestDataModule(LightningDataModule):
    """Concat datamodule with weighted source-ratio sampler.

    Each manifest spec is a dict with these case-insensitive keys:

        name:           label used in logs (string, required)
        data_format:    'hdf5' | 'agibot_hdf5'
        data_root:      root for relative hdf5 paths (required)
        sample_weight:  target source ratio (float, required, must be > 0)
        h5_root:        AgiBot per-episode h5 root (required for `agibot_hdf5`)
        manifest_path:  AgiBot manifest jsonl, relative to data_root or absolute
                        (required for `agibot_hdf5` format)
        video_key:      AgiBot view (default `head_color.mp4`)
        episode_root:   hdf5 split-root containing {train,test}/episode_*.h5
                        (required for `hdf5` format)
        chunk_stride:   per-source override of the global chunk_stride
        stride_choices: optional train-time per-sample stride choices
        stride_probs:   probabilities for stride_choices
        eval_chunk_stride: fixed val/test stride; defaults to chunk_stride
        split_ratio:    per-source override of the global split_ratio
        image_aug:      per-source override of the global image_aug flag
        min_frame_count: per-source override of the global min_frame_count

    Global args (`batch_size`, `num_frames`, `chunk_stride`, `resolution`,
    `num_workers`, `split_ratio`, `image_aug`, `min_frame_count`) match the
    semantics of `MultiManifestDROIDDataModule` and apply per-source unless a
    manifest override is set.
    """

    SUPPORTED_FORMATS = ('hdf5', 'agibot_hdf5')

    def __init__(
        self,
        manifests: List[Dict[str, Any]],
        batch_size: int = 16,
        resolution: int = 256,
        num_frames: int = 17,
        chunk_stride: int = 1,
        split_ratio: float = 0.98,
        image_aug: bool = False,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        min_frame_count: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not manifests:
            raise ValueError('WeightedMultiManifestDataModule requires at least one manifest spec.')

        self.manifests = [self._normalize_manifest(idx, spec) for idx, spec in enumerate(manifests)]
        weight_sum = sum(m['sample_weight'] for m in self.manifests)
        if weight_sum <= 0:
            raise ValueError('Sum of sample_weight across manifests must be > 0.')
        # Auto-normalize so sample_weight always sums to 1.
        for m in self.manifests:
            m['sample_weight'] = m['sample_weight'] / weight_sum

        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size or batch_size)
        self.resolution = (int(resolution), int(resolution))
        self.num_frames = int(num_frames)
        self.chunk_stride = int(chunk_stride)
        self.split_ratio = float(split_ratio)
        self.image_aug = bool(image_aug)
        self.num_workers = int(num_workers)
        self.min_frame_count = int(min_frame_count)
        self.seed = int(seed)
        self.collate_fn = ChunkedCollatorForLatentAction()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_sample_weights: Optional[torch.Tensor] = None
        self.train_num_samples: int = 0
        self.save_hyperparameters()

    @classmethod
    def _normalize_manifest(cls, idx: int, spec: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(spec, dict):
            raise TypeError(f'manifests[{idx}] must be a dict, got {type(spec).__name__}.')
        normalized = {k.lower(): v for k, v in spec.items()}
        name = _coerce_str(normalized.get('name')).strip()
        if not name:
            raise ValueError(f'manifests[{idx}] missing required field "name".')
        data_format = _coerce_str(normalized.get('data_format', 'hdf5')).strip().lower() or 'hdf5'
        if data_format not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f'manifests[{idx}] (name={name!r}) has unsupported data_format={data_format!r}. '
                f'Expected one of {cls.SUPPORTED_FORMATS}.'
            )
        data_root = _expand_path(_coerce_str(normalized.get('data_root')))
        if not data_root:
            raise ValueError(f'manifests[{idx}] (name={name!r}) missing required field "data_root".')
        sample_weight = _coerce_float(normalized.get('sample_weight'), -1.0)
        if sample_weight <= 0:
            raise ValueError(
                f'manifests[{idx}] (name={name!r}) requires sample_weight > 0; got {sample_weight!r}.'
            )
        normalized['name'] = name
        normalized['data_root'] = data_root
        normalized['data_format'] = data_format
        normalized['sample_weight'] = float(sample_weight)
        normalized['jsonl_path'] = _coerce_str(normalized.get('jsonl_path'))
        normalized['video_key'] = _coerce_str(normalized.get('video_key'))
        normalized['episode_root'] = _expand_path(_coerce_str(normalized.get('episode_root')))
        normalized['manifest_path'] = _expand_path(_coerce_str(normalized.get('manifest_path')))
        normalized['h5_root'] = _expand_path(_coerce_str(normalized.get('h5_root')))
        normalized['extract_stride'] = _coerce_int(normalized.get('extract_stride'), 0)
        normalized['chunk_stride'] = normalized.get('chunk_stride')
        normalized['stride_choices'] = normalized.get('stride_choices')
        normalized['stride_probs'] = normalized.get('stride_probs')
        normalized['eval_chunk_stride'] = normalized.get('eval_chunk_stride')
        return normalized

    def _resolve_chunk_stride(self, manifest: Dict[str, Any]) -> int:
        return _coerce_int(manifest.get('chunk_stride'), self.chunk_stride)

    def _resolve_stride_kwargs(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        eval_stride = _coerce_int(manifest.get('eval_chunk_stride'), 0)
        return {
            'chunk_stride': self._resolve_chunk_stride(manifest),
            'stride_choices': manifest.get('stride_choices') or None,
            'stride_probs': manifest.get('stride_probs') or None,
            'eval_chunk_stride': eval_stride or None,
        }

    def _build_one_dataset(self, manifest: Dict[str, Any], split: str) -> Dataset:
        name = manifest['name']
        data_format = manifest['data_format']
        data_root = manifest['data_root']

        split_ratio = _coerce_float(manifest.get('split_ratio'), self.split_ratio)
        image_aug = _coerce_bool(manifest.get('image_aug'), self.image_aug)
        min_frame_count = _coerce_int(manifest.get('min_frame_count'), self.min_frame_count)
        stride_kwargs = self._resolve_stride_kwargs(manifest)
        chunk_stride = int(stride_kwargs['chunk_stride'])

        build_start = time.monotonic()
        _temp_timing_log(
            f'manifest={name!r} split={split} format={data_format} '
            f'chunk_stride={chunk_stride} stride_choices={stride_kwargs["stride_choices"]} '
            f'eval_chunk_stride={stride_kwargs["eval_chunk_stride"]} min_frames={min_frame_count} '
            f'image_aug={image_aug}'
        )

        if data_format == 'agibot_hdf5':
            h5_root = manifest['h5_root']
            if not h5_root:
                raise ValueError(f'manifest {name!r} with data_format=agibot_hdf5 requires "h5_root".')
            manifest_path = manifest['manifest_path'] or 'manifest.jsonl'
            video_key = manifest['video_key'] or AgiBotHDF5Dataset.DEFAULT_VIDEO_KEY
            extract_stride = int(manifest.get('extract_stride') or 2)
            dataset: Dataset = AgiBotHDF5Dataset(
                h5_root=h5_root,
                manifest_path=manifest_path,
                data_root=data_root,
                video_key=video_key,
                extract_stride=extract_stride,
                resolution=self.resolution,
                num_frames=self.num_frames,
                **stride_kwargs,
                split=split,
                split_ratio=split_ratio,
                image_aug=image_aug,
            )
        elif data_format == 'hdf5':
            episode_root = manifest['episode_root'] or data_root
            dataset = DROIDHDF5Dataset(
                data_root=episode_root,
                split=split,
                resolution=self.resolution,
                num_frames=self.num_frames,
                **stride_kwargs,
                image_aug=image_aug if split == 'train' else False,
                min_frame_count=min_frame_count,
            )
        else:
            raise ValueError(
                f'manifest {name!r} has unsupported data_format={data_format!r}; '
                f'expected one of {self.SUPPORTED_FORMATS}.'
            )
        _temp_timing_log(
            f'manifest={name!r} split={split} built in {time.monotonic() - build_start:.2f}s '
            f'len={len(dataset)}'
        )
        return _HarmonizedSourceDataset(dataset, source_name=name, num_frames=self.num_frames)

    def _per_source_within_weights(self, base_dataset: Dataset, manifest: Dict[str, Any]) -> torch.Tensor:
        """Returns the un-rescaled within-source weight vector for `base_dataset`.

        Default: uniform (all 1.0).
        AgiBot: inverse-frequency from `AgiBotHDF5Dataset.sample_weights`.
        """
        n = len(base_dataset)
        underlying = base_dataset.base if isinstance(base_dataset, _HarmonizedSourceDataset) else base_dataset
        within = getattr(underlying, 'sample_weights', None)
        if within is None:
            return torch.ones(n, dtype=torch.float64)
        within = torch.as_tensor(within, dtype=torch.float64)
        if within.numel() != n:
            raise ValueError(
                f'manifest {manifest["name"]!r}: sample_weights length {within.numel()} '
                f'does not match dataset length {n}.'
            )
        if (within < 0).any():
            raise ValueError(f'manifest {manifest["name"]!r}: sample_weights must be non-negative.')
        total = float(within.sum().item())
        if total <= 0:
            raise ValueError(
                f'manifest {manifest["name"]!r}: sample_weights sum is non-positive.'
            )
        return within

    def _build_train_split(self) -> None:
        per_source: List[Dataset] = []
        per_source_weights: List[torch.Tensor] = []
        sizes: List[int] = []
        breakdowns: List[str] = []

        for manifest in self.manifests:
            ds = self._build_one_dataset(manifest, 'train')
            within = self._per_source_within_weights(ds, manifest)
            target_ratio = float(manifest['sample_weight'])
            within_normalized = within / float(within.sum().item())
            scaled = within_normalized * target_ratio
            per_source.append(ds)
            per_source_weights.append(scaled)
            sizes.append(len(ds))
            breakdowns.append(f'{manifest["name"]}={len(ds)}@{target_ratio:.3f}')

        total = sum(sizes)
        if total == 0:
            raise ValueError('WeightedMultiManifest train split produced 0 rows across all manifests.')

        if len(per_source) == 1:
            self.train_dataset = per_source[0]
            self.train_sample_weights = per_source_weights[0]
        else:
            self.train_dataset = ConcatDataset(per_source)
            self.train_sample_weights = torch.cat(per_source_weights, dim=0)

        self.train_num_samples = total

        weight_sum = float(self.train_sample_weights.sum().item())
        print(
            f'[WeightedMultiManifestDataModule] split=train total={total} '
            f'({", ".join(breakdowns)}) num_samples_per_epoch={self.train_num_samples} '
            f'weight_sum={weight_sum:.6f}',
            flush=True,
        )

    def _build_val_split(self) -> None:
        per_source: List[Dataset] = []
        sizes: List[int] = []
        breakdowns: List[str] = []
        for manifest in self.manifests:
            ds = self._build_one_dataset(manifest, 'val')
            per_source.append(ds)
            sizes.append(len(ds))
            breakdowns.append(f'{manifest["name"]}={len(ds)}')
        total = sum(sizes)
        if total == 0:
            raise ValueError('WeightedMultiManifest val split produced 0 rows across all manifests.')

        if len(per_source) == 1:
            self.val_dataset = per_source[0]
        else:
            self.val_dataset = ConcatDataset(per_source)
        print(
            f'[WeightedMultiManifestDataModule] split=val total={total} '
            f'({", ".join(breakdowns)})',
            flush=True,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        setup_start = time.monotonic()
        _temp_timing_log(f'setup(stage={stage}) start')
        if stage in (None, 'fit'):
            self._build_train_split()
            self._build_val_split()
        if stage in (None, 'test'):
            # Test split mirrors val (same row construction).
            self._build_val_split()
            self.test_dataset = self.val_dataset
        if stage not in (None, 'fit', 'test'):
            raise ValueError(f'Invalid stage: {stage}')
        _temp_timing_log(
            f'setup(stage={stage}) finished in {time.monotonic() - setup_start:.2f}s'
        )

    def _loader_kwargs(self) -> dict:
        kwargs: Dict[str, Any] = dict(
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.num_workers > 0:
            kwargs['persistent_workers'] = True
            kwargs['prefetch_factor'] = 2
        return kwargs

    @staticmethod
    def _ddp_world() -> tuple:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None or self.train_sample_weights is None:
            raise RuntimeError('train_dataloader called before setup()')
        world_size, rank = self._ddp_world()
        sampler = DistributedWeightedRandomSampler(
            weights=self.train_sample_weights,
            num_samples=self.train_num_samples,
            num_replicas=world_size,
            rank=rank,
            seed=self.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError('val_dataloader called before setup()')
        world_size, rank = self._ddp_world()
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError('test_dataloader called before setup()')
        world_size, rank = self._ddp_world()
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(),
        )
