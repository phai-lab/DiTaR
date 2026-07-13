"""AgiBot Beta head-camera dataset, reading from pre-extracted JPEG/HDF5 files.

Reads from the per-episode h5 produced by an offline JPEG-extraction step.
Flattens to per-segment rows and returns the standard episode dict via a fast
JPEG-decode-from-h5 I/O path (~ms per frame vs mp4 random-access ~10–30ms).

Frame-index translation
  The extractor subsamples by `stride=2` (30 fps -> 15 fps). So a segment
  spanning frames `[s, e)` in the original 30 fps frame grid corresponds to
  h5 indices `[s // stride, e // stride)`. The dataset reads native h5
  indices and applies its own `chunk_stride` or sampled stride over them.
"""
from __future__ import annotations

import io
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from ditar_encoder.data.dataset_index_cache import file_fingerprint, load_or_build_index_cache
from ditar_encoder.data.stride_sampling import normalize_stride_schedule, sample_chunk_stride


def _temp_timing_enabled() -> bool:
    flag = os.environ.get('DITAR_TEMP_TIMING', os.environ.get('TASKVAE_TEMP_TIMING', '0')).strip().lower()
    return flag not in {'', '0', 'false', 'no'}


def _temp_timing_log(message: str) -> None:
    if _temp_timing_enabled():
        print(f'[TEMP TIMING][AgiBotHDF5Dataset] {message}', flush=True)


class AgiBotHDF5Dataset(Dataset):
    """AgiBot Beta head-camera dataset over pre-extracted JPEG/HDF5 files."""

    DEFAULT_VIDEO_KEY = 'head_color.mp4'
    _MAX_SAMPLE_RECOVERY_ATTEMPTS = 8
    _FRAME_DECODE_EXCEPTIONS = (UnidentifiedImageError, OSError, ValueError)

    def __init__(
        self,
        h5_root: str,
        manifest_path: str,
        data_root: Optional[str] = None,
        video_key: str = DEFAULT_VIDEO_KEY,
        extract_stride: int = 2,
        resolution: Tuple[int, int] = (256, 256),
        num_frames: int = 17,
        chunk_stride: int = 1,
        stride_choices: Sequence[int] | str | None = None,
        stride_probs: Sequence[float] | str | None = None,
        eval_chunk_stride: int | None = None,
        split: str = 'train',
        split_ratio: float = 0.98,
        image_aug: bool = False,
    ) -> None:
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'Unsupported split={split!r}.')

        init_start = time.monotonic()
        self.h5_root = Path(h5_root)
        if not self.h5_root.exists():
            raise FileNotFoundError(f'AgiBot h5 root not found: {self.h5_root}')

        self.data_root = Path(data_root) if data_root else self.h5_root.parent
        manifest_p = Path(manifest_path)
        if not manifest_p.is_absolute():
            manifest_p = self.data_root / manifest_p
        if not manifest_p.exists():
            raise FileNotFoundError(f'AgiBot manifest not found: {manifest_p}')
        self.manifest_path = manifest_p

        self.video_key = video_key
        self.extract_stride = int(extract_stride)
        if self.extract_stride <= 0:
            raise ValueError('extract_stride must be > 0.')
        self.num_frames = int(num_frames)
        self.chunk_stride = int(chunk_stride)
        self.stride_schedule = normalize_stride_schedule(
            chunk_stride=self.chunk_stride,
            stride_choices=stride_choices,
            stride_probs=stride_probs,
            eval_chunk_stride=eval_chunk_stride,
        )
        self.split = split
        self.split_ratio = float(split_ratio)
        self.resolution = resolution
        self.image_transform = self._build_image_transform(resolution, image_aug and split == 'train')

        # span counted in extracted h5-frame units (already at 1/extract_stride
        # of the source fps).
        self._span_h5 = self.stride_schedule.max_span(self.num_frames)

        train_rows, val_rows = self._load_or_build_split_rows()
        if split == 'train':
            self.rows = train_rows
        else:
            # 'val' and 'test' both pull from the held-out tail.
            self.rows = val_rows
        if not self.rows:
            raise ValueError(f'No AgiBot segments available for split={split!r}.')

        text_counter: Counter[str] = Counter(self._normalize_text(row['action_text']) for row in self.rows)
        self.text_freq: Dict[str, int] = dict(text_counter)
        self.unique_text_count: int = len(self.text_freq)

        weights = torch.empty(len(self.rows), dtype=torch.float64)
        for i, row in enumerate(self.rows):
            key = self._normalize_text(row['action_text'])
            weights[i] = 1.0 / float(self.text_freq[key])
        self.sample_weights: torch.Tensor = weights

        _temp_timing_log(
            f'split={split} segments={len(self.rows)} unique_texts={self.unique_text_count} '
            f'span_h5={self._span_h5} build_time={time.monotonic() - init_start:.2f}s'
        )
        print(
            f'[AgiBotHDF5Dataset] split={split} segments={len(self.rows)} '
            f'unique_action_texts={self.unique_text_count} extract_stride={self.extract_stride} '
            f'chunk_stride={self.chunk_stride} eval_chunk_stride={self.stride_schedule.eval_stride} '
            f'max_span_h5_frames={self._span_h5}',
            flush=True,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return (text or '').strip().lower()

    @staticmethod
    def _build_image_transform(resolution: Tuple[int, int], image_aug: bool):
        augmentations = []
        if image_aug:
            augmentations.extend(
                [
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ]
            )
        augmentations.append(transforms.ToTensor())
        return transforms.Compose(augmentations)

    def _episode_h5_path(self, task_id: int, episode_id: int) -> Path:
        return self.h5_root / f'task_{task_id}' / f'episode_{episode_id}.h5'

    def _load_and_flatten_manifest(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        skipped_short = 0
        skipped_missing_h5 = 0
        skipped_empty = 0
        load_start = time.monotonic()
        with self.manifest_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                episode = json.loads(line)
                videos = episode.get('videos') or {}
                rel_video = videos.get(self.video_key)
                if not rel_video:
                    continue
                instructions = episode.get('instructions') or []
                if not instructions:
                    skipped_empty += 1
                    continue
                task_id = int(episode.get('task_id', -1))
                episode_id = int(episode.get('episode_id', -1))
                h5_path = self._episode_h5_path(task_id, episode_id)
                if not h5_path.exists():
                    skipped_missing_h5 += 1
                    continue
                for seg_idx, segment in enumerate(instructions):
                    src_start = int(segment.get('start_frame', 0))
                    src_end = int(segment.get('end_frame', 0))
                    action_text = str(segment.get('action_text') or '').strip()
                    # Translate from source frame grid to extracted h5 frame grid.
                    h5_start = src_start // self.extract_stride
                    h5_end = src_end // self.extract_stride
                    if h5_end - h5_start < self._span_h5 or not action_text:
                        skipped_short += 1
                        continue
                    rows.append(
                        dict(
                            episode_id=episode_id,
                            task_id=task_id,
                            seg_idx=seg_idx,
                            h5_path=str(h5_path),
                            h5_start=h5_start,
                            h5_end=h5_end,
                            action_text=action_text,
                        )
                    )
        rows.sort(key=lambda r: (int(r['episode_id']), int(r['seg_idx'])))
        _temp_timing_log(
            f'manifest load: rows_kept={len(rows)} '
            f'skipped_missing_h5={skipped_missing_h5} skipped_empty_inst={skipped_empty} '
            f'skipped_short={skipped_short} in {time.monotonic() - load_start:.2f}s'
        )
        return rows

    def _cache_key_parts(self) -> Dict[str, object]:
        return {
            'version': 1,
            'h5_root': str(self.h5_root.resolve()),
            'data_root': str(self.data_root.resolve()),
            'manifest': file_fingerprint(self.manifest_path),
            'video_key': self.video_key,
            'extract_stride': self.extract_stride,
            'num_frames': self.num_frames,
            'chunk_stride': self.chunk_stride,
            'stride_choices': self.stride_schedule.choices,
            'stride_probs': self.stride_schedule.probs,
            'eval_stride': self.stride_schedule.eval_stride,
            'split_ratio': self.split_ratio,
            'span_h5': self._span_h5,
        }

    def _load_or_build_split_rows(self) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        def build_payload() -> Dict[str, List[Dict[str, object]]]:
            all_rows = self._load_and_flatten_manifest()
            train_rows, val_rows = self._split_rows(all_rows)
            return {'train_rows': train_rows, 'val_rows': val_rows}

        payload = load_or_build_index_cache(
            namespace='agibot_hdf5_rows',
            key_parts=self._cache_key_parts(),
            builder=build_payload,
        )
        return list(payload['train_rows']), list(payload['val_rows'])

    def _split_rows(
        self, rows: List[Dict[str, object]]
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        if not rows:
            return [], []
        cut = max(1, min(len(rows) - 1, int(len(rows) * self.split_ratio)))
        return rows[:cut], rows[cut:]

    def _sample_indices(
        self, h5_start: int, h5_end: int
    ) -> Tuple[List[int], torch.Tensor, int]:
        sampled_stride = sample_chunk_stride(self.split, self.stride_schedule)
        span = self.stride_schedule.span(self.num_frames, sampled_stride)
        max_offset = max(h5_end - span - h5_start, 0)
        if self.split == 'train' and max_offset > 0:
            offset = int(torch.randint(max_offset + 1, (1,)).item())
        else:
            offset = max_offset // 2
        start = h5_start + offset

        indices: List[int] = []
        pad_mask: List[bool] = []
        last_valid = max(h5_end - 1, h5_start)
        for i in range(self.num_frames):
            raw_index = start + i * sampled_stride
            pad_mask.append(raw_index < h5_end)
            indices.append(min(raw_index, last_valid))
        return indices, torch.tensor(pad_mask, dtype=torch.bool), int(sampled_stride)

    @staticmethod
    def _decode_jpeg(buf: bytes) -> Image.Image:
        return Image.open(io.BytesIO(buf)).convert('RGB')

    def _read_h5_frames(self, h5_path: Path, frame_indices: List[int]) -> List[Image.Image]:
        with h5py.File(h5_path, 'r') as handle:
            n = int(handle.attrs.get('num_frames', 0))
            if n <= 0:
                raise ValueError(f'Empty AgiBot h5 file: {h5_path}')
            frames_ds = handle['frames']
            frames_out: List[Image.Image] = []
            last_valid = n - 1
            for fi in frame_indices:
                fi_clamped = max(0, min(int(fi), last_valid))
                jpeg_bytes = np.asarray(frames_ds[fi_clamped]).tobytes()
                img = self._decode_jpeg(jpeg_bytes)
                if img.size != (self.resolution[1], self.resolution[0]):
                    img = img.resize((self.resolution[1], self.resolution[0]), Image.BILINEAR)
                frames_out.append(img)
        return frames_out

    def __len__(self) -> int:
        return len(self.rows)

    def _fetch_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        h5_path = Path(str(row['h5_path']))
        h5_start = int(row['h5_start'])
        h5_end = int(row['h5_end'])
        if h5_end - h5_start < self._span_h5:
            h5_end = h5_start + self._span_h5

        frame_indices, pad_mask, sampled_stride = self._sample_indices(h5_start, h5_end)
        frames = self._read_h5_frames(h5_path, frame_indices)
        video_tensor = torch.stack([self.image_transform(f) for f in frames], dim=0)

        return dict(
            videos=video_tensor,
            pad_mask=pad_mask,
            sampled_chunk_stride=torch.tensor(int(sampled_stride), dtype=torch.long),
            task_instruction=str(row['action_text']).lower(),
            action=torch.zeros(self.num_frames, 1, dtype=torch.float32),
            dataset_name='agibot',
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        last_error: Optional[BaseException] = None
        n = len(self.rows)
        if n == 0:
            raise RuntimeError('AgiBotHDF5Dataset is empty.')
        max_attempts = min(n, self._MAX_SAMPLE_RECOVERY_ATTEMPTS)
        # Adjacent rows in self.rows belong to the same episode. Jump far on
        # retry so we land in a different file.
        stride = max(1, n // max_attempts)
        if stride % 2 == 0:
            stride += 1
        sample_idx = idx % n
        for _ in range(max_attempts):
            try:
                return self._fetch_sample(sample_idx)
            except (*self._FRAME_DECODE_EXCEPTIONS, RuntimeError, FileNotFoundError) as error:
                last_error = error
                row = self.rows[sample_idx]
                warnings.warn(
                    f'Skipping bad AgiBot h5 sample idx={sample_idx} '
                    f'h5={row.get("h5_path")} segment=({row.get("h5_start")},'
                    f'{row.get("h5_end")}): {error}',
                    RuntimeWarning,
                    stacklevel=2,
                )
                sample_idx = (sample_idx + stride) % n
                continue
        raise RuntimeError(
            f'Unable to recover a valid AgiBot h5 sample after {max_attempts} attempts '
            f'starting at idx={idx}.'
        ) from last_error
