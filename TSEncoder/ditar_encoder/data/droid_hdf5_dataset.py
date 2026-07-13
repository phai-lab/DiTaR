from io import BytesIO
import os
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple
import warnings

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from ditar_encoder.data.dataset_index_cache import dir_fingerprint, load_or_build_index_cache
from ditar_encoder.data.stride_sampling import normalize_stride_schedule, sample_chunk_stride


def _temp_timing_enabled() -> bool:
    flag = os.environ.get('DITAR_TEMP_TIMING', os.environ.get('TASKVAE_TEMP_TIMING', '0')).strip().lower()
    return flag not in {'', '0', 'false', 'no'}


def _temp_timing_log(message: str) -> None:
    if _temp_timing_enabled():
        print(f'[TEMP TIMING][DROIDHDF5Dataset] {message}', flush=True)


class DROIDHDF5Dataset(Dataset):
    _FRAME_DECODE_EXCEPTIONS = (UnidentifiedImageError, OSError, ValueError)
    _MAX_SAMPLE_RECOVERY_ATTEMPTS = 8

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        resolution: Tuple[int, int] = (224, 224),
        num_frames: int = 4,
        chunk_stride: int = 1,
        stride_choices: Sequence[int] | str | None = None,
        stride_probs: Sequence[float] | str | None = None,
        eval_chunk_stride: int | None = None,
        image_aug: bool = False,
        min_frame_count: int = 0,
    ) -> None:
        init_start = time.monotonic()
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'Unsupported split={split!r}.')

        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = int(num_frames)
        self.chunk_stride = int(chunk_stride)
        self.stride_schedule = normalize_stride_schedule(
            chunk_stride=self.chunk_stride,
            stride_choices=stride_choices,
            stride_probs=stride_probs,
            eval_chunk_stride=eval_chunk_stride,
        )
        self.max_chunk_span = self.stride_schedule.max_span(self.num_frames)
        self.resolution = resolution
        self.min_frame_count = int(min_frame_count)
        if self.min_frame_count < 0:
            raise ValueError('min_frame_count must be non-negative.')
        self.effective_min_frame_count = (
            max(self.min_frame_count, self.max_chunk_span)
            if self.stride_schedule.configured
            else self.min_frame_count
        )
        self.image_transform = self._build_image_transform(resolution, image_aug and split == 'train')

        split_dir_name = 'train' if split == 'train' else 'test'
        self.split_dir = self.data_root / split_dir_name
        if not self.split_dir.exists():
            raise FileNotFoundError(f'DROID HDF5 split directory not found: {self.split_dir}')

        self.episode_paths = [Path(path) for path in self._load_or_build_episode_paths()]
        if not self.episode_paths:
            raise ValueError(f'No DROID HDF5 episode files found for split={split!r} under {self.split_dir}')
        _temp_timing_log(
            f'split={split} dataset init finished in {time.monotonic() - init_start:.2f}s'
        )

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

    def _cache_key_parts(self) -> Dict[str, object]:
        return {
            'version': 1,
            'data_root': str(self.data_root.resolve()),
            'split_dir': dir_fingerprint(self.split_dir),
            'split': self.split,
            'num_frames': self.num_frames,
            'chunk_stride': self.chunk_stride,
            'stride_choices': self.stride_schedule.choices,
            'stride_probs': self.stride_schedule.probs,
            'eval_stride': self.stride_schedule.eval_stride,
            'effective_min_frame_count': self.effective_min_frame_count,
            'max_chunk_span': self.max_chunk_span,
        }

    def _scan_episode_paths(self) -> List[str]:
        scan_start = time.monotonic()
        episode_paths = sorted(self.split_dir.glob('episode_*.h5'))
        _temp_timing_log(
            f'split={self.split} discovered {len(episode_paths)} files under {self.split_dir} in '
            f'{time.monotonic() - scan_start:.2f}s'
        )
        if self.effective_min_frame_count > 0:
            filter_start = time.monotonic()
            filtered_episode_paths = []
            for episode_path in episode_paths:
                with h5py.File(episode_path, 'r') as handle:
                    frame_count = int(handle.attrs['num_frames'])
                if frame_count >= self.effective_min_frame_count:
                    filtered_episode_paths.append(episode_path)
            episode_paths = filtered_episode_paths
            _temp_timing_log(
                f'split={self.split} min_frame_count>={self.effective_min_frame_count} '
                f'kept {len(episode_paths)} files in {time.monotonic() - filter_start:.2f}s'
            )
        return [str(path) for path in episode_paths]

    def _load_or_build_episode_paths(self) -> List[str]:
        return list(
            load_or_build_index_cache(
                namespace='hdf5_episode_paths',
                key_parts=self._cache_key_parts(),
                builder=self._scan_episode_paths,
            )
        )

    @staticmethod
    def _center_crop_square(frame: Image.Image) -> Image.Image:
        width, height = frame.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        return frame.crop((left, top, left + crop_size, top + crop_size))

    def _prepare_frame(self, frame: Image.Image) -> Image.Image:
        if frame.size[0] != frame.size[1]:
            frame = self._center_crop_square(frame)
        if frame.size != (self.resolution[1], self.resolution[0]):
            frame = frame.resize((self.resolution[1], self.resolution[0]), Image.BILINEAR)
        return frame

    def _sample_indices(self, frame_count: int) -> Tuple[List[int], torch.Tensor, int, int]:
        sampled_stride = sample_chunk_stride(self.split, self.stride_schedule)
        span = 1 + (self.num_frames - 1) * sampled_stride
        max_start = max(frame_count - span, 0)
        if self.split == 'train' and max_start > 0:
            start = int(torch.randint(max_start + 1, (1,)).item())
        else:
            start = max_start // 2

        indices: List[int] = []
        pad_mask = []
        last_valid = max(frame_count - 1, 0)
        for offset in range(self.num_frames):
            raw_index = start + offset * sampled_stride
            pad_mask.append(raw_index < frame_count)
            indices.append(min(raw_index, last_valid))
        return indices, torch.tensor(pad_mask, dtype=torch.bool), int(start), int(sampled_stride)

    @staticmethod
    def _decode_frame(frame_bytes: np.ndarray) -> Image.Image:
        return Image.open(BytesIO(frame_bytes.tobytes())).convert('RGB')

    @staticmethod
    def _frame_fallback_candidates(frame_index: int, frame_count: int) -> List[int]:
        candidates = [frame_index]
        for distance in range(1, frame_count):
            left = frame_index - distance
            right = frame_index + distance
            if left >= 0:
                candidates.append(left)
            if right < frame_count:
                candidates.append(right)
        return candidates

    def _decode_prepared_frame(self, frame_array, frame_index: int) -> Image.Image:
        return self._prepare_frame(self._decode_frame(frame_array[frame_index]))

    def _recover_frame(
        self,
        frame_array,
        frame_index: int,
        frame_count: int,
        episode_path: Path,
        dataset_name: str,
    ) -> Tuple[Image.Image, int]:
        last_error: Exception | None = None
        for candidate_index in self._frame_fallback_candidates(frame_index, frame_count):
            try:
                frame = self._decode_prepared_frame(frame_array, candidate_index)
            except self._FRAME_DECODE_EXCEPTIONS as error:
                last_error = error
                continue
            if candidate_index != frame_index:
                warnings.warn(
                    f'Recovered corrupt frame in {episode_path} dataset={dataset_name!r} '
                    f'by using nearby frame {candidate_index} instead of {frame_index}.',
                    RuntimeWarning,
                    stacklevel=2,
                )
            return frame, candidate_index

        raise RuntimeError(
            f'Unable to decode any nearby frame for {episode_path} dataset={dataset_name!r} '
            f'around frame_index={frame_index}. Last error: {last_error!r}'
        ) from last_error

    def _sample_recovery_indices(self, idx: int) -> List[int]:
        if len(self.episode_paths) == 0:
            return []
        max_attempts = min(len(self.episode_paths), self._MAX_SAMPLE_RECOVERY_ATTEMPTS)
        return [int((idx + offset) % len(self.episode_paths)) for offset in range(max_attempts)]

    def __len__(self) -> int:
        return len(self.episode_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        last_error: Exception | None = None
        for sample_idx in self._sample_recovery_indices(idx):
            episode_path = self.episode_paths[sample_idx]
            try:
                with h5py.File(episode_path, 'r') as handle:
                    frame_count = int(handle.attrs['num_frames'])
                    if frame_count <= 0:
                        raise ValueError(f'Encountered empty DROID HDF5 episode file: {episode_path}')

                    frame_indices, pad_mask, _chunk_start, sampled_stride = self._sample_indices(frame_count)
                    frames = []
                    actions = []
                    instruction = str(handle.attrs.get('instruction', 'null')).lower()
                    action_array = handle['action']
                    frame_array = handle['frames']
                    text_embedding = None
                    if 'text_embedding' in handle:
                        text_embedding = torch.as_tensor(np.asarray(handle['text_embedding'], dtype=np.float32))

                    for frame_index in frame_indices:
                        frame, resolved_index = self._recover_frame(
                            frame_array=frame_array,
                            frame_index=frame_index,
                            frame_count=frame_count,
                            episode_path=episode_path,
                            dataset_name='frames',
                        )
                        frames.append(self.image_transform(frame))
                        actions.append(torch.as_tensor(np.asarray(action_array[resolved_index], dtype=np.float32)))
            except RuntimeError as error:
                last_error = error
                warnings.warn(
                    f'Skipping bad DROID HDF5 sample {episode_path} after frame recovery failure: {error}',
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            output = dict(
                videos=torch.stack(frames, dim=0),
                pad_mask=pad_mask,
                sampled_chunk_stride=torch.tensor(int(sampled_stride), dtype=torch.long),
                task_instruction=instruction or 'null',
                action=torch.stack(actions, dim=0),
                dataset_name='droid_hdf5',
            )
            if text_embedding is not None:
                output['text_embedding'] = text_embedding
            return output

        raise RuntimeError(f'Unable to recover a valid DROID HDF5 sample for idx={idx}.') from last_error
