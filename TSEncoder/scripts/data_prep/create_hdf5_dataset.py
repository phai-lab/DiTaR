import argparse
import os
import io
import json
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image

def _ditar_data_root() -> str:
    """Base dir for datasets; set DITAR_DATA_ROOT (see TRAINING.md).

    Falls back to the current directory so ``--help`` works and explicit
    --data-root / --out-root arguments always win.
    """
    return os.environ.get("DITAR_DATA_ROOT", "").strip() or "."


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DROID videos into JPEG-in-HDF5 episode files with train/test split."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(_ditar_data_root()) / "data" / "droid",
        help="Root directory of the DROID dataset.",
    )
    parser.add_argument(
        "--jsonl_path",
        type=Path,
        default=Path("episode_instruction_exterior12.jsonl"),
        help="Path to the DROID JSONL index, absolute or relative to data_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to store converted HDF5 train/test episode files and the manifest.",
    )
    parser.add_argument(
        "--video_key",
        type=str,
        default="video_exterior_1_left",
        help="Which video column from the JSONL to convert.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("HEIGHT", "WIDTH"),
        help="Resize output images to this size before JPEG encoding.",
    )
    parser.add_argument(
        "--frame_transform",
        type=str,
        choices=("resize", "center_crop_square", "preserve_original"),
        default="resize",
        help="How to preprocess each frame before JPEG encoding.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=80,
        help="JPEG quality for frame compression.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.98,
        help="Train split ratio. Defaults to 0.98 to match the current DROID loader.",
    )
    parser.add_argument(
        "--split_group_key",
        type=str,
        default="",
        help=(
            "Optional JSONL key used to split each group independently. "
            "For Maniskill, use task_name so every task contributes held-out episodes."
        ),
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap on the total number of episodes to convert after sorting.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes. Use 1 for sequential conversion.",
    )
    parser.add_argument(
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        help="Skip already converted valid episodes. This is the default behavior.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force rewriting episodes even if a valid output file already exists.",
    )
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def resolve_jsonl_path(data_root: Path, jsonl_path: Path) -> Path:
    if jsonl_path.is_absolute():
        return jsonl_path
    return data_root / jsonl_path


def load_rows(jsonl_path: Path, video_key: str, max_episodes: Optional[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if video_key not in row:
                raise KeyError(f"video_key={video_key!r} not found in JSONL row keys={tuple(row.keys())}")
            row["episode_index"] = int(row["episode_index"])
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in JSONL: {jsonl_path}")

    rows.sort(key=lambda row: row["episode_index"])
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError(f"max_episodes must be positive when provided, got {max_episodes}")
        rows = rows[:max_episodes]
    return rows


def _split_row_sequence(
    rows: Sequence[Dict[str, object]],
    split_ratio: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if len(rows) <= 1:
        return list(rows), []
    split_index = max(1, min(len(rows) - 1, int(len(rows) * split_ratio)))
    return list(rows[:split_index]), list(rows[split_index:])


def split_rows(
    rows: Sequence[Dict[str, object]],
    split_ratio: float,
    split_group_key: str = "",
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")
    if not split_group_key:
        return _split_row_sequence(rows, split_ratio)

    rows_by_group: "OrderedDict[str, List[Dict[str, object]]]" = OrderedDict()
    for row in rows:
        if split_group_key not in row:
            raise KeyError(
                f"split_group_key={split_group_key!r} not found in JSONL row keys={tuple(row.keys())}"
            )
        rows_by_group.setdefault(str(row[split_group_key]), []).append(row)

    train_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    for group_rows in rows_by_group.values():
        group_train_rows, group_test_rows = _split_row_sequence(group_rows, split_ratio)
        train_rows.extend(group_train_rows)
        test_rows.extend(group_test_rows)

    if test_rows:
        return train_rows, test_rows
    return _split_row_sequence(rows, split_ratio)


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int], frame_transform: str) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    if frame_transform == "resize":
        return image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    if frame_transform == "center_crop_square":
        image = center_crop_square(image)
        if image.size != (target_size[1], target_size[0]):
            image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        return image
    if frame_transform == "preserve_original":
        return image
    raise ValueError(f"Unsupported frame_transform={frame_transform!r}")


def encode_jpeg(image: Image.Image, jpeg_quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def decode_and_encode_video(
    video_path: Path,
    target_size: Tuple[int, int],
    jpeg_quality: int,
    frame_transform: str,
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    encoded_frames: List[np.ndarray] = []
    output_size: Optional[Tuple[int, int]] = None
    for frame in iio.imiter(video_path):
        image = preprocess_frame(frame, target_size, frame_transform)
        current_size = (image.height, image.width)
        if output_size is None:
            output_size = current_size
        elif current_size != output_size:
            raise ValueError(
                f"Inconsistent processed frame sizes for {video_path}: expected {output_size}, got {current_size}"
            )
        encoded_frames.append(encode_jpeg(image, jpeg_quality))
    if not encoded_frames:
        raise ValueError(f"Decoded zero frames from {video_path}")
    assert output_size is not None
    return encoded_frames, output_size


def is_valid_hdf5_episode(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        with h5py.File(output_path, "r") as handle:
            if not all(key in handle for key in ("frames", "state", "action")):
                return False
            num_frames = int(handle.attrs.get("num_frames", -1))
            if num_frames <= 0:
                return False
            if len(handle["frames"]) != num_frames:
                return False
            if handle["state"].shape != (num_frames, 7):
                return False
            if handle["action"].shape != (num_frames, 7):
                return False
            if handle.attrs.get("image_format", "") != "jpeg":
                return False
            first_frame = handle["frames"][0]
            if len(first_frame) == 0:
                return False
    except Exception:
        return False
    return True


def write_hdf5_episode(
    output_path: Path,
    encoded_frames: Sequence[np.ndarray],
    instruction: str,
    episode_index: int,
    video_path: Path,
    split_name: str,
    stored_size: Tuple[int, int],
    jpeg_quality: int,
    frame_transform: str,
) -> None:
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    zero_state = np.zeros((len(encoded_frames), 7), dtype=np.float32)
    zero_action = np.zeros((len(encoded_frames), 7), dtype=np.float32)

    with h5py.File(output_path, "w") as handle:
        frames_ds = handle.create_dataset("frames", shape=(len(encoded_frames),), dtype=vlen_uint8)
        for i, frame_bytes in enumerate(encoded_frames):
            frames_ds[i] = frame_bytes
        handle.create_dataset("state", data=zero_state, compression="gzip", compression_opts=1)
        handle.create_dataset("action", data=zero_action, compression="gzip", compression_opts=1)
        handle.attrs["instruction"] = instruction
        handle.attrs["episode_index"] = episode_index
        handle.attrs["video_path"] = str(video_path)
        handle.attrs["split"] = split_name
        handle.attrs["image_format"] = "jpeg"
        handle.attrs["jpeg_quality"] = jpeg_quality
        handle.attrs["frame_transform"] = frame_transform
        handle.attrs["target_height"] = stored_size[0]
        handle.attrs["target_width"] = stored_size[1]
        handle.attrs["num_frames"] = len(encoded_frames)


def convert_episode(
    task: Tuple[str, int, Dict[str, object], str, str, str, Tuple[int, int], int, str, bool]
) -> Dict[str, object]:
    split_name, output_index, row, data_root, video_key, split_dir, target_size, jpeg_quality, frame_transform, skip_existing = task
    data_root_path = Path(data_root)
    split_dir_path = Path(split_dir)
    output_path = split_dir_path / f"episode_{output_index:08d}.h5"
    video_path = data_root_path / str(row[video_key])

    result = {
        "split": split_name,
        "output_index": output_index,
        "episode_index": int(row["episode_index"]),
        "instruction": row["instruction"],
        "video_path": str(video_path),
        "output_path": str(output_path),
    }

    if skip_existing and is_valid_hdf5_episode(output_path):
        result["status"] = "skipped_existing"
        return result

    if output_path.exists() and not is_valid_hdf5_episode(output_path):
        output_path.unlink()

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"DROID video not found: {video_path}")
        if video_path.stat().st_size == 0:
            raise OSError(f"DROID video is empty: {video_path}")

        encoded_frames, stored_size = decode_and_encode_video(video_path, target_size, jpeg_quality, frame_transform)
        write_hdf5_episode(
            output_path=output_path,
            encoded_frames=encoded_frames,
            instruction=str(row["instruction"]),
            episode_index=int(row["episode_index"]),
            video_path=video_path,
            split_name=split_name,
            stored_size=stored_size,
            jpeg_quality=jpeg_quality,
            frame_transform=frame_transform,
        )
        result["num_frames"] = len(encoded_frames)
        result["status"] = "converted"
        return result
    except Exception as exc:
        if output_path.exists() and not is_valid_hdf5_episode(output_path):
            output_path.unlink()
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def iter_with_progress(tasks: Sequence[Tuple], num_workers: int, desc: str):
    total = len(tasks)
    progress = tqdm(total=total, desc=desc, unit="ep") if tqdm is not None else None
    try:
        if num_workers <= 1:
            for task in tasks:
                result = convert_episode(task)
                if progress is not None:
                    progress.update(1)
                yield result
            return

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunksize = max(1, min(64, total // max(num_workers * 8, 1)))
            for result in executor.map(convert_episode, tasks, chunksize=chunksize):
                if progress is not None:
                    progress.update(1)
                yield result
    finally:
        if progress is not None:
            progress.close()


def output_size_summary(output_dir: Path) -> int:
    total_bytes = 0
    for path in output_dir.rglob("*"):
        if path.is_file():
            total_bytes += path.stat().st_size
    return total_bytes


def build_tasks(
    rows: Sequence[Dict[str, object]],
    split_name: str,
    output_dir: Path,
    data_root: Path,
    video_key: str,
    target_size: Tuple[int, int],
    jpeg_quality: int,
    frame_transform: str,
    skip_existing: bool,
) -> List[Tuple[str, int, Dict[str, object], str, str, str, Tuple[int, int], int, str, bool]]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return [
        (
            split_name,
            output_index,
            row,
            str(data_root),
            video_key,
            str(split_dir),
            target_size,
            jpeg_quality,
            frame_transform,
            skip_existing,
        )
        for output_index, row in enumerate(rows)
    ]


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    jsonl_path = resolve_jsonl_path(data_root, args.jsonl_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(jsonl_path, args.video_key, args.max_episodes)
    train_rows, test_rows = split_rows(rows, args.split_ratio, split_group_key=args.split_group_key)
    target_size = tuple(args.target_size)

    train_tasks = build_tasks(
        train_rows,
        split_name="train",
        output_dir=output_dir,
        data_root=data_root,
        video_key=args.video_key,
        target_size=target_size,
        jpeg_quality=args.jpeg_quality,
        frame_transform=args.frame_transform,
        skip_existing=args.skip_existing,
    )
    test_tasks = build_tasks(
        test_rows,
        split_name="test",
        output_dir=output_dir,
        data_root=data_root,
        video_key=args.video_key,
        target_size=target_size,
        jpeg_quality=args.jpeg_quality,
        frame_transform=args.frame_transform,
        skip_existing=args.skip_existing,
    )

    start_time = time.time()
    train_manifest = list(iter_with_progress(train_tasks, args.num_workers, desc="train split"))
    test_manifest = list(iter_with_progress(test_tasks, args.num_workers, desc="test split"))

    train_manifest.sort(key=lambda entry: entry["output_index"])
    test_manifest.sort(key=lambda entry: entry["output_index"])

    train_converted = sum(entry["status"] == "converted" for entry in train_manifest)
    train_skipped = sum(entry["status"] == "skipped_existing" for entry in train_manifest)
    train_failed = sum(entry["status"] == "failed" for entry in train_manifest)
    test_converted = sum(entry["status"] == "converted" for entry in test_manifest)
    test_skipped = sum(entry["status"] == "skipped_existing" for entry in test_manifest)
    test_failed = sum(entry["status"] == "failed" for entry in test_manifest)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "jsonl_path": str(jsonl_path),
                "video_key": args.video_key,
                "target_size": list(target_size),
                "jpeg_quality": args.jpeg_quality,
                "frame_transform": args.frame_transform,
                "split_ratio": args.split_ratio,
                "split_group_key": args.split_group_key,
                "num_workers": args.num_workers,
                "skip_existing": args.skip_existing,
                "max_episodes": args.max_episodes,
                "num_train_episodes": len(train_rows),
                "num_test_episodes": len(test_rows),
                "train_episodes": train_manifest,
                "test_episodes": test_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = time.time() - start_time
    total_bytes = output_size_summary(output_dir)
    print("DROID full HDF5 conversion summary")
    print(f"  output_dir: {output_dir}")
    print(f"  manifest: {manifest_path}")
    print(f"  jpeg_quality: {args.jpeg_quality}")
    print(f"  frame_transform: {args.frame_transform}")
    print(f"  split_ratio: {args.split_ratio}")
    print(f"  split_group_key: {args.split_group_key}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  skip_existing: {args.skip_existing}")
    print(f"  train_converted: {train_converted}")
    print(f"  train_skipped: {train_skipped}")
    print(f"  train_failed: {train_failed}")
    print(f"  test_converted: {test_converted}")
    print(f"  test_skipped: {test_skipped}")
    print(f"  test_failed: {test_failed}")
    print(f"  elapsed_sec: {elapsed:.2f}")
    print(f"  output_bytes: {total_bytes}")


if __name__ == "__main__":
    main()
