import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def _ditar_data_root() -> Path:
    """Base dir for datasets; set DITAR_DATA_ROOT (see TRAINING.md).

    Falls back to the current directory so ``--help`` works and explicit
    --source-root / --output-root arguments always win.
    """
    return Path(os.environ.get("DITAR_DATA_ROOT", "").strip() or ".")


# Expert frame dirs rendered from MetaWorld-v3 (one dir of PNG/JPG frames per episode).
DEFAULT_SOURCE_ROOT = _ditar_data_root() / "data" / "metaworldv3_expert_frames"
# Task -> instruction map. Shipped at the repo root of this release.
DEFAULT_INSTRUCTION_JSON = (
    Path(__file__).resolve().parents[3] / "metaworld_v3_language_instructions.json"
)
DEFAULT_OUTPUT_ROOT = _ditar_data_root() / "data" / "metaworldv3"
MANIFEST_NAME = "episode_instructions.jsonl"
METADATA_NAME = "metadata.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_instruction_map(instruction_json_path: str | Path) -> dict[str, str]:
    instruction_json_path = Path(instruction_json_path)
    with instruction_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    instruction_map = {}
    for item in data["tasks"]:
        task_name = item["task_name"]
        instruction = item["instruction_en"].strip()
        if not instruction:
            raise ValueError(f"task {task_name} has an empty instruction")
        instruction_map[task_name] = instruction
    return instruction_map


def _episode_sort_key(episode_dir: Path) -> tuple[int, str]:
    try:
        return (int(episode_dir.name), episode_dir.name)
    except ValueError:
        return (10**12, episode_dir.name)


def _iter_episodes(source_root: str | Path, instruction_map: dict[str, str]):
    source_root = Path(source_root)
    for task_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        task_name = task_dir.name
        if task_name not in instruction_map:
            raise KeyError(f"missing instruction for task {task_name}")
        for split_name in ("train", "test"):
            split_dir = task_dir / split_name
            if not split_dir.is_dir():
                continue
            for episode_dir in sorted((path for path in split_dir.iterdir() if path.is_dir()), key=_episode_sort_key):
                yield {
                    "task_name": task_name,
                    "instruction": instruction_map[task_name],
                    "split": split_name,
                    "source_episode_id": episode_dir.name,
                    "episode_dir": episode_dir,
                }


def _frame_sort_key(frame_path: Path) -> tuple[int, str]:
    try:
        return (int(frame_path.stem), frame_path.name)
    except ValueError:
        return (10**12, frame_path.name)


def _sorted_frame_paths(episode_dir: Path) -> list[Path]:
    frame_paths = sorted(episode_dir.glob("*.png"), key=_frame_sort_key)
    if not frame_paths:
        raise ValueError(f"no PNG frames found in {episode_dir}")
    return frame_paths


def _load_frames(frame_paths: list[Path]) -> list[np.ndarray]:
    frames = []
    expected_size = None
    for frame_path in frame_paths:
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        if expected_size is None:
            expected_size = frame.shape[:2]
        elif frame.shape[:2] != expected_size:
            raise ValueError(
                f"inconsistent frame size in {frame_path.parent}: expected {expected_size}, got {frame.shape[:2]}"
            )
        frames.append(frame)
    return frames


def _write_video(video_path: Path, frames: list[np.ndarray], fps: int, codec: str) -> None:
    _ensure_dir(video_path.parent)
    writer = imageio.get_writer(video_path, fps=fps, codec=codec, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def export_dataset(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    instruction_json_path: str | Path = DEFAULT_INSTRUCTION_JSON,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    fps: int = 10,
    codec: str = "mpeg4",
    limit: int | None = None,
) -> dict:
    source_root = Path(source_root)
    output_root = Path(output_root)
    videos_dir = output_root / "videos"
    _ensure_dir(videos_dir)

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    instruction_map = _load_instruction_map(instruction_json_path)
    episodes = list(_iter_episodes(source_root, instruction_map))
    if limit is not None:
        episodes = episodes[:limit]

    manifest_path = output_root / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for episode_index, episode in enumerate(episodes):
            frame_paths = _sorted_frame_paths(episode["episode_dir"])
            frames = _load_frames(frame_paths)
            video_relpath = Path("videos") / f"{episode_index}.mp4"
            video_path = output_root / video_relpath
            _write_video(video_path=video_path, frames=frames, fps=fps, codec=codec)

            row = {
                "episode_index": episode_index,
                "instruction": episode["instruction"],
                "video": video_relpath.as_posix(),
                "task_name": episode["task_name"],
            }
            manifest_file.write(json.dumps(row, ensure_ascii=True) + "\n")

            if (episode_index + 1) % 100 == 0 or episode_index + 1 == len(episodes):
                print(
                    json.dumps(
                        {
                            "progress": episode_index + 1,
                            "total": len(episodes),
                            "latest_video": video_relpath.as_posix(),
                            "task_name": episode["task_name"],
                        }
                    )
                )

    metadata = {
        "source_root": str(source_root),
        "instruction_json_path": str(Path(instruction_json_path)),
        "output_root": str(output_root),
        "manifest_path": MANIFEST_NAME,
        "videos_dir": "videos",
        "fps": fps,
        "codec": codec,
        "num_tasks": len({episode["task_name"] for episode in episodes}),
        "num_episodes": len(episodes),
        "tasks": sorted({episode["task_name"] for episode in episodes}),
    }
    (output_root / METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten MetaWorld v3 task-specific image episodes into MP4 videos plus instruction JSONL."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Root directory with task/split/episode/frame layout.")
    parser.add_argument(
        "--instruction-json",
        default=str(DEFAULT_INSTRUCTION_JSON),
        help="Path to the MetaWorld instruction mapping JSON.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output dataset directory.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output videos.")
    parser.add_argument("--codec", default="mpeg4", help="Video codec passed to imageio ffmpeg writer.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of episodes to export for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = export_dataset(
        source_root=args.source_root,
        instruction_json_path=args.instruction_json,
        output_root=args.output_root,
        fps=args.fps,
        codec=args.codec,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
