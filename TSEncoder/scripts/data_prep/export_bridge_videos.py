import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "bridge" / "raw"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "bridge_multiview"
MANIFEST_NAME = "episode_instruction_multiview.jsonl"
METADATA_NAME = "metadata.json"
REQUIRED_CAMERAS = ("images0", "images1", "images2")
EXPORTED_CAMERAS = ("images1", "images2")
FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_instruction(lang_path: str | Path) -> str:
    lang_path = Path(lang_path)
    lines = []
    for raw_line in lang_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("confidence:"):
            continue
        lines.append(line)

    if not lines:
        raise ValueError(f"no usable instruction found in {lang_path}")
    return lines[0]


def _iter_episodes(source_root: str | Path):
    source_root = Path(source_root)
    lang_paths = sorted(source_root.rglob("lang.txt"), key=lambda path: path.relative_to(source_root).as_posix())
    for lang_path in lang_paths:
        traj_dir = lang_path.parent
        if not all((traj_dir / camera_name).is_dir() for camera_name in REQUIRED_CAMERAS):
            continue
        rel_traj_path = traj_dir.relative_to(source_root)
        yield {
            "instruction": _load_instruction(lang_path),
            "traj_dir": traj_dir,
            "rel_traj_path": rel_traj_path,
        }


def _frame_sort_key(frame_path: Path) -> tuple[int, str]:
    stem = frame_path.stem
    suffix = stem.rsplit("_", 1)[-1]
    try:
        return (int(suffix), frame_path.name)
    except ValueError:
        return (10**12, frame_path.name)


def _sorted_frame_paths(camera_dir: Path) -> list[Path]:
    frame_paths = sorted(
        (
            path
            for path in camera_dir.iterdir()
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in FRAME_SUFFIXES
        ),
        key=_frame_sort_key,
    )
    if not frame_paths:
        raise ValueError(f"no frames found in {camera_dir}")
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

    episodes = []
    for episode in _iter_episodes(source_root):
        episodes.append(episode)
        if limit is not None and len(episodes) >= limit:
            break

    manifest_path = output_root / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for episode_index, episode in enumerate(episodes, start=1):
            row = {
                "episode_index": episode_index,
                "instruction": episode["instruction"],
            }

            for camera_name in EXPORTED_CAMERAS:
                frame_paths = _sorted_frame_paths(episode["traj_dir"] / camera_name)
                frames = _load_frames(frame_paths)
                video_relpath = Path("videos") / episode["rel_traj_path"] / f"{camera_name}.mp4"
                _write_video(output_root / video_relpath, frames=frames, fps=fps, codec=codec)
                row[f"video_{camera_name}"] = video_relpath.as_posix()

            manifest_file.write(json.dumps(row, ensure_ascii=True) + "\n")

            if episode_index % 100 == 0 or episode_index == len(episodes):
                print(
                    json.dumps(
                        {
                            "progress": episode_index,
                            "total": len(episodes),
                            "latest_images1": row["video_images1"],
                            "latest_images2": row["video_images2"],
                        }
                    )
                )

    metadata = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "manifest_path": MANIFEST_NAME,
        "videos_dir": "videos",
        "fps": fps,
        "codec": codec,
        "num_episodes": len(episodes),
    }
    (output_root / METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Bridge three-view trajectories into two MP4 views plus a multiview instruction manifest."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Root directory containing Bridge raw trajectories.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output dataset directory.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output videos.")
    parser.add_argument("--codec", default="mpeg4", help="Video codec passed to imageio ffmpeg writer.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of episodes to export for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = export_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        fps=args.fps,
        codec=args.codec,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
