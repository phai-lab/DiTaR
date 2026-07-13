#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg


REPO_ID = "agibot-world/AgiBotWorld-Beta"
TASK_ID_RE = re.compile(r"observations/(\d+)/")
KEEP_CAMERAS = ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4")
LOCK_STALE_SECONDS = 30 * 60


def run(cmd):
    subprocess.run(cmd, check=True)


def safe_name(path):
    return path.replace("/", "__")


def tar_tmp_dir(root, tar_rel):
    return root / "tmp" / safe_name(tar_rel)


def cleanup_tar_tmp_dir(root, tar_rel):
    shutil.rmtree(tar_tmp_dir(root, tar_rel), ignore_errors=True)


def cleanup_successful_tar(root, tar_rel, tar_path):
    tar_path.unlink(missing_ok=True)
    cleanup_tar_tmp_dir(root, tar_rel)


def probe_video(ffmpeg, path):
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        check=False,
        text=True,
        capture_output=True,
    )
    text = proc.stderr + proc.stdout
    match = re.search(r"Video:.*? (\d{2,5})x(\d{2,5})", text)
    fps_match = re.search(r"(\d+(?:\.\d+)?) fps", text)
    return {
        "width": int(match.group(1)) if match else None,
        "height": int(match.group(2)) if match else None,
        "fps": float(fps_match.group(1)) if fps_match else None,
    }


def convert_video(ffmpeg, src, dst, width, crf):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(src),
            "-vf",
            f"scale={width}:-2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-threads",
            "1",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )


def load_task_info(task_info_dir):
    by_task = {}
    for path in sorted(task_info_dir.glob("task_*.json")):
        task_id = int(path.stem.split("_")[1])
        records = json.loads(path.read_text())
        by_task[task_id] = {int(item["episode_id"]): item for item in records}
    return by_task


def load_completed_episodes(manifest_path):
    completed = set()
    if not manifest_path.exists():
        return completed
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        completed.add((int(item["task_id"]), int(item["episode_id"])))
    return completed


def append_jsonl(path, item):
    with path.open("a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_task_id(tar_rel):
    match = TASK_ID_RE.search(tar_rel)
    if not match:
        raise ValueError(f"Could not parse task id from {tar_rel}")
    return int(match.group(1))


class DownloadTimeout(Exception):
    pass


_DOWNLOAD_HELPER = """
import os, sys
from huggingface_hub import hf_hub_download
repo_id, tar_rel, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
path = hf_hub_download(
    repo_id=repo_id,
    filename=tar_rel,
    repo_type="dataset",
    local_dir=local_dir,
    token=os.environ["HF_TOKEN"],
    etag_timeout=30,
)
sys.stdout.write(path + "\\n")
sys.stdout.flush()
"""


def download_tar(root, tar_rel, token, timeout_seconds=900):
    """Download a tar in an isolated subprocess so the kernel can enforce the timeout.

    HF's resumable download caches partial bytes via .incomplete sidecars, so a
    killed subprocess does not lose progress: the next attempt resumes.
    """
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HF_TOKEN": token,
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "120"),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DOWNLOAD_HELPER, REPO_ID, tar_rel, str(raw_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise DownloadTimeout(
            f"hf_hub_download exceeded {timeout_seconds}s for {tar_rel}"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(
            f"download_tar failed rc={proc.returncode} for {tar_rel}: {' | '.join(tail)}"
        )
    out_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not out_lines:
        raise RuntimeError(f"download_tar produced no output for {tar_rel}")
    return Path(out_lines[-1])


def claim_tar_lock(root, tar_rel):
    lock_marker = root / "lock_tars" / f"{safe_name(tar_rel)}.lock"
    try:
        lock_marker.mkdir()
        (lock_marker / "claimed_at").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        return lock_marker
    except FileExistsError:
        try:
            age = time.time() - lock_marker.stat().st_mtime
        except FileNotFoundError:
            return claim_tar_lock(root, tar_rel)
        if age > LOCK_STALE_SECONDS:
            shutil.rmtree(lock_marker, ignore_errors=True)
            return claim_tar_lock(root, tar_rel)
        return None


def _transcode_one(ffmpeg, tmp, out, width, crf, tar_rel, task_id, episode_id, camera):
    try:
        source_video = probe_video(ffmpeg, tmp)
        source_bytes = tmp.stat().st_size
        convert_video(ffmpeg, tmp, out, width, crf)
        lowres_video = probe_video(ffmpeg, out)
        lowres_bytes = out.stat().st_size
        return {
            "ok": True,
            "task_id": task_id,
            "episode_id": episode_id,
            "camera": camera,
            "source_bytes": source_bytes,
            "lowres_bytes": lowres_bytes,
            "source_video": source_video,
            "lowres_video": lowres_video,
            "source_tar": tar_rel,
        }
    except subprocess.CalledProcessError as exc:
        out.unlink(missing_ok=True)
        return {
            "ok": False,
            "task_id": task_id,
            "episode_id": episode_id,
            "camera": camera,
            "source_tar": tar_rel,
            "reason": f"ffmpeg_failed: {exc!r}",
        }
    finally:
        tmp.unlink(missing_ok=True)


def process_tar(root, tar_rel, tar_path, task_info, completed, width, crf, workers):
    task_id = get_task_id(tar_rel)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_dir = tar_tmp_dir(root, tar_rel)
    lowres_root = root / "lowres"
    stats_path = root / "stats.jsonl"
    manifest_path = root / "manifest.jsonl"
    skipped_path = root / "skipped_episodes.jsonl"
    task_records = task_info.get(task_id, {})

    episode_members = defaultdict(dict)
    extract_jobs = []  # (episode_id, camera, tmp, out)
    episode_out_paths = {}  # episode_id -> {camera: out_path}

    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[1] == "videos" and parts[2] in KEEP_CAMERAS:
                episode_members[int(parts[0])][parts[2]] = member

        for episode_id in sorted(episode_members):
            members = episode_members[episode_id]
            if not all(camera in members for camera in KEEP_CAMERAS):
                continue

            out_paths = {
                camera: lowres_root / f"task_{task_id}" / str(episode_id) / "videos" / camera
                for camera in KEEP_CAMERAS
            }
            episode_out_paths[episode_id] = out_paths

            episode_key = (task_id, episode_id)
            if episode_key in completed:
                continue

            for camera, member in members.items():
                out = out_paths[camera]
                if out.exists() and out.stat().st_size > 0:
                    continue
                tmp = tmp_dir / f"task_{task_id}" / member.name
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extract_jobs.append((episode_id, camera, tmp, out))

    failed_episodes = set()
    if extract_jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _transcode_one,
                    ffmpeg,
                    tmp,
                    out,
                    width,
                    crf,
                    tar_rel,
                    task_id,
                    episode_id,
                    camera,
                )
                for (episode_id, camera, tmp, out) in extract_jobs
            ]
            for fut in as_completed(futures):
                result = fut.result()
                if result["ok"]:
                    append_jsonl(stats_path, {k: v for k, v in result.items() if k != "ok"})
                else:
                    failed_episodes.add(result["episode_id"])
                    append_jsonl(
                        skipped_path,
                        {**{k: v for k, v in result.items() if k != "ok"},
                         "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                    )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    for episode_id, out_paths in episode_out_paths.items():
        episode_key = (task_id, episode_id)
        if episode_key in completed:
            continue
        if episode_id in failed_episodes:
            continue
        if not all(path.exists() and path.stat().st_size > 0 for path in out_paths.values()):
            continue
        episode = task_records.get(episode_id, {})
        append_jsonl(
            manifest_path,
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "task_name": episode.get("task_name"),
                "init_scene_text": episode.get("init_scene_text"),
                "videos": {
                    camera: str(out_paths[camera].relative_to(root))
                    for camera in KEEP_CAMERAS
                },
                "instructions": episode.get("label_info", {}).get("action_config", []),
            },
        )
        completed.add(episode_key)


def summarize(root):
    manifest = root / "manifest.jsonl"
    stats = root / "stats.jsonl"
    episodes = 0
    videos = 0
    source_bytes = 0
    lowres_bytes = 0
    if manifest.exists():
        episodes = sum(1 for line in manifest.read_text().splitlines() if line.strip())
    if stats.exists():
        for line in stats.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            videos += 1
            source_bytes += int(item.get("source_bytes", 0))
            lowres_bytes += int(item.get("lowres_bytes", 0))
    summary = {
        "episodes": episodes,
        "videos_with_stats": videos,
        "source_gib_with_stats": source_bytes / 1024**3,
        "lowres_gib_with_stats": lowres_bytes / 1024**3,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tar-list", type=Path, required=True)
    parser.add_argument("--task-info-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--sleep-after-download", type=float, default=20.0)
    parser.add_argument("--max-runtime-hours", type=float, default=46.0)
    parser.add_argument("--limit-tars", type=int, default=0)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Clear failed_tars markers before iterating so prior failures get re-attempted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel ffmpeg transcode workers per tar.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=900,
        help="Per-tar hf_hub_download wall-clock timeout in seconds.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    root = args.root
    for subdir in ("raw", "tmp", "lowres", "done_tars", "failed_tars", "lock_tars", "logs"):
        (root / subdir).mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    max_seconds = args.max_runtime_hours * 3600
    task_info = load_task_info(args.task_info_dir)
    completed = load_completed_episodes(root / "manifest.jsonl")
    tar_rels = [line.strip() for line in args.tar_list.read_text().splitlines() if line.strip()]

    if args.retry_failed:
        for marker in (root / "failed_tars").glob("*.failed"):
            marker.unlink()

    processed_this_run = 0
    for tar_rel in tar_rels:
        if args.limit_tars and processed_this_run >= args.limit_tars:
            break
        if time.monotonic() - started > max_seconds:
            break

        done_marker = root / "done_tars" / f"{safe_name(tar_rel)}.done"
        fail_marker = root / "failed_tars" / f"{safe_name(tar_rel)}.failed"
        if done_marker.exists():
            continue
        lock_marker = claim_tar_lock(root, tar_rel)
        if lock_marker is None:
            continue

        print(f"[{time.strftime('%F %T')}] download {tar_rel}", flush=True)
        tar_path = None
        try:
            tar_path = download_tar(root, tar_rel, token, timeout_seconds=args.download_timeout)
            time.sleep(args.sleep_after_download)
            print(f"[{time.strftime('%F %T')}] process {tar_rel}", flush=True)
            process_tar(
                root,
                tar_rel,
                tar_path,
                task_info,
                completed,
                args.width,
                args.crf,
                args.workers,
            )
            done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            fail_marker.unlink(missing_ok=True)
            cleanup_successful_tar(root, tar_rel, tar_path)
            summarize(root)
            processed_this_run += 1
            print(f"[{time.strftime('%F %T')}] done {tar_rel}", flush=True)
        except Exception as exc:
            fail_marker.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {exc!r}\n")
            cleanup_tar_tmp_dir(root, tar_rel)
            if tar_path is not None:
                tar_path.unlink(missing_ok=True)
            summarize(root)
            print(f"[{time.strftime('%F %T')}] failed {tar_rel}: {exc!r}", flush=True)
            continue
        finally:
            shutil.rmtree(lock_marker, ignore_errors=True)

    summarize(root)


if __name__ == "__main__":
    main()
