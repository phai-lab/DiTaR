"""Pre-extract AgiBot Beta head-camera videos to per-episode JPEG/HDF5 files.

Output layout:
  <out_root>/task_<id>/episode_<id>.h5

Each h5 file mirrors DROIDHDF5Dataset's schema:
  frames : variable-length np.bytes_ dataset of JPEG bytes (one entry per kept frame)
  attrs  : num_frames, source_fps, output_fps, subsample_stride,
           task_id, episode_id, source_video, target_height, target_width,
           jpeg_quality, image_format ('jpeg'), instruction (left empty;
           per-segment text lives in the original manifest)

Resumability
  Each episode writes to a temp file then atomic-renames into place. If the
  final h5 already exists and is well-formed (`num_frames` attr present and
  `frames` dataset readable), the worker skips it. So `Ctrl-C` / `kill` mid-run
  followed by re-launch picks up where it left off.

Parallelism
  Top-level uses `multiprocessing.Pool(num_workers)` over the episode list;
  default 16 workers (host has 96 CPUs but is shared with other users — be
  polite). Adjust with `--num-workers` if needed.

Liveness across logout
  This script does NOT background itself. Wrap the invocation in `nohup ...
  &` and `disown` (or run via `setsid`) before logging out — that detaches it
  from the SSH session so HUP doesn't kill it.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import decord
import h5py
import numpy as np
from PIL import Image

def _ditar_data_root() -> str:
    """Base dir for datasets; set DITAR_DATA_ROOT (see TRAINING.md).

    Falls back to the current directory so ``--help`` works and explicit
    --data-root / --out-root arguments always win.
    """
    return os.environ.get("DITAR_DATA_ROOT", "").strip() or "."


# Avoid noisy ffmpeg log spam (we still get exceptions on real failures).
decord.bridge.set_bridge('native')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data-root',
                   default=os.path.join(_ditar_data_root(), 'data', 'agibot_beta_threeview_320'),
                   help='root holding manifest.jsonl + the low-res videos')
    p.add_argument('--manifest', default='manifest.jsonl', help='Relative or absolute path to AgiBot manifest jsonl.')
    p.add_argument('--video-key', default='head_color.mp4')
    p.add_argument('--out-root',
                   default=os.path.join(_ditar_data_root(), 'data', 'agibot_beta_threeview_320', 'extracted_jpeg_h5'),
                   help='output HDF5 root (the training config reads this)')
    p.add_argument('--target-size', type=int, default=256)
    p.add_argument('--subsample-stride', type=int, default=2, help='Keep every Nth frame (2 -> 30fps source becomes 15fps).')
    p.add_argument('--source-fps', type=float, default=30.0, help='AgiBot source fps (only used as h5 attr).')
    p.add_argument('--jpeg-quality', type=int, default=85)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--log-every', type=int, default=200, help='Print progress every N completed episodes.')
    p.add_argument('--limit', type=int, default=0, help='Process only the first N episodes (0 = all). Useful for smoke tests.')
    p.add_argument('--state-file', default=None, help='JSON state path for progress / failure log. Defaults to <out_root>/.extract_state.json')
    p.add_argument('--failed-list', default=None, help='Plain-text path for the list of failed episodes (one per line). Defaults to <out_root>/.extract_failed.txt')
    p.add_argument('--decode-batch', type=int, default=64, help='Decord get_batch chunk size (memory budget per worker).')
    return p.parse_args()


def encode_frame(frame: np.ndarray, target_size: int, quality: int) -> bytes:
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    img = Image.fromarray(frame)
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def is_complete_h5(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, 'r') as h:
            if 'frames' not in h:
                return False
            n = int(h.attrs.get('num_frames', -1))
            if n <= 0:
                return False
            if h['frames'].shape[0] != n:
                return False
            return True
    except Exception:
        return False


def process_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Decode one episode; write per-episode h5; return status."""
    src = Path(job['src'])
    out = Path(job['out'])
    target_size = int(job['target_size'])
    stride = int(job['stride'])
    quality = int(job['quality'])
    decode_batch = int(job['decode_batch'])
    task_id = int(job['task_id'])
    episode_id = int(job['episode_id'])
    source_fps = float(job['source_fps'])
    rel_video = str(job['rel_video'])

    # Idempotent skip.
    if is_complete_h5(out):
        return {'episode_id': episode_id, 'status': 'skipped', 'frames': 0}

    if not src.exists():
        return {'episode_id': episode_id, 'status': 'missing_source', 'frames': 0, 'error': str(src)}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.h5.tmp')
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    try:
        vr = decord.VideoReader(str(src))
    except Exception as error:
        return {'episode_id': episode_id, 'status': 'open_failed', 'frames': 0, 'error': repr(error)}

    n_total = len(vr)
    keep_indices = list(range(0, n_total, stride))
    n_keep = len(keep_indices)
    if n_keep == 0:
        return {'episode_id': episode_id, 'status': 'empty_video', 'frames': 0}

    jpeg_records: List[bytes] = []
    try:
        for chunk_start in range(0, n_keep, decode_batch):
            chunk_idx = keep_indices[chunk_start:chunk_start + decode_batch]
            try:
                batch = vr.get_batch(chunk_idx).asnumpy()
            except Exception:
                # Per-frame fallback for partial corruption.
                batch_list = []
                for fi in chunk_idx:
                    try:
                        batch_list.append(np.asarray(vr[fi]))
                    except Exception as inner_err:
                        if batch_list:
                            batch_list.append(batch_list[-1])
                        else:
                            raise inner_err
                batch = np.stack(batch_list, axis=0)
            for f_idx in range(batch.shape[0]):
                jpeg_records.append(encode_frame(np.asarray(batch[f_idx], dtype=np.uint8), target_size, quality))
    except Exception as error:
        return {'episode_id': episode_id, 'status': 'decode_failed', 'frames': 0, 'error': repr(error)}

    if len(jpeg_records) != n_keep:
        return {'episode_id': episode_id, 'status': 'frame_count_mismatch', 'frames': len(jpeg_records),
                'error': f'expected {n_keep} got {len(jpeg_records)}'}

    try:
        with h5py.File(tmp, 'w') as h:
            dt = h5py.vlen_dtype(np.uint8)
            ds = h.create_dataset('frames', (n_keep,), dtype=dt, chunks=(1,))
            for i, b in enumerate(jpeg_records):
                ds[i] = np.frombuffer(b, dtype=np.uint8)
            h.attrs['num_frames'] = n_keep
            h.attrs['source_fps'] = source_fps
            h.attrs['output_fps'] = source_fps / stride
            h.attrs['subsample_stride'] = stride
            h.attrs['task_id'] = task_id
            h.attrs['episode_id'] = episode_id
            h.attrs['source_video'] = rel_video
            h.attrs['target_height'] = target_size
            h.attrs['target_width'] = target_size
            h.attrs['jpeg_quality'] = quality
            h.attrs['image_format'] = 'jpeg'
        os.replace(tmp, out)
    except Exception as error:
        try:
            tmp.unlink()
        except Exception:
            pass
        return {'episode_id': episode_id, 'status': 'write_failed', 'frames': 0, 'error': repr(error)}

    return {'episode_id': episode_id, 'status': 'ok', 'frames': n_keep}


def build_jobs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], int]:
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = data_root / manifest_path
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    skipped = 0
    seen = 0
    with manifest_path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            videos = ep.get('videos') or {}
            rel_video = videos.get(args.video_key)
            if not rel_video:
                continue
            seen += 1
            task_id = int(ep.get('task_id', -1))
            episode_id = int(ep.get('episode_id', -1))
            out_path = out_root / f'task_{task_id}' / f'episode_{episode_id}.h5'
            if is_complete_h5(out_path):
                skipped += 1
                continue
            src_path = data_root / rel_video
            jobs.append(
                dict(
                    src=str(src_path),
                    out=str(out_path),
                    target_size=args.target_size,
                    stride=args.subsample_stride,
                    quality=args.jpeg_quality,
                    decode_batch=args.decode_batch,
                    task_id=task_id,
                    episode_id=episode_id,
                    source_fps=args.source_fps,
                    rel_video=rel_video,
                )
            )
            if args.limit and len(jobs) >= args.limit:
                break
    return jobs, skipped, seen


def init_worker():
    # Children should not handle Ctrl-C; let parent kill them.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_file) if args.state_file else Path(args.out_root) / '.extract_state.json'
    failed_path = Path(args.failed_list) if args.failed_list else Path(args.out_root) / '.extract_failed.txt'

    print(f'[extract] data_root={args.data_root}')
    print(f'[extract] out_root={args.out_root}')
    print(f'[extract] target_size={args.target_size} stride={args.subsample_stride} quality={args.jpeg_quality}')
    print(f'[extract] num_workers={args.num_workers} decode_batch={args.decode_batch}')
    print(f'[extract] state_file={state_path}')
    print(f'[extract] failed_list={failed_path}')
    sys.stdout.flush()

    t0 = time.monotonic()
    jobs, skipped, seen = build_jobs(args)
    print(f'[extract] manifest scan: seen={seen} already_done={skipped} todo={len(jobs)} (in {time.monotonic()-t0:.1f}s)')
    sys.stdout.flush()
    if not jobs:
        print('[extract] nothing to do (all complete or no eligible episodes).')
        return

    state_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_fh = failed_path.open('a', buffering=1)

    completed = 0
    failed = 0
    frame_total = 0
    last_log_at = time.monotonic()

    with mp.Pool(args.num_workers, initializer=init_worker) as pool:
        try:
            for result in pool.imap_unordered(process_one, jobs, chunksize=4):
                if result['status'] in ('ok', 'skipped'):
                    completed += 1
                    frame_total += int(result.get('frames', 0))
                else:
                    failed += 1
                    failed_fh.write(json.dumps(result) + '\n')
                if completed % args.log_every == 0 and (time.monotonic() - last_log_at) > 5.0:
                    elapsed = time.monotonic() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta = (len(jobs) - completed - failed) / max(rate, 1e-6)
                    print(
                        f'[extract] done={completed}/{len(jobs)} failed={failed} '
                        f'frames={frame_total:,} elapsed={elapsed/60:.1f}min '
                        f'rate={rate:.2f} ep/s eta={eta/60:.1f}min',
                        flush=True,
                    )
                    last_log_at = time.monotonic()
                    state_path.write_text(json.dumps({
                        'completed': completed,
                        'failed': failed,
                        'frames': frame_total,
                        'todo_initial': len(jobs),
                        'elapsed_seconds': time.monotonic() - t0,
                        'pid': os.getpid(),
                    }) + '\n')
        except KeyboardInterrupt:
            print('[extract] caught SIGINT — terminating pool.')
            pool.terminate()
            pool.join()
            raise

    failed_fh.close()
    elapsed = time.monotonic() - t0
    print(
        f'[extract] FINISHED. completed={completed} failed={failed} frames_written={frame_total:,} '
        f'elapsed={elapsed/60:.2f} min',
        flush=True,
    )
    state_path.write_text(json.dumps({
        'completed': completed,
        'failed': failed,
        'frames': frame_total,
        'todo_initial': len(jobs),
        'elapsed_seconds': elapsed,
        'pid': os.getpid(),
        'status': 'finished',
    }) + '\n')


if __name__ == '__main__':
    main()
