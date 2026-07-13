"""Pre-extract Bridge multiview videos to per-episode JPEG/HDF5 files in
DROID's HDF5 schema, so DROIDHDF5Dataset can read them as-is.

Output layout (matches DROIDHDF5Dataset's expectation):
  <out_root>/train/episode_<NNNNNNNN>.h5     (first split_ratio fraction by sorted episode_index)
  <out_root>/test/episode_<NNNNNNNN>.h5      (the rest)

Each h5 file:
  frames    : vlen uint8 dataset of JPEG-encoded frames (1 entry per frame)
  action    : float32 [num_frames, 1] zeros (placeholder; Bridge has no actions
              and stage-1 LAM never reads them anyway — `_HarmonizedSourceDataset`
              overwrites this field at the datamodule level)
  attrs     : num_frames, instruction, episode_index, source_fps, image_format,
              target_height, target_width, jpeg_quality, source_video, split

Resumability is per-episode: if the final h5 file exists with a valid
`num_frames` attr and matching `frames` dataset shape, the worker skips it.
Atomic rename guarantees no half-written files survive a kill.
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
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


decord.bridge.set_bridge('native')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data-root',
                   default=os.path.join(_ditar_data_root(), 'data', 'bridge_multiview'),
                   help='root holding the exported videos/ + jsonl manifest')
    p.add_argument('--manifest', default='episode_instruction_multiview.jsonl')
    p.add_argument('--video-key', default='video_images1')
    p.add_argument('--out-root',
                   default=os.path.join(_ditar_data_root(), 'data', 'bridge_multiview', 'extracted_jpeg_h5'),
                   help='output HDF5 root (the training config reads this)')
    p.add_argument('--target-size', type=int, default=256)
    p.add_argument('--source-fps', type=float, default=10.0)
    p.add_argument('--jpeg-quality', type=int, default=85)
    p.add_argument('--split-ratio', type=float, default=0.98)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--state-file', default=None)
    p.add_argument('--failed-list', default=None)
    p.add_argument('--decode-batch', type=int, default=64)
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
            if 'frames' not in h or 'action' not in h:
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
    src = Path(job['src'])
    out = Path(job['out'])
    target_size = int(job['target_size'])
    quality = int(job['quality'])
    decode_batch = int(job['decode_batch'])
    episode_index = int(job['episode_index'])
    source_fps = float(job['source_fps'])
    rel_video = str(job['rel_video'])
    instruction = str(job['instruction'])
    split = str(job['split'])

    if is_complete_h5(out):
        return {'episode_index': episode_index, 'status': 'skipped', 'frames': 0}

    if not src.exists():
        return {'episode_index': episode_index, 'status': 'missing_source', 'frames': 0, 'error': str(src)}

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
        return {'episode_index': episode_index, 'status': 'open_failed', 'frames': 0, 'error': repr(error)}

    n_total = len(vr)
    if n_total <= 0:
        return {'episode_index': episode_index, 'status': 'empty_video', 'frames': 0}

    jpeg_records: List[bytes] = []
    try:
        for chunk_start in range(0, n_total, decode_batch):
            chunk_idx = list(range(chunk_start, min(chunk_start + decode_batch, n_total)))
            try:
                batch = vr.get_batch(chunk_idx).asnumpy()
            except Exception:
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
        return {'episode_index': episode_index, 'status': 'decode_failed', 'frames': 0, 'error': repr(error)}

    n_keep = len(jpeg_records)
    if n_keep != n_total:
        return {'episode_index': episode_index, 'status': 'frame_count_mismatch', 'frames': n_keep,
                'error': f'expected {n_total} got {n_keep}'}

    try:
        with h5py.File(tmp, 'w') as h:
            dt_jpeg = h5py.vlen_dtype(np.uint8)
            ds_frames = h.create_dataset('frames', (n_keep,), dtype=dt_jpeg, chunks=(1,))
            for i, b in enumerate(jpeg_records):
                ds_frames[i] = np.frombuffer(b, dtype=np.uint8)
            # Placeholder action (DROIDHDF5Dataset reads action[idx] for every
            # frame; stage-1 LAM never consumes it — the harmonized wrapper
            # at the datamodule overwrites this field anyway).
            h.create_dataset('action', data=np.zeros((n_keep, 1), dtype=np.float32))
            h.attrs['num_frames'] = n_keep
            h.attrs['instruction'] = instruction
            h.attrs['episode_index'] = episode_index
            h.attrs['source_fps'] = source_fps
            h.attrs['image_format'] = 'jpeg'
            h.attrs['target_height'] = target_size
            h.attrs['target_width'] = target_size
            h.attrs['jpeg_quality'] = quality
            h.attrs['source_video'] = rel_video
            h.attrs['split'] = split
        os.replace(tmp, out)
    except Exception as error:
        try:
            tmp.unlink()
        except Exception:
            pass
        return {'episode_index': episode_index, 'status': 'write_failed', 'frames': 0, 'error': repr(error)}

    return {'episode_index': episode_index, 'status': 'ok', 'frames': n_keep}


def build_jobs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], int, int]:
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = data_root / manifest_path
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'train').mkdir(exist_ok=True)
    (out_root / 'test').mkdir(exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with manifest_path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.video_key not in r:
                continue
            rows.append(r)
    rows.sort(key=lambda r: int(r['episode_index']))

    cut = max(1, min(len(rows) - 1, int(len(rows) * args.split_ratio)))
    train_rows = rows[:cut]
    test_rows = rows[cut:]

    jobs: List[Dict[str, Any]] = []
    skipped = 0
    seen = 0
    for split_name, split_rows in (('train', train_rows), ('test', test_rows)):
        for r in split_rows:
            seen += 1
            episode_index = int(r['episode_index'])
            out_path = out_root / split_name / f'episode_{episode_index:08d}.h5'
            if is_complete_h5(out_path):
                skipped += 1
                continue
            jobs.append(
                dict(
                    src=str(data_root / r[args.video_key]),
                    out=str(out_path),
                    target_size=args.target_size,
                    quality=args.jpeg_quality,
                    decode_batch=args.decode_batch,
                    episode_index=episode_index,
                    source_fps=args.source_fps,
                    rel_video=str(r[args.video_key]),
                    instruction=str(r.get('instruction') or ''),
                    split=split_name,
                )
            )
            if args.limit and len(jobs) >= args.limit:
                break
        if args.limit and len(jobs) >= args.limit:
            break
    return jobs, skipped, seen


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_file) if args.state_file else Path(args.out_root) / '.extract_state.json'
    failed_path = Path(args.failed_list) if args.failed_list else Path(args.out_root) / '.extract_failed.txt'

    print(f'[extract-bridge] data_root={args.data_root}')
    print(f'[extract-bridge] out_root={args.out_root}')
    print(f'[extract-bridge] target_size={args.target_size} quality={args.jpeg_quality} fps={args.source_fps}')
    print(f'[extract-bridge] split_ratio={args.split_ratio} num_workers={args.num_workers}')
    sys.stdout.flush()

    t0 = time.monotonic()
    jobs, skipped, seen = build_jobs(args)
    print(f'[extract-bridge] manifest scan: seen={seen} already_done={skipped} todo={len(jobs)} (in {time.monotonic()-t0:.1f}s)')
    sys.stdout.flush()
    if not jobs:
        print('[extract-bridge] nothing to do.')
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            'completed': skipped, 'failed': 0, 'frames': 0,
            'todo_initial': 0, 'elapsed_seconds': 0.0,
            'pid': os.getpid(), 'status': 'finished',
        }) + '\n')
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
                        f'[extract-bridge] done={completed}/{len(jobs)} failed={failed} '
                        f'frames={frame_total:,} elapsed={elapsed/60:.1f}min '
                        f'rate={rate:.2f} ep/s eta={eta/60:.1f}min',
                        flush=True,
                    )
                    last_log_at = time.monotonic()
                    state_path.write_text(json.dumps({
                        'completed': completed, 'failed': failed, 'frames': frame_total,
                        'todo_initial': len(jobs), 'elapsed_seconds': time.monotonic() - t0,
                        'pid': os.getpid(),
                    }) + '\n')
        except KeyboardInterrupt:
            print('[extract-bridge] caught SIGINT — terminating pool.')
            pool.terminate()
            pool.join()
            raise

    failed_fh.close()
    elapsed = time.monotonic() - t0
    print(
        f'[extract-bridge] FINISHED. completed={completed} failed={failed} frames_written={frame_total:,} '
        f'elapsed={elapsed/60:.2f} min', flush=True,
    )
    state_path.write_text(json.dumps({
        'completed': completed, 'failed': failed, 'frames': frame_total,
        'todo_initial': len(jobs), 'elapsed_seconds': elapsed,
        'pid': os.getpid(), 'status': 'finished',
    }) + '\n')


if __name__ == '__main__':
    main()
