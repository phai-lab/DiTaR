"""Build a tiny synthetic DROID-style HDF5 fixture for smoke tests.

This is purely synthetic random data — NO real dataset is read or migrated.
The on-disk schema matches what ``DROIDHDF5Dataset`` (data_format: hdf5) reads:

    <out>/<split>/episode_<i>.h5
        frames : variable-length uint8, one PNG-encoded RGB frame per entry
        action : (num_frames, 7) float32   (zeros; unused by the encoder)
        state  : (num_frames, 7) float32   (zeros; unused by the encoder)
        attrs["num_frames"]  : int
        attrs["instruction"] : str         (mapped to batch["task_instruction"])

Usage:
    python scripts/make_smoke_data.py --out /path/to/fixture --frames 12
"""
from __future__ import annotations

import argparse
import io
import os

import h5py
import numpy as np
from PIL import Image

_INSTRUCTIONS = [
    "open the drawer",
    "pick up the cube",
    "close the box",
    "push the button",
    "turn the faucet",
    "press the handle",
]


def _encode_png(array: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def _write_episode(path: str, n_frames: int, h: int, w: int, seed: int, instruction: str) -> None:
    rng = np.random.default_rng(seed)
    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "w") as handle:
        frames = handle.create_dataset("frames", shape=(n_frames,), dtype=vlen_uint8)
        for idx in range(n_frames):
            frames[idx] = _encode_png(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))
        handle.create_dataset("action", data=np.zeros((n_frames, 7), dtype=np.float32))
        handle.create_dataset("state", data=np.zeros((n_frames, 7), dtype=np.float32))
        handle.attrs["num_frames"] = n_frames
        handle.attrs["instruction"] = instruction


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="fixture root (gets train/ and test/ subdirs)")
    ap.add_argument("--train", type=int, default=4, help="number of train episodes")
    ap.add_argument("--test", type=int, default=2, help="number of test episodes")
    ap.add_argument("--frames", type=int, default=12, help="frames per episode (>= 9 for diffusion history mode)")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()

    for split, count in (("train", args.train), ("test", args.test)):
        split_dir = os.path.join(args.out, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(count):
            _write_episode(
                os.path.join(split_dir, f"episode_{i:08d}.h5"),
                n_frames=args.frames,
                h=args.height,
                w=args.width,
                seed=(abs(hash((split, i))) % (2**31)),
                instruction=_INSTRUCTIONS[i % len(_INSTRUCTIONS)],
            )
    print(
        f"[make_smoke_data] wrote fixture at {args.out}: "
        f"train={args.train} test={args.test} frames={args.frames} {args.height}x{args.width}"
    )


if __name__ == "__main__":
    main()
