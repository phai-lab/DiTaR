# Data preparation

The reward model trains on a weighted mixture of robot-video datasets. Every
manifest in `TSEncoder/configs/*.yaml` points at **JPEG-in-HDF5** episodes; the
scripts here turn raw dataset downloads into that format.

All paths are relative to `$DITAR_DATA_ROOT` (the scripts default to it):

```bash
export DITAR_DATA_ROOT=/path/to/datasets    # the base dir that contains data/
```

Dependencies (beyond the training deps):

```bash
pip install h5py decord imageio imageio-ffmpeg Pillow numpy tqdm huggingface_hub
```

> You do **not** need all five datasets. Train on any subset — delete the
> manifests you don't have from `data.manifests` and renormalize the remaining
> `sample_weight` values.

---

## The target format

Every `data_format: hdf5` manifest reads one HDF5 file per episode:

```
<episode_root>/{train,test}/episode_<i>.h5
    frames                 vlen uint8   # one JPEG/PNG bytestring per frame
    action                 (T, 7) float32
    state                  (T, 7) float32
    attrs["num_frames"]    int
    attrs["instruction"]   str          # drives the text conditioning
```

`TSEncoder/scripts/make_smoke_data.py` writes exactly this schema — use it as the
reference implementation and to validate your own converter.

`create_hdf5_dataset.py` is the **generic converter**: it takes a JSONL manifest
plus a tree of mp4s and emits the layout above. It serves DROID, MetaWorld **and**
ManiSkill — only the `--video_key` and `--target_size` differ. Output dirs are
named `hdf5_processed_<video_key>_<size>` by convention.

---

## Per-dataset chains

### MetaWorld-v3  ✅ complete

```bash
# 1. expert frame dirs (task/split/episode/*.png, rendered from the simulator)
#    -> mp4 + episode_instructions.jsonl
python export_metaworld_videos.py \
  --source-root $DITAR_DATA_ROOT/data/metaworldv3_expert_frames \
  --output-root $DITAR_DATA_ROOT/data/metaworldv3

# 2. -> HDF5
python create_hdf5_dataset.py \
  --data_root   $DITAR_DATA_ROOT/data/metaworldv3 \
  --jsonl_path  episode_instructions.jsonl \
  --output_dir  $DITAR_DATA_ROOT/data/metaworldv3/hdf5_processed_video_256 \
  --video_key   video --target_size 256 256 --frame_transform resize \
  --jpeg_quality 80 --split_ratio 0.98
```

The instruction map (`metaworld_v3_language_instructions.json`) ships at the repo
root. You supply the expert frames by rolling out MetaWorld-v3 expert policies.

### Bridge (BridgeData V2)  ✅ complete

Download the **raw** BridgeData V2 release (the variant that keeps per-camera
image folders — each trajectory dir must contain `lang.txt` and `images0/`,
`images1/`, `images2/`; the exporter skips trajectories missing any of them).

```bash
# 1. raw -> mp4 + episode_instruction_multiview.jsonl
python export_bridge_videos.py \
  --source-root $DITAR_DATA_ROOT/data/bridge/raw \
  --output-root $DITAR_DATA_ROOT/data/bridge_multiview

# 2. -> HDF5
python extract_bridge_jpeg_h5.py \
  --data-root $DITAR_DATA_ROOT/data/bridge_multiview \
  --out-root  $DITAR_DATA_ROOT/data/bridge_multiview/extracted_jpeg_h5
```

### AgiBot World (beta)  ✅ complete (download is manual)

The AgiBot repo is gated — get an `HF_TOKEN` and fetch `task_info/` plus the list
of observation tars from the dataset repo. Then:

```bash
export HF_TOKEN=...
# 1. stream the tars -> low-res three-view videos + manifest.jsonl
python prepare_agibot_lowres.py            # see --help for the tar list / task_info args

# 2. -> HDF5
python extract_agibot_jpeg_h5.py \
  --data-root $DITAR_DATA_ROOT/data/agibot_beta_threeview_320 \
  --out-root  $DITAR_DATA_ROOT/data/agibot_beta_threeview_320/extracted_jpeg_h5 \
  --video-key head_color.mp4
```

`prepare_agibot_lowres.py` is what writes `manifest.jsonl` (the extractor only
reads it). This manifest uses `data_format: agibot_hdf5`.

### DROID  ⚠️ you must build the manifest

The converter is here and works, but **the JSONL manifest builder is not part of
this release** — build it from your DROID (LeRobot) download, which ships the
episode→language-instruction metadata. One JSON object per line:

```json
{"episode_index": 1,
 "instruction": "Put the blue block in the green bowl",
 "video_exterior_1_left": "videos/chunk-000/observation.images.exterior_1_left/episode_000001.mp4",
 "video_exterior_2_left": "videos/chunk-000/observation.images.exterior_2_left/episode_000001.mp4"}
```

Video paths are **relative to `--data_root`**. Then:

```bash
python create_hdf5_dataset.py \
  --data_root   $DITAR_DATA_ROOT/data/droid \
  --jsonl_path  episode_instruction_exterior12.jsonl \
  --output_dir  $DITAR_DATA_ROOT/data/droid/hdf5_processed_orisize \
  --video_key   video_exterior_1_left \
  --target_size 224 224 --frame_transform preserve_original \
  --jpeg_quality 80 --split_ratio 0.98 --num_workers 32
```

### ManiSkill  ⚠️ you must render the videos and build the manifest

The expert videos come from replaying ManiSkill's own demonstrations (use
ManiSkill's official demo download + replay tooling to render per-episode camera
mp4s). Then build a JSONL — one object per line:

```json
{"episode_index": 1, "task_name": "LiftPegUpright-v1", "task_episode_index": 0,
 "instruction": "grasp the red-and-blue peg lying flat on the table, lift it, and stand it upright on one end",
 "video_left_camera": "<path>/0.mp4", "video_right_camera": "<path>/0.mp4"}
```

Instructions per task are in `maniskill_exp/task_registry_v2.json` at the repo
root. Then run the generic converter:

```bash
python create_hdf5_dataset.py \
  --data_root   $DITAR_DATA_ROOT/data/maniskill \
  --jsonl_path  episode_instruction_multiview_v2.jsonl \
  --output_dir  $DITAR_DATA_ROOT/data/maniskill/hdf5_processed_left_camera_256 \
  --video_key   video_left_camera \
  --target_size 256 256 --frame_transform resize \
  --jpeg_quality 80 --split_ratio 0.98 --split_group_key task_name
```

---

## Sanity check

Once a dataset is converted, point a manifest at it and run the synthetic smoke
test's later stages, or simply open an episode:

```python
import h5py
with h5py.File("<episode_root>/train/episode_00000000.h5") as f:
    print(list(f.keys()), dict(f.attrs))   # ['action','frames','state'] + num_frames/instruction
```
