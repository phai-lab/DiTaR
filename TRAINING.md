# Training from scratch

This repository releases the **complete training code and configs for both
halves** of the method: the task-aware reward model (`TSEncoder/`) and the RL
that consumes it (`ours/` + `rl_engine/`). You train the reward model on public
robot-video datasets, then train RL against it.

The pipeline, end to end:

```
datasets ──► Stage-1 encoder ──► Stage-2 encoder ──► latent stats ──► TC diffusion ──► reward
             (task-irrelevant)   (task-relevant,      (q4_stats)       (= the reward     calibration ──► RL
                                  warmup → alltrain)                     model)
```

Every step below is a real entry point in this repo. If you just want to check
the code runs without any data, jump to [Synthetic smoke test](#synthetic-smoke-test).

---

## 0. Datasets

The reward model trains on a **weighted mixture** of manifests declared in
`TSEncoder/configs/*.yaml` under `data.manifests`:

| manifest | dataset | default weight | source |
|---|---|---|---|
| `agibot` | AgiBot World (beta, three-view, 320px) | 0.38 | public release |
| `droid` | DROID | 0.38 | public release |
| `bridge` | BridgeData V2 (multi-view) | 0.19 | public release |
| `metaworldv3` | MetaWorld-v3 expert videos | 0.025 | rendered from the simulator |
| `maniskill` | ManiSkill expert videos | small | rendered from the simulator |

> You do **not** need all five. Training works on any subset — delete the
> manifests you don't have from `data.manifests` and renormalize the remaining
> `sample_weight` values. Start with one dataset (e.g. DROID) to get going.

### Expected on-disk layout

All manifest paths are relative to `$DITAR_DATA_ROOT` (no absolute paths live in
the configs):

```
$DITAR_DATA_ROOT/data/agibot_beta_threeview_320/extracted_jpeg_h5   (+ manifest.jsonl)
$DITAR_DATA_ROOT/data/droid/hdf5_processed_orisize
$DITAR_DATA_ROOT/data/bridge_multiview/extracted_jpeg_h5
$DITAR_DATA_ROOT/data/metaworldv3/hdf5_processed_video_256
$DITAR_DATA_ROOT/data/maniskill/hdf5_processed_left_camera_256
```

### HDF5 schema (`data_format: hdf5`)

Each manifest's `episode_root` holds one HDF5 file per episode, split into
`train/` and `test/`:

```
<episode_root>/{train,test}/episode_<i>.h5
    frames                    vlen uint8   # PNG-encoded RGB, one entry per frame
    action                    (T, 7) float32
    state                     (T, 7) float32
    attrs["num_frames"]       int
    attrs["instruction"]      str          # the task instruction (drives text conditioning)
```

**`TSEncoder/scripts/make_smoke_data.py` writes exactly this schema** — read it
as the reference implementation and use it to validate your own converter.

The AgiBot manifest uses `data_format: agibot_hdf5` instead, which reads a
`manifest.jsonl` plus a per-episode `h5_root` and a `video_key`
(default `head_color.mp4`).

Per-manifest knobs (`chunk_stride`, `stride_choices`, `stride_probs`,
`min_frame_count`) control temporal sub-sampling; the shipped values are the
ones used for the paper recipe.

### Converting raw downloads → training HDF5

**[`TSEncoder/scripts/data_prep/`](TSEncoder/scripts/data_prep/README.md)** holds
the converters, with the exact command chain per dataset:

| dataset | chain | status |
|---|---|---|
| MetaWorld-v3 | `export_metaworld_videos.py` → `create_hdf5_dataset.py` | ✅ complete |
| Bridge | `export_bridge_videos.py` → `extract_bridge_jpeg_h5.py` | ✅ complete |
| AgiBot | `prepare_agibot_lowres.py` → `extract_agibot_jpeg_h5.py` | ✅ complete (manual download) |
| DROID | *(build the JSONL manifest)* → `create_hdf5_dataset.py` | ⚠️ manifest builder is yours to write |
| ManiSkill | *(render demos + build JSONL)* → `create_hdf5_dataset.py` | ⚠️ video render + manifest are yours |

`create_hdf5_dataset.py` is a **generic** JSONL+mp4 → HDF5 converter shared by
DROID, MetaWorld and ManiSkill. For the two ⚠️ rows the data-prep README gives the
exact JSONL schema with a real example line, so the remaining step is a small
join against metadata your download already ships.

---

## 1. Environment

```bash
source env.sh                     # puts ours/, rl_engine/, TSEncoder/ on PYTHONPATH
pip install -e TSEncoder          # or add TSEncoder/ to PYTHONPATH (env.sh already does)
```

The reward model additionally needs `diffusers`, `transformers`, `lightning`,
`einops`, `rotary-embedding-torch`, `sentencepiece`, and a HuggingFace cache for
the **frozen** WAN 2.2 VAE and T5 text encoder (auto-downloaded):

```bash
export HF_HOME=~/.cache/huggingface        # WAN VAE + T5 land here
export DITAR_DATA_ROOT=/path/to/datasets   # root of the layout above
```

---

## 2. Train the reward model (`TSEncoder/`)

All four stages go through one launcher, `TSEncoder/scripts/train.sh`, which
wraps a LightningCLI `fit`. Pass extra `fit` args straight through.

### Stage 1 — task-irrelevant encoder (`z_irr`)

```bash
cd TSEncoder
DITAR_DATA_ROOT=$DITAR_DATA_ROOT scripts/train.sh stage1
# -> logs/stage1_.../checkpoints/last.ckpt
```

### Stage 2 — task-relevant encoder (`z_rel`), two phases

Stage-2 **requires** the Stage-1 checkpoint to construct (it seeds and freezes
the task-irrelevant branch):

```bash
# Phase A: warmup
DITAR_DATA_ROOT=$DITAR_DATA_ROOT \
DITAR_STAGE1_CKPT=logs/stage1/last.ckpt \
  scripts/train.sh stage2-warmup

# Phase B: alltrain, warm-started from Phase A's full weights
DITAR_DATA_ROOT=$DITAR_DATA_ROOT \
DITAR_STAGE1_CKPT=logs/stage1/last.ckpt \
  scripts/train.sh stage2-alltrain --init_ckpt logs/stage2_warmup/last.ckpt
```

The Phase-B checkpoint is the **diffusion teacher** — call it
`$DITAR_STAGE2_CKPT`.

> `stage_one_ckpt` (a *model* arg) and `--init_ckpt` (a *CLI* arg) are different
> things: the former is required to build Stage-2, the latter is an optional
> full-weights warm start. See the header of `scripts/train.sh`.

### Latent normalization stats

The diffusion normalizes its WAN-source / TC-target latents with per-channel
stats computed **from your data** with your Stage-2 teacher:

```bash
python scripts/compute_diffusion_stats.py \
    --config configs/diffusion_q4.yaml \
    --stage2-ckpt $DITAR_STAGE2_CKPT \
    --out configs/stats/q4_stats.json \
    --max-batches 200
```

This reuses the model's exact frozen-teacher extraction path, so the stats match
training. See `TSEncoder/configs/stats/README.md`.

### The TC diffusion — this *is* the reward model

```bash
DITAR_DATA_ROOT=$DITAR_DATA_ROOT \
DITAR_STAGE2_CKPT=$DITAR_STAGE2_CKPT \
  scripts/train.sh diffusion
# -> logs/wan_tc_latent_diffusion_q4_.../checkpoints/*.ckpt
```

The canonical recipe (`configs/diffusion_q4.yaml`) is the **text-joint** model:
`text_mode: in_joint_attn`, `source_mode: history_5frame_wan2`. Keep those
consistent — the RL wrapper builds the model from the checkpoint's own hparams,
so a checkpoint trained with a different `text_mode` will not load.

---

## 3. Point the RL at your reward model

The three artifacts you just produced:

```bash
export WAN_TC_CKPT=TSEncoder/logs/wan_tc_latent_diffusion_q4_.../checkpoints/<best>.ckpt
export WAN_TC_STAGE2_CKPT=$DITAR_STAGE2_CKPT
export WAN_TC_STATS=TSEncoder/configs/stats/q4_stats.json
```

## 4. Calibrate the reward statistics ⚠️ required

The RL reward is z-scored by demo statistics `(μ, σ)` that are **specific to your
reward-model checkpoint**. The values in `ours/reward_stats/*.json` correspond to
our own trained model — **regenerate them for yours**, or the reward will be
normalized against the wrong distribution:

```bash
python ours/scripts/calibrate_reward_stats.py \
  --benchmark maniskill --n_demos 128 --out ours/reward_stats/maniskill.json
# repeat for: metaworld
```

This needs expert demos under `<repo>/data/` (see the script's header for the
per-benchmark manifest paths).

## 5. Train RL

```bash
python ours/scripts/train_drqv2_maniskill_ours.py \
  --config-path "$PWD/ours/configs/rl" --config-name default \
  task@_global_=PickCube-v1 seed=121 \
  env_reward_mode=sparse relax_success=true num_train_frames=3000000
```

For MetaWorld, see the Quick start in [`README.md`](README.md).

---

## Synthetic smoke test

To prove the reward-model code runs end to end **without any real data**,
`TSEncoder/examples/smoke.sh` builds a tiny synthetic HDF5 fixture and runs every
stage (Stage-1 → Stage-2 warmup → Stage-2 alltrain → diffusion → encode) for a
couple of steps each:

```bash
cd TSEncoder && PYTHON_BIN=$(which python) bash examples/smoke.sh
```

It needs one GPU and the frozen WAN VAE + T5 in the HF cache. It is a *runs*
check, not an accuracy check.

For the RL half, see the smoke command in [`INSTALL.md`](INSTALL.md).
