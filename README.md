<h2 align="center"> <a href="https://arxiv.org/abs/2610.14159">DiTaR: Disentangling Task-Relevant Latent Dynamics for Reward Modeling</a></h2>

<p align="center">
  <a href="https://edisiondyli.github.io/">Dayou Li</a><sup>1,*</sup>,
  <a href="https://shanemankiw.github.io/">Jionghao Wang</a><sup>1,*</sup>,
  <a href="https://peihaowang.github.io/">Peihao Wang</a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=X52xke0AAAAJ">Jiuzhou Lei</a><sup>1</sup>,
  <a href="https://haohww.github.io/">Hao Wang</a><sup>1</sup>,
  Jichen Chai<sup>3</sup>,
  <a href="https://alexhu.top/">Hezhen Hu</a><sup>2</sup>,
  <a href="https://engineering.tamu.edu/mechanical/profiles/zheng-minghui.html">Minghui Zheng</a><sup>1</sup>,
  <a href="https://www.vita-group.space/pi">Zhangyang Wang</a><sup>2</sup>,
  <a href="https://people.tamu.edu/~xinli/">Xin Li</a><sup>1</sup>,
  <a href="https://engineering.tamu.edu/cse/profiles/Wang-Wenping.html">Wenping Wang</a><sup>1</sup>,
  <a href="https://zhiwenfan.github.io/">Zhiwen Fan</a><sup>1,†</sup>
  <br>
  <sup>1</sup>Texas A&amp;M University &nbsp; <sup>2</sup>The University of Texas at Austin &nbsp; <sup>3</sup>Georgia Institute of Technology
  <br>
  <sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author
</p>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2610.14159-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2610.14159)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://ditarr.github.io/)

</h5>

<p align="center">
  <a href="https://ditarr.github.io/">
    <img src="assets/teaser.gif" width="100%" alt="DiTaR in 30 seconds: dense rewards from video through task-relevant dynamics, recognizing task progress, and policy learning results">
  </a>
</p>

<div align="center">
This repository is the official implementation of DiTaR.
DiTaR turns action-free videos into dense rewards for robot reinforcement learning. It separates task-relevant dynamics from visual variation, so rewards stay stable under lighting and camera changes, yet respond when the execution is disrupted.
</div>
<br>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Updates](#updates)
- [Overview](#overview)
  - [Pipeline](#pipeline)
  - [How the Reward Works](#how-the-reward-works)
  - [Repository Structure](#repository-structure)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Training the Reward Model](#training-the-reward-model)
  - [Reward Calibration](#reward-calibration)
  - [Training RL](#training-rl)
  - [Smoke Test](#smoke-test)
  - [Configuration](#configuration)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Updates

**[2026-10-13]** The paper is available on arXiv: [2610.14159](https://arxiv.org/abs/2610.14159).

**[2026-07-13]** Released the full training code and configs for the reward model and the RL pipeline.

## Overview

DiTaR plugs a task-aware learned reward into online reinforcement learning: a WAN-TC latent-diffusion reward model drives a DrQ-v2 agent on image-based manipulation benchmarks, **ManiSkill** and **MetaWorld**. The repository has two halves that connect through a small interface:

- **`TSEncoder/`**: the reward *model* (the `ditar_encoder` package: latent encoder + flow-matching diffusion), with its complete training code.
- **`ours/` + `rl_engine/`**: the RL half (the reward wrapper and training scripts + a vendored DrQ-v2 engine).

`ours/reward_models` imports the reward model **lazily**, so the RL code builds and imports even before you have a trained reward model.

### Pipeline

```
datasets ──► Stage-1 encoder ──► Stage-2 encoder ──► latent stats ──► TC diffusion ──► reward
             (task-irrelevant)   (task-relevant)                       (= reward model)  calibration ──► RL
```

Every step is a real entry point in this repository; [`TRAINING.md`](TRAINING.md) walks through it end to end.

### How the Reward Works

`ours/reward_models/wan_tc_diffusion_reward.py` exposes a `calc_reward(imgs)` API compatible with the DrQ-v2 loop:

- **input** `imgs`: a tensor `[1, T, H, W, 3]` in `[-1, 1]` or `[0, 1]`
- **output**: a per-step reward tensor `[T]`

Internally it slides a window over the episode, computes a paired-Monte-Carlo flow-matching x0-residual `E_t` from the diffusion model, and maps it to one of four reward variants (then applies demo-based z-score normalization):

| `reward_type` | formula |
|---|---|
| `neg_e`       | `r_t = -E_t` |
| `exp`         | `r_t = exp(-E_t)`  ∈ (0, 1]  *(default)* |
| `td_residual` | `r_t = td_scale · (E_{t-k} - E_t)` |
| `td_exp`      | `r_t = td_scale · (exp(-E_t) - exp(-E_{t-k}))` |

The DrQ-v2 loop combines the learned reward with the env reward and an optional RND exploration bonus:
`r = learned_scale · r_learned + r_env + expl_scale · r_rnd`.

To use your own reward, register it in `make_rm(cfg)` (`ours/reward_models/__init__.py`) against the same `calc_reward(imgs) -> [T]` contract.

### Repository Structure

```
.
├── ours/                    # this work
│   ├── reward_models/       #   WAN-TC reward wrapper + make_rm registry + instructions
│   ├── scripts/             #   train (maniskill / shared) + reward-stats calibration
│   ├── configs/             #   Hydra configs (rl = ManiSkill, rl_shared = MetaWorld)
│   └── reward_stats/        #   precomputed (μ, σ) z-score stats per benchmark/variant
├── rl_engine/               # vendored DrQ-v2 engine + MetaWorld env factory + RND (MIT, see NOTICE)
├── maniskill_exp/           # ManiSkill env factory + task registries
├── TSEncoder/               # the reward MODEL — `ditar_encoder` (encoder + diffusion, full training code)
│   ├── configs/             #   stage1 / stage2 (warmup, alltrain) / diffusion training configs
│   ├── scripts/train.sh     #   the 4-stage training launcher
│   └── examples/smoke.sh    #   synthetic end-to-end training smoke test
├── setup.py · env.sh        # install, or just set PYTHONPATH
└── LICENSE · LICENSE-rl_engine · NOTICE · TRAINING.md · CHECKPOINTS.md · INSTALL.md
```

## Get Started

### Installation
1. **Download repo:**
   ```bash
   git clone https://github.com/phai-lab/DiTaR.git
   cd DiTaR
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n ditar python=3.11 -y
   conda activate ditar
   ```

3. **Install Python dependencies:**
   ```bash
   # RL engine + training loop
   pip install -r requirements.txt
   pip install -e .

   # reward model (TSEncoder/)
   pip install diffusers==0.37.1 transformers lightning einops rotary-embedding-torch sentencepiece safetensors huggingface_hub
   ```

4. **Install a simulator** (only the one(s) you use; we recommend a separate env per benchmark):
   - MetaWorld: [`metaworld`](https://github.com/Farama-Foundation/Metaworld) + MuJoCo
   - ManiSkill: [`mani_skill`](https://github.com/haosulab/ManiSkill) (>=3.0) + SAPIEN

   Follow each project's upstream guide for exact versions; [`INSTALL.md`](INSTALL.md) lists known pitfalls.

5. **Set up paths and check the install:**
   ```bash
   source env.sh   # puts ours/, rl_engine/, maniskill_exp/ and TSEncoder/ on PYTHONPATH
   python -c "import ours.reward_models, rl_engine.rl.drqv2.agent; print('ok')"
   ```

   The frozen WAN 2.2 VAE and T5 text encoder are downloaded from Hugging Face into `HF_HOME` on first use.

### Data Preparation
The reward model trains on a weighted mixture of robot-video datasets (AgiBot World, DROID, BridgeData V2, and MetaWorld / ManiSkill expert videos). Training works on any subset.

- Place the converted datasets under `$DITAR_DATA_ROOT` (expected layout and HDF5 schema: [`TRAINING.md`](TRAINING.md#0-datasets)).
- Converters from raw downloads to the training format: [`TSEncoder/scripts/data_prep/`](TSEncoder/scripts/data_prep/README.md).

### Training the Reward Model
All stages go through one launcher, `TSEncoder/scripts/train.sh`:

```bash
cd TSEncoder
export DITAR_DATA_ROOT=/path/to/datasets

# Stage 1: task-irrelevant encoder
scripts/train.sh stage1

# Stage 2: task-relevant encoder (warmup, then alltrain)
DITAR_STAGE1_CKPT=logs/stage1/last.ckpt scripts/train.sh stage2-warmup
DITAR_STAGE1_CKPT=logs/stage1/last.ckpt scripts/train.sh stage2-alltrain --init_ckpt logs/stage2_warmup/last.ckpt
export DITAR_STAGE2_CKPT=logs/stage2_alltrain/last.ckpt

# Latent normalization stats
python scripts/compute_diffusion_stats.py --config configs/diffusion_q4.yaml \
  --stage2-ckpt $DITAR_STAGE2_CKPT --out configs/stats/q4_stats.json --max-batches 200

# TC diffusion (this is the reward model)
scripts/train.sh diffusion
```

Then point the RL code at the trained artifacts:

```bash
export WAN_TC_CKPT=TSEncoder/logs/wan_tc_latent_diffusion_q4_.../checkpoints/<best>.ckpt
export WAN_TC_STAGE2_CKPT=$DITAR_STAGE2_CKPT
export WAN_TC_STATS=TSEncoder/configs/stats/q4_stats.json
```

See [`TRAINING.md`](TRAINING.md) for details on each stage.

### Reward Calibration
The RL reward is z-scored by demo statistics that are specific to the reward-model checkpoint. After training your own reward model, regenerate them:

```bash
python ours/scripts/calibrate_reward_stats.py \
  --benchmark maniskill --n_demos 128 --out ours/reward_stats/maniskill.json
# repeat for: metaworld
```

### Training RL

```bash
source env.sh
export WAN_TC_CKPT=/path/to/wan_tc.ckpt    # see CHECKPOINTS.md

# ManiSkill
export VK_VISIBLE_DEVICES=0 SAPIEN_VK_DEVICE_ID=0
python ours/scripts/train_drqv2_maniskill_ours.py \
  --config-path "$PWD/ours/configs/rl" --config-name default \
  task@_global_=PickCube-v1 seed=121 \
  env_reward_mode=sparse relax_success=true num_train_frames=3000000

# MetaWorld
export MUJOCO_GL=egl
python ours/scripts/train_drqv2_shared_ours.py \
  --config-path "$PWD/ours/configs/rl_shared" --config-name default \
  task@_global_=door-open-v3 seed=121 num_train_frames=3000000
```

Training logs, evaluation results and videos are written under `outputs/rl/` (override with `OUTPUT_ROOT`).

### Smoke Test
To check that the reward-model code runs end to end without any real data (builds a tiny synthetic dataset and runs every stage for a few steps):

```bash
cd TSEncoder && PYTHON_BIN=$(which python) bash examples/smoke.sh
```

A one-minute RL smoke run is in [`INSTALL.md`](INSTALL.md#4-end-to-end-smoke-test).

### Configuration
Paths are not hard-coded; configs read them via `${oc.env:...}`:

| var | meaning |
|---|---|
| `REPO_ROOT` / `OURS_ROOT` | repo root / `ours` dir (set by `env.sh`) |
| `REWARD_MODEL_ROOT` | override the bundled reward model (default: `TSEncoder/`) |
| `WAN_TC_CKPT` | WAN-TC diffusion checkpoint (required to run the reward) |
| `WAN_TC_STAGE2_CKPT` | frozen TC-teacher checkpoint (optional override) |
| `WAN_TC_STATS` | latent-stats JSON for the diffusion model |
| `WAN_VAE_PATH` / `T5_PATH` | VAE / T5 (default: HF repo ids, auto-downloaded) |
| `HF_HOME` | HuggingFace cache (default `~/.cache/huggingface`) |

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Diffusion Reward](https://github.com/TaoHuang13/diffusion_reward) and [DrQ-v2](https://github.com/facebookresearch/drqv2)
- [Wan 2.2](https://github.com/Wan-Video/Wan2.2) and [T5](https://huggingface.co/google-t5/t5-base)
- [Meta-World](https://github.com/Farama-Foundation/Metaworld) and [ManiSkill](https://github.com/haosulab/ManiSkill)
- The AgiBot World, DROID and BridgeData V2 datasets

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibtex
@misc{li2026disentanglingtaskrelevantlatentdynamics,
      title={Disentangling Task-Relevant Latent Dynamics for Reward Modeling}, 
      author={Dayou Li and Jionghao Wang and Peihao Wang and Jiuzhou Lei and Hao Wang and Jichen Chai and Hezhen Hu and Minghui Zheng and Zhangyang Wang and Xin Li and Wenping Wang and Zhiwen Fan},
      year={2026},
      eprint={2610.14159},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.14159}, 
}
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

`rl_engine/` (plus `maniskill_exp/`) is a vendored **MIT** subset of
[Diffusion Reward](https://github.com/TaoHuang13/diffusion_reward)
(Huang et al., 2023) and remains under its original MIT license, preserved
verbatim in [`LICENSE-rl_engine`](LICENSE-rl_engine).
