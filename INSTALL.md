# Installation

The repo splits into three concerns: (1) the **RL engine + training code**, (2)
one or more **simulators**, and (3) the **reward model** (bundled under
`TSEncoder/` — see [`CHECKPOINTS.md`](CHECKPOINTS.md)).

## 1. This repo

```bash
# option A: editable install (installs ours + rl_engine + maniskill_exp)
pip install -e .

# option B: no install, just set PYTHONPATH
source env.sh
```

Then `pip install -r requirements.txt` for the core RL dependencies
(torch, hydra-core, omegaconf, opencv-python, imageio, dm-env, gym, gymnasium,
wandb, tensorboard).

Sanity check (does **not** need a simulator or the reward model):

```bash
python -c "import ours.reward_models, rl_engine.rl.drqv2.agent, maniskill_exp; print('ok')"
```

## 2. Simulators

Each benchmark needs its own simulator stack. These are platform-specific and
not pip-trivial; install only the one(s) you use. We recommend a **separate
conda env per benchmark**.

| benchmark | needs | notes |
|---|---|---|
| MetaWorld  | `metaworld` + MuJoCo            | headless rendering via `MUJOCO_GL=egl` |
| ManiSkill  | `mani_skill` (>=3.0) + `sapien` | uses Vulkan/EGL rendering |

Follow each project's upstream install guide for the exact versions and system
libraries (MuJoCo, Vulkan, EGL). The env factories are dispatched by task name:
`rl_engine.envs.make_env` (MetaWorld) and
`maniskill_exp.rl.envs.make_env` (ManiSkill).

> Adroit and LIBERO env modules from the upstream Diffusion Reward project are
> intentionally **not** bundled in this release.

### Known pitfalls

- **Use the benchmark's own env.** Don't run a MetaWorld task in the ManiSkill
  env (or vice versa) — the simulator stacks pin conflicting versions.
- **metaworld ↔ gymnasium version clash.** If `import metaworld` raises
  `AttributeError: module 'gymnasium.vector' has no attribute 'AutoresetMode'`,
  your `metaworld` is newer than your `gymnasium`. Pin the two together in the
  MetaWorld env.
- **ManiSkill rendering** needs Vulkan: export `VK_VISIBLE_DEVICES` and
  `SAPIEN_VK_DEVICE_ID` (set both to your GPU index).
- **`num_seed_frames` must span at least one full episode.** DrQ-v2 only flushes
  an episode to the replay buffer when it *ends*; if training starts before any
  episode is stored you get
  `IndexError: Cannot choose from an empty sequence` in `_sample_episode`. The
  shipped configs (`num_seed_frames: 4000`) are safe — only aggressive debug
  overrides can trip this.

## 3. Reward model

The reward model (`ditar_encoder`) lives under `TSEncoder/`; `env.sh` puts it on
`PYTHONPATH` (or `pip install -e TSEncoder`). Install its deps: `diffusers`,
`transformers`, `lightning`, `einops`, `rotary-embedding-torch`, `sentencepiece`.
The frozen WAN 2.2 VAE and T5 are auto-downloaded into `HF_HOME`.

**Train it with [`TRAINING.md`](TRAINING.md)** — that walks the datasets, the
four training stages, and the latent stats. Then set `WAN_TC_CKPT` /
`WAN_TC_STAGE2_CKPT` / `WAN_TC_STATS` ([`CHECKPOINTS.md`](CHECKPOINTS.md)).

Until then you can still import and develop against the RL half, and plug in your
own reward via `ours/reward_models/make_rm`.

## 4. End-to-end smoke test

Once a simulator env and `WAN_TC_CKPT` are in place, this short run exercises the
whole loop (env → agent → learned reward → RND → replay → gradient update → eval)
in about a minute on one GPU:

```bash
source env.sh
export WAN_TC_CKPT=/path/to/wan_tc.ckpt          # + WAN_TC_STAGE2_CKPT / WAN_TC_STATS
export VK_VISIBLE_DEVICES=0 SAPIEN_VK_DEVICE_ID=0

python ours/scripts/train_drqv2_maniskill_ours.py \
  --config-path "$PWD/ours/configs/rl" --config-name default \
  task@_global_=PickCube-v1 seed=121 \
  episode_length=50 num_seed_frames=110 num_train_frames=260 \
  eval_every_frames=120 num_eval_episodes=1 batch_size=16 \
  save_snapshot=false save_video=false use_wandb=false use_tb=false \
  reward.mc_samples=1
```

You should see the reward model load, the task instruction auto-resolve, and
alternating `| train |` / `| eval |` rows.

## 5. Reward calibration ⚠️ required after training your own reward model

`ours/reward_stats/*.json` holds the demo z-score `(μ, σ)` the RL reward is
normalized by. These values are **specific to the reward-model checkpoint they
were computed from** — once you train your own reward model
([`TRAINING.md`](TRAINING.md)), regenerate them, or the reward gets normalized
against the wrong distribution:

```bash
python ours/scripts/calibrate_reward_stats.py \
  --benchmark metaworld --n_demos 128 --out ours/reward_stats/metaworld.json
# repeat for: maniskill
```

This needs the trained reward model **and** expert demos under `<repo>/data/`
(see the script's header for the per-benchmark manifest paths). If a stats file
is missing, the wrapper falls back to identity normalization with a warning
rather than silently using the wrong scale.
