# Reward model: artifacts & wiring

This repo ships the **full training code and configs** for the reward model. You
train it on public robot-video datasets (see [`TRAINING.md`](TRAINING.md)), which
produces the artifacts below; the RL side then reads them from env vars.

## The `ditar_encoder` package

`ours/reward_models/wan_tc_diffusion_reward.py` imports
`WanTcLatentDiffusionLAM` from
`ditar_encoder.diffusion.wan_tc_latent_diffusion_model`. The package lives under
`TSEncoder/` and is put on the path automatically (`env.sh` adds it; the wrapper
also defaults `REWARD_MODEL_ROOT` to `TSEncoder/`). To point at a different
checkout:

```bash
export REWARD_MODEL_ROOT=/path/to/TSEncoder     # dir that contains ditar_encoder/
```

## Artifacts and where they come from

| env var | artifact | produced by |
|---|---|---|
| `WAN_TC_CKPT`        | the WAN-TC latent-diffusion checkpoint (**the reward model**) | `scripts/train.sh diffusion` |
| `WAN_TC_STAGE2_CKPT` | the frozen TC-teacher (Stage-2 encoder)                       | `scripts/train.sh stage2-alltrain` |
| `WAN_TC_STATS`       | latent normalization stats (`q4_stats.json`)                  | `scripts/compute_diffusion_stats.py` |
| `WAN_VAE_PATH`       | WAN 2.2 VAE — default `Wan-AI/Wan2.2-TI2V-5B-Diffusers`        | HuggingFace (auto-download, frozen) |
| `T5_PATH`            | T5 text encoder — default `t5-base`                           | HuggingFace (auto-download, frozen) |
| `HF_HOME`            | HuggingFace cache dir (default `~/.cache/huggingface`)        | — |

`load_from_checkpoint` accepts overrides for the VAE / T5 / stats paths, so any
absolute paths baked into a checkpoint's hparams are repointed at load time —
set the env vars above and you never need to edit a checkpoint.

Nothing here belongs in git: keep checkpoints out of the repo (`.gitignore`
already excludes `*.ckpt` / `*.pt` / `*.safetensors`).

## Keep the architecture consistent

The canonical recipe (`TSEncoder/configs/diffusion_q4.yaml`) is the **text-joint**
model: `text_mode: in_joint_attn` (the diffusion core's `type_embed` has 3 rows —
source / target / text) with `source_mode: history_5frame_wan2`.

The RL wrapper rebuilds the model from the checkpoint's own hparams, so
`WAN_TC_CKPT` must come from a run with the same architecture — otherwise
`load_from_checkpoint` raises a state-dict size mismatch (e.g. `core.type_embed`
`[2,512]` vs `[3,512]` if the checkpoint was trained without joint text
attention).

## Reward statistics (RL side)

`ours/reward_stats/*.json` holds the demo z-score `(μ, σ)` the RL reward is
normalized by. These are **specific to a given reward-model checkpoint** — after
training your own, regenerate them:

```bash
python ours/scripts/calibrate_reward_stats.py \
  --benchmark maniskill --n_demos 128 --out ours/reward_stats/maniskill.json
```

If the stats file is missing the wrapper falls back to identity normalization
(with a warning) rather than silently using the wrong scale.

## Verifying the wiring

With `WAN_TC_CKPT` set, this loads the reward model (WAN VAE + T5 + your
checkpoint) and returns a per-step reward:

```python
from types import SimpleNamespace
import torch
from ours.reward_models import make_rm

cfg = SimpleNamespace(rm_model="wan_tc_diffusion", ckpt_path="<WAN_TC_CKPT>",
                      reward_type="exp", mc_samples=2, default_instruction="open the door")
rm = make_rm(cfg).cuda().eval()
print(rm.calc_reward(torch.rand(1, 30, 64, 64, 3).cuda()).shape)   # -> torch.Size([30])
```
