#!/usr/bin/env bash
# Source this to make the packages importable without installing:
#     source env.sh
# It also exports the path env-vars the configs read.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

export REPO_ROOT="$HERE"
export OURS_ROOT="$HERE/ours"
# All of our packages (ours, rl_engine, maniskill_exp) live at
# REPO_ROOT; TSEncoder/ holds the reward model (the ditar_encoder package).
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/TSEncoder:${PYTHONPATH:-}"

# The reward model (ditar_encoder) is picked up from TSEncoder/ automatically.
# To use a different checkout, set REWARD_MODEL_ROOT. See CHECKPOINTS.md.
# export REWARD_MODEL_ROOT=/path/to/TSEncoder
# export WAN_TC_CKPT=/path/to/wan_tc_diffusion.ckpt
# export WAN_TC_STAGE2_CKPT=/path/to/stage2_teacher.ckpt
# export WAN_TC_STATS=/path/to/latent_stats.json

echo "[env.sh] REPO_ROOT=$REPO_ROOT"
echo "[env.sh] PYTHONPATH set (ours + rl_engine + maniskill_exp + TSEncoder)."
