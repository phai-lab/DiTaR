#!/usr/bin/env bash
# Reusable training launcher for DiTaR / TSEncoder.
#
# Usage:
#   scripts/train.sh <stage> [extra LightningCLI `fit` args...]
#     stage = stage1 | stage2-warmup | stage2-alltrain | diffusion
#
# Path standard (see README): no absolute paths live in configs. Provide via env:
#   DITAR_DATA_ROOT   base dir for ${DITAR_DATA_ROOT}/... dataset roots
#   DITAR_STAGE1_CKPT trained Stage-1 ckpt — REQUIRED to construct Stage-2 (configs read
#                     ${DITAR_STAGE1_CKPT}; equivalently pass --model.stage_one_ckpt <ckpt>)
#   DITAR_STAGE2_CKPT trained Stage-2 encoder ckpt (diffusion teacher)
#   PYTHON_BIN        python to use (default: python)
#   CONFIG            override the default config path for the chosen stage
#   HF cache          standard HF_HOME / HF_HUB_CACHE for frozen WAN-VAE + T5
#
# Stage-2 checkpoint contract:
#   * stage_one_ckpt (a MODEL arg) is the Stage-1 checkpoint; it is consumed in __init__ to
#     seed/freeze the task-irrelevant branch and is REQUIRED for both warmup and alltrain.
#   * --init_ckpt (a CLI arg) is an OPTIONAL post-construction full-weights warm-start; use it
#     for Phase B (alltrain) to continue from the Phase-A (warmup) checkpoint. It does NOT
#     satisfy stage_one_ckpt.
#
# Examples:
#   DITAR_DATA_ROOT=/data scripts/train.sh stage1
#   DITAR_DATA_ROOT=/data DITAR_STAGE1_CKPT=logs/stage1/last.ckpt scripts/train.sh stage2-warmup
#   DITAR_DATA_ROOT=/data DITAR_STAGE1_CKPT=logs/stage1/last.ckpt scripts/train.sh stage2-alltrain \
#       --init_ckpt logs/stage2_warmup/last.ckpt
#   DITAR_DATA_ROOT=/data DITAR_STAGE2_CKPT=logs/stage2/last.ckpt scripts/train.sh diffusion
set -euo pipefail

STAGE="${1:?usage: train.sh <stage1|stage2-warmup|stage2-alltrain|diffusion> [fit args...]}"
shift || true

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${PYTHON_BIN:-python}"

case "$STAGE" in
  stage1)          MOD=ditar_encoder.cli.main_stage1;    DEF="$ROOT/configs/stage1_task_irrelevant.yaml";;
  stage2-warmup)   MOD=ditar_encoder.cli.main_stage2;    DEF="$ROOT/configs/stage2_task_relevant_warmup.yaml";;
  stage2-alltrain) MOD=ditar_encoder.cli.main_stage2;    DEF="$ROOT/configs/stage2_task_relevant_alltrain.yaml";;
  diffusion)       MOD=ditar_encoder.cli.main_diffusion; DEF="$ROOT/configs/diffusion_q4.yaml";;
  *) echo "[train.sh] unknown stage: $STAGE" >&2; exit 2;;
esac

CFG="${CONFIG:-$DEF}"
echo "[train.sh] stage=$STAGE module=$MOD config=$CFG python=$PY"
exec "$PY" -m "$MOD" fit --config "$CFG" "$@"
