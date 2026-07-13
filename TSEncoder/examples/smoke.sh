#!/usr/bin/env bash
# End-to-end SMOKE test for DiTaR / TSEncoder — tiny, fast, synthetic-only.
#
# Proves the released code RUNS end to end with no real data:
#   1. build a tiny synthetic DROID-style HDF5 fixture
#   2. generate smoke configs (single synthetic manifest, 2-step trainer)
#   3. train Stage-1 (task-irrelevant z_irr)
#   4. train Stage-2 warmup (task-relevant z_rel, init from Stage-1)
#   5. train Stage-2 alltrain (warm-started from warmup)
#   6. train the TC diffusion (teacher = Stage-2 alltrain) -> exercises z_rel + DiT
#   7. encode a clip -> z_rel (inference)
#
# Needs the frozen WAN-2.2 VAE + T5-base (HF cache). Set PYTHON_BIN and the HF
# cache env before running. Uses 1 GPU briefly. NOT for accuracy — only "runs".
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${PYTHON_BIN:-python}"
SMOKE_DIR="${SMOKE_DIR:-$PWD/smoke_run}"
DATA="$SMOKE_DIR/data"; CFG="$SMOKE_DIR/configs"; LOGS="$SMOKE_DIR/logs"
mkdir -p "$DATA" "$CFG" "$LOGS"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "== [1/7] build synthetic fixture =="
"$PY" "$ROOT/scripts/make_smoke_data.py" --out "$DATA" --train 4 --test 2 --frames 12

echo "== [2/7] generate smoke configs =="
"$PY" - "$ROOT" "$DATA" "$CFG" "$LOGS" <<'PY'
import os, sys, yaml
root, data, cfgdir, logs = sys.argv[1:5]
def gen(base, out, ckname):
    c = yaml.safe_load(open(os.path.join(root, "configs", base)))
    c["data"]["manifests"] = [dict(name="smoke", data_format="hdf5",
                                   data_root=data, episode_root=data, sample_weight=1.0,
                                   chunk_stride=1, eval_chunk_stride=1)]
    c["data"]["batch_size"] = 2
    c["data"]["val_batch_size"] = 2
    c["data"]["chunk_stride"] = 1
    c["data"]["num_workers"] = 0
    c["data"]["min_frame_count"] = c["data"]["num_frames"]
    # smoke uses identity latent stats (real stats are data-derived, not shipped)
    for k in ("target_stats_path", "source_stats_path"):
        if isinstance(c.get("model"), dict) and k in c["model"]:
            c["model"][k] = ""
    c["trainer"] = dict(
        max_steps=2, limit_val_batches=1, num_sanity_val_steps=0,
        accelerator="auto", devices=1, precision="32-true",
        log_every_n_steps=1, enable_progress_bar=False, logger=False,
        callbacks=[dict(class_path="lightning.pytorch.callbacks.ModelCheckpoint",
                        init_args=dict(dirpath=os.path.join(logs, ckname),
                                       save_last=True, every_n_train_steps=1, save_top_k=0))],
    )
    yaml.safe_dump(c, open(os.path.join(cfgdir, out), "w"), sort_keys=False)
    print("  wrote", out)
gen("stage1_task_irrelevant.yaml", "smoke_stage1.yaml", "stage1")
gen("stage2_task_relevant_warmup.yaml", "smoke_stage2_warmup.yaml", "stage2_warmup")
gen("stage2_task_relevant_alltrain.yaml", "smoke_stage2_alltrain.yaml", "stage2_alltrain")
gen("diffusion_q4.yaml", "smoke_diffusion.yaml", "diffusion")
PY

S1CK="$LOGS/stage1/last.ckpt"
S2WARMCK="$LOGS/stage2_warmup/last.ckpt"
S2CK="$LOGS/stage2_alltrain/last.ckpt"

echo "== [3/7] train Stage-1 (task-irrelevant) =="
CONFIG="$CFG/smoke_stage1.yaml" bash "$ROOT/scripts/train.sh" stage1 \
  --trainer.default_root_dir "$(mktemp -d)"
[ -f "$S1CK" ] || { echo "FAIL: no Stage-1 checkpoint at $S1CK"; exit 1; }

echo "== [4/7] train Stage-2 warmup (task-relevant, frozen branch from Stage-1) =="
CONFIG="$CFG/smoke_stage2_warmup.yaml" bash "$ROOT/scripts/train.sh" stage2-warmup \
  --model.stage_one_ckpt "$S1CK" --trainer.default_root_dir "$(mktemp -d)"
[ -f "$S2WARMCK" ] || { echo "FAIL: no Stage-2 warmup checkpoint at $S2WARMCK"; exit 1; }

echo "== [5/7] train Stage-2 alltrain (full model, warm-start from warmup) =="
CONFIG="$CFG/smoke_stage2_alltrain.yaml" bash "$ROOT/scripts/train.sh" stage2-alltrain \
  --model.stage_one_ckpt "$S1CK" --init_ckpt "$S2WARMCK" \
  --trainer.default_root_dir "$(mktemp -d)"
[ -f "$S2CK" ] || { echo "FAIL: no Stage-2 alltrain checkpoint at $S2CK"; exit 1; }

echo "== [6/7] train TC diffusion (teacher = Stage-2 alltrain) =="
CONFIG="$CFG/smoke_diffusion.yaml" DITAR_STAGE2_CKPT="$S2CK" \
  bash "$ROOT/scripts/train.sh" diffusion --model.stage2_checkpoint "$S2CK" \
  --trainer.default_root_dir "$(mktemp -d)"

echo "== [7/7] encode smoke: clip -> z_rel =="
"$PY" "$ROOT/scripts/encode.py" --stage2-ckpt "$S2CK" --batch 2

echo "== SMOKE OK =="
