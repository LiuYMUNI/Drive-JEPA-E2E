#!/usr/bin/env bash
set -Eeuo pipefail

# Unattended continuation for the official NAVSIM v2 Drive-JEPA action run.
# The downloader writes one marker after each verified sensor shard; this
# script waits for all eight, validates the data view, runs a one-batch smoke
# test, then starts the uncached full navtrain action training.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/media/dell/Storage/closed_loop_evaluation/navsim/.venv/bin/python}"
DATA_ROOT="${OPENSCENE_DATA_ROOT:-/media/dell/Storage/Drive-JEPA-data-4060-full}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/media/dell/Storage}"
CACHE_ROOT="${DRIVE_JEPA_CACHE_ROOT:-$EXP_ROOT/Drive-JEPA-cache}"
OUTPUT_ROOT="${NAVSIM_OUTPUT_ROOT:-/media/dell/Storage/Drive-JEPA-runs/full-navtrain-4060-bs32}"
DOWNLOAD_ROOT="${NAVSIM_DOWNLOAD_ROOT:-/media/dell/Storage/Drive-JEPA-navsim2}"
LOG_ROOT="${NAVSIM_RESUME_LOG_ROOT:-/media/dell/Storage/Drive-JEPA-runs/full-navtrain-4060-bs32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

export NAVSIM_DEVKIT_ROOT="$ROOT/navsim_v2"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export DRIVE_JEPA_CACHE_ROOT="$CACHE_ROOT"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/media/dell/Storage/closed_loop_evaluation/dataset/maps}"
export PYTHONPATH="$ROOT/navsim_v2:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DRIVE_JEPA_FORCE_PYTORCH_DEFORMABLE="${DRIVE_JEPA_FORCE_PYTORCH_DEFORMABLE:-1}"
export DRIVE_JEPA_FREEZE_VJEPA="${DRIVE_JEPA_FREEZE_VJEPA:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export BATCH_SIZE ACCUMULATE_GRAD_BATCHES

mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/resume.lock"
flock -n 9 || { echo "resume job already running"; exit 0; }
exec > >(tee -a "$LOG_ROOT/resume.log") 2>&1

markers=(
  navtrain_current_1.done navtrain_current_2.done
  navtrain_current_3.done navtrain_current_4.done
  navtrain_history_1.done navtrain_history_2.done
  navtrain_history_3.done navtrain_history_4.done
)
echo "[$(date '+%F %T')] waiting for verified NAVSIM navtrain shards under $DOWNLOAD_ROOT"
GLOBAL_BATCH=$((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))
echo "[$(date '+%F %T')] batch_size=$BATCH_SIZE accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES (global batch $GLOBAL_BATCH)"
while :; do
  missing=0
  for marker in "${markers[@]}"; do
    [[ -f "$DOWNLOAD_ROOT/.markers/$marker" ]] || missing=$((missing + 1))
  done
  [[ "$missing" == 0 ]] && break
  echo "[$(date '+%F %T')] $missing shard marker(s) remain; retrying in 60s"
  sleep 60
done

test -x "$PYTHON"
test -d "$DATA_ROOT/sensor_blobs/trainval"
test -d "$DATA_ROOT/navsim_logs/trainval"
test -d "$CACHE_ROOT/train_metric_cache_v2_pruned"
test -d "$CACHE_ROOT/anchors_scores_index_v2"
test -f "$CACHE_ROOT/vitl_merge_3dataset_e50.pt"

cd "$NAVSIM_DEVKIT_ROOT"
echo "[$(date '+%F %T')] running full-data preflight"
"$ROOT/scripts/preflight_navsim_v2_4060.sh"

echo "[$(date '+%F %T')] running one-batch streaming action smoke test"
MAX_EPOCHS=1 LIMIT_TRAIN_BATCHES=1 LIMIT_VAL_BATCHES=1 \
  NAVSIM_OUTPUT_ROOT="$OUTPUT_ROOT/smoke" \
  SKIP_DATA_PREFLIGHT=1 \
  bash "$ROOT/scripts/train_navsim_v2_4060.sh" \
    cache_path=null force_cache_computation=false \
    trainer.params.num_sanity_val_steps=0 +trainer.params.log_every_n_steps=1

echo "[$(date '+%F %T')] starting full NAVSIM v2 Drive-JEPA action training"
MAX_EPOCHS="${MAX_EPOCHS:-20}" \
  NAVSIM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  SKIP_DATA_PREFLIGHT=1 \
  bash "$ROOT/scripts/train_navsim_v2_4060.sh" \
    cache_path=null force_cache_computation=false \
    trainer.params.num_sanity_val_steps=0 +trainer.params.log_every_n_steps=50
