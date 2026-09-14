#!/usr/bin/env bash
set -euo pipefail

# Single RTX 4060 action-training entrypoint. This intentionally runs the
# Drive-JEPA NAVSIM planner training path, not the V-JEPA video pretraining.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/media/dell/Storage/closed_loop_evaluation/navsim/.venv/bin/python}"
DATA_ROOT="${OPENSCENE_DATA_ROOT:-/media/dell/Storage/Drive-JEPA-data-4060-full}"
SOURCE_DATA_ROOT="${NAVSIM_SOURCE_DATA_ROOT:-}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/media/dell/Storage}"
CACHE_ROOT="${DRIVE_JEPA_CACHE_ROOT:-$EXP_ROOT/Drive-JEPA-cache}"
OUT_ROOT="${NAVSIM_OUTPUT_ROOT:-/media/dell/Storage/Drive-JEPA-runs}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-16}"
PDM_SUPERVISION="${DRIVE_JEPA_PDM_SUPERVISION:-1}"

export NAVSIM_DEVKIT_ROOT="$ROOT/navsim_v2"
mkdir -p "$DATA_ROOT"
if [[ -n "$SOURCE_DATA_ROOT" && ! -e "$DATA_ROOT/sensor_blobs" ]]; then
  ln -s "$SOURCE_DATA_ROOT/sensor_blobs" "$DATA_ROOT/sensor_blobs"
fi
if [[ -n "$SOURCE_DATA_ROOT" && ! -e "$DATA_ROOT/navsim_logs" ]]; then
  ln -s "$SOURCE_DATA_ROOT/navsim_logs" "$DATA_ROOT/navsim_logs"
fi
if [[ -n "$SOURCE_DATA_ROOT" && ! -e "$DATA_ROOT/synthetic_scenes" && -e "$SOURCE_DATA_ROOT/synthetic_scenes" ]]; then
  ln -s "$SOURCE_DATA_ROOT/synthetic_scenes" "$DATA_ROOT/synthetic_scenes"
fi
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export DRIVE_JEPA_CACHE_ROOT="$CACHE_ROOT"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/media/dell/Storage/closed_loop_evaluation/dataset/maps}"
export PYTHONPATH="$ROOT/navsim_v2:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# The bundled mmcv wheel targets sm_86; RTX 4060 is sm_89. Use its correct
# differentiable PyTorch fallback unless an sm_89 extension is supplied.
export DRIVE_JEPA_FORCE_PYTORCH_DEFORMABLE="${DRIVE_JEPA_FORCE_PYTORCH_DEFORMABLE:-1}"
# 8 GB profile: keep the 1.2B-parameter V-JEPA representation frozen.
export DRIVE_JEPA_FREEZE_VJEPA="${DRIVE_JEPA_FREEZE_VJEPA:-1}"

test -x "$PYTHON" || { echo "Python not found: $PYTHON" >&2; exit 2; }
test -d "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" || { echo "Missing NAVSIM v2 sensor data: $OPENSCENE_DATA_ROOT/sensor_blobs/trainval" >&2; exit 2; }
test -d "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" || { echo "Missing NAVSIM v2 logs: $OPENSCENE_DATA_ROOT/navsim_logs/trainval" >&2; exit 2; }
if [[ "$PDM_SUPERVISION" == "1" ]]; then
  test -d "$CACHE_ROOT/train_metric_cache_v2_pruned" || { echo "Missing train metric cache under $CACHE_ROOT" >&2; exit 2; }
  test -d "$CACHE_ROOT/anchors_scores_index_v2" || { echo "Missing anchor score index under $CACHE_ROOT" >&2; exit 2; }
fi
test -f "$CACHE_ROOT/vitl_merge_3dataset_e50.pt" || { echo "Missing Drive-JEPA encoder weights under $CACHE_ROOT" >&2; exit 2; }

if [[ "${SKIP_DATA_PREFLIGHT:-0}" != "1" ]]; then
  "$ROOT/scripts/preflight_navsim_v2_4060.sh"
fi

mkdir -p "$OUT_ROOT"
cd "$NAVSIM_DEVKIT_ROOT"

# Stream raw NAVSIM samples by default. To use precomputed features, pass an
# explicit cache_path together with use_cache_without_dataset=true.
exec "$PYTHON" -u navsim/planning/script/run_training.py \
  agent=drive_jepa_perception_based_agent \
  train_test_split=navtrain \
  experiment_name=drive_jepa_navsim_v2_4060 \
  +agent.config.pdm_supervision="$PDM_SUPERVISION" \
  output_dir="$OUT_ROOT/drive_jepa_navsim_v2_4060" \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers=2 \
  dataloader.params.prefetch_factor=2 \
  trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
  trainer.params.max_epochs="${MAX_EPOCHS:-20}" \
  trainer.params.accelerator=gpu \
  +trainer.params.devices=1 \
  trainer.params.strategy=auto \
  trainer.params.precision=16-mixed \
  trainer.params.gradient_clip_val=1.0 \
  trainer.params.limit_train_batches="${LIMIT_TRAIN_BATCHES:-1.0}" \
  trainer.params.limit_val_batches="${LIMIT_VAL_BATCHES:-1.0}" \
  cache_path=null \
  use_cache_without_dataset=false \
  force_cache_computation=false \
  "$@"
