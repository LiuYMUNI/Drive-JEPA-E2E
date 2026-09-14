#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/media/dell/Storage/closed_loop_evaluation/navsim/.venv/bin/python}"
DATA_ROOT="${OPENSCENE_DATA_ROOT:-/media/dell/Storage/Drive-JEPA-data-4060-full}"
SOURCE_DATA_ROOT="${NAVSIM_SOURCE_DATA_ROOT:-}"
EXP_ROOT="${NAVSIM_EXP_ROOT:-/media/dell/Storage}"
CACHE_ROOT="${DRIVE_JEPA_CACHE_ROOT:-$EXP_ROOT/Drive-JEPA-cache}"
PDM_SUPERVISION="${DRIVE_JEPA_PDM_SUPERVISION:-1}"

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
export NAVSIM_DEVKIT_ROOT="$ROOT/navsim_v2"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/media/dell/Storage/closed_loop_evaluation/dataset/maps}"
export PYTHONPATH="$ROOT/navsim_v2:${PYTHONPATH:-}"

echo "python: $PYTHON"
"$PYTHON" - <<'PY'
import os, sys
import torch
print("python_version:", sys.version.split()[0])
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("OPENSCENE_DATA_ROOT:", os.environ["OPENSCENE_DATA_ROOT"])
print("NAVSIM_EXP_ROOT:", os.environ["NAVSIM_EXP_ROOT"])
PY

for p in \
  "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
  "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
  "$CACHE_ROOT/vitl_merge_3dataset_e50.pt"; do
  test -e "$p" && echo "OK: $p" || { echo "MISSING: $p" >&2; exit 2; }
done

if [[ "$PDM_SUPERVISION" == "1" ]]; then
  for p in "$CACHE_ROOT/train_metric_cache_v2_pruned" "$CACHE_ROOT/metric_cache_v2" "$CACHE_ROOT/anchors_scores_index_v2"; do
    test -e "$p" && echo "OK: $p" || { echo "MISSING: $p" >&2; exit 2; }
  done
else
  echo "PDM/anchor cache checks skipped (DRIVE_JEPA_PDM_SUPERVISION=0)"
fi

"$PYTHON" - <<'PY'
import os
import pickle
from pathlib import Path
import yaml

data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
log_root = data_root / "navsim_logs" / "trainval"
sensor_root = data_root / "sensor_blobs" / "trainval"
requested = os.environ.get("NAVSIM_PREFLIGHT_LOG")
logs = [log_root / f"{requested}.pkl"] if requested else sorted(log_root.glob("*.pkl"))
scene_filter = Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
tokens = set(yaml.safe_load(scene_filter.read_text())["tokens"])
selected_anchors = 0
selected_tokens = set()
available_tokens = set()
logs_used = 0
required_refs = 0
missing = []
for log_path in logs:
    if not log_path.is_file():
        continue
    rows = pickle.load(log_path.open("rb"))
    if not rows:
        continue
    available_tokens.update(
        rows[idx].get("token") for idx in range(3, len(rows) - 10)
        if rows[idx].get("token") is not None
    )
    log_selected = 0
    # SceneLoader starts 14-frame windows at each row and uses row 3 as the
    # center token. Check exactly the sensors requested by DriveJEPA's
    # SensorConfig for every selected navtrain anchor.
    for center_idx in range(3, len(rows) - 10):
        token = rows[center_idx].get("token")
        if token not in tokens:
            continue
        selected_tokens.add(token)
        selected_anchors += 1
        log_selected += 1
        required = [(center_idx - 1, "CAM_F0"), (center_idx, "CAM_F0")]
        required.extend((center_idx, camera) for camera in ("CAM_L0", "CAM_R0", "CAM_B0"))
        for frame_idx, camera in required:
            required_refs += 1
            path = sensor_root / rows[frame_idx]["cams"][camera]["data_path"]
            if not path.is_file():
                if len(missing) < 8:
                    missing.append(str(path))
    logs_used += int(log_selected > 0)
if selected_anchors == 0:
    raise SystemExit("No trainval log was found for sensor reference validation")
expected_tokens = tokens if requested is None else tokens & available_tokens
if selected_tokens != expected_tokens:
    print("NAVTRAIN_TOKEN_COVERAGE_MISMATCH:", len(expected_tokens - selected_tokens), "missing;", len(selected_tokens - expected_tokens), "unexpected")
    raise SystemExit(2)
if missing:
    print("MISSING_SENSOR_REFERENCES:")
    print("\n".join(missing[:8]))
    raise SystemExit(2)
print("sensor_reference_logs:", logs_used)
print("sensor_reference_anchors:", selected_anchors)
print("sensor_reference_files_checked:", required_refs)
print("OK: all required Drive-JEPA camera references")
PY

echo "NAVSIM v2 action-training preflight passed."
