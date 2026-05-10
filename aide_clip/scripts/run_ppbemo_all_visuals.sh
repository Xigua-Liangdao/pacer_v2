#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/scripts/run_ppbemo_aide_best.sh"
PY="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
DATA_ROOT="${PPBEMO_ROOT:-$ROOT/data/bbpemo}"
ANNOTATION_XLSX="${PPBEMO_ANNOTATION_XLSX:-$DATA_ROOT/Psychological_data/Emotion_label.xlsx}"
LOGDIR="$ROOT/logs/ppbemo"
mkdir -p "$LOGDIR"
SUMMARY_LOG="$LOGDIR/ppbemo_all_visuals_$(date +%Y%m%d_%H%M%S).log"

VIDEO_COLUMNS=(face_crgb face_lrgb face_rrgb face_cir body road)
NUM_FRAMES="${NUM_FRAMES:-3}"
PROMPT_SET="${PROMPT_SET:-ppbemo_natural_7}"
RUN_PREFIX="${RUN_PREFIX:-ppbemo_pipeline_exact}"

{
  echo "[INFO] root=$ROOT"
  echo "[INFO] data_root=$DATA_ROOT"
  echo "[INFO] annotation_xlsx=$ANNOTATION_XLSX"
  echo "[INFO] num_frames=$NUM_FRAMES prompt_set=$PROMPT_SET"
} | tee -a "$SUMMARY_LOG"

for col in "${VIDEO_COLUMNS[@]}"; do
  count="$($PY - <<PY | tail -n 1
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
path = Path('$ROOT/src/clip_ppbemo_emotion_train.py')
spec = spec_from_file_location('clip_ppbemo_emotion_train', path)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
mod.log = lambda message: None
try:
    samples = mod.collect_samples('$DATA_ROOT', '$ANNOTATION_XLSX', '$col', 0)
    print(len(samples))
except Exception:
    print(0)
PY
)"

  if [ "$count" -lt 10 ]; then
    echo "[SKIP] $col available_samples=$count" | tee -a "$SUMMARY_LOG"
    continue
  fi

  echo "[RUN] $col available_samples=$count" | tee -a "$SUMMARY_LOG"
  (
    cd "$ROOT"
    VIDEO_COLUMN="$col" \
    NUM_FRAMES="$NUM_FRAMES" \
    PROMPT_SET="$PROMPT_SET" \
    RUN_NAME="${RUN_PREFIX}_${col}" \
    "$RUNNER"
  ) 2>&1 | tee -a "$SUMMARY_LOG"
done

echo "[DONE] summary_log=$SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
