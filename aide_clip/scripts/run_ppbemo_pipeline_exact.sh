#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
SCRIPT="$ROOT/src/clip_ppbemo_emotion_train.py"
OUTDIR="$ROOT/results/ppbemo"
LOGDIR="$ROOT/logs/ppbemo"
CACHEDIR="$ROOT/cache/ppbemo_features"
mkdir -p "$OUTDIR" "$LOGDIR" "$CACHEDIR"

DATA_ROOT="${PPBEMO_ROOT:-$ROOT/data/bbpemo}"
ANNOTATION_XLSX="${PPBEMO_ANNOTATION_XLSX:-$DATA_ROOT/Psychological_data/Emotion_label.xlsx}"
VIDEO_COLUMN="${VIDEO_COLUMN:-face_crgb}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_ID="${MODEL_ID:-openai/clip-vit-base-patch32}"
RUN_NAME="${RUN_NAME:-ppbemo_pipeline_exact_${VIDEO_COLUMN}}"
OUTPUT="$OUTDIR/${RUN_NAME}.json"
CHECKPOINT="$OUTDIR/${RUN_NAME}.ckpt.pt"
LOGFILE="$LOGDIR/${RUN_NAME}.log"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

COMMON_ARGS=(
  --data_root "$DATA_ROOT"
  --annotation_xlsx "$ANNOTATION_XLSX"
  --video_column "$VIDEO_COLUMN"
  --device "$DEVICE"
  --clip_mode offline_only
  --model_id "$MODEL_ID"
  --strict_frozen_clip
  --prompt_set ppbemo_7
  --epochs 40
  --batch_size 32
  --lr 1.5e-4
  --weight_decay 5e-4
  --num_frames 3
  --adapter_hidden_dim 1024
  --adapter_dropout 0.2
  --use_class_weight
  --label_smoothing 0.03
  --select_metric weighted_f1
  --use_test_ensemble
  --ensemble_group_size 2
  --seed 42
  --feature_cache_dir "$CACHEDIR"
  --checkpoint_output "$CHECKPOINT"
  --output "$OUTPUT"
)

echo "[RUN] $RUN_NAME"
echo "[RUN] log -> $LOGFILE"
echo "[RUN] output -> $OUTPUT"
"$PY" "$SCRIPT" "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$LOGFILE"
echo "[DONE] $RUN_NAME"
