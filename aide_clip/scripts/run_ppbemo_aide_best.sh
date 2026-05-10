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
RUN_NAME="${RUN_NAME:-ppbemo_aide_best_${VIDEO_COLUMN}}"
OUTPUT="$OUTDIR/${RUN_NAME}.json"
CHECKPOINT="$OUTDIR/${RUN_NAME}.ckpt.pt"
LOGFILE="$LOGDIR/${RUN_NAME}.log"
PROMPT_SET="${PROMPT_SET:-ppbemo_natural_7}"
SPLIT_MODE="${SPLIT_MODE:-participant}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
NUM_FRAMES="${NUM_FRAMES:-5}"
ADAPTER_HIDDEN_DIM="${ADAPTER_HIDDEN_DIM:-2048}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.2}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.03}"
ENSEMBLE_GROUP_SIZE="${ENSEMBLE_GROUP_SIZE:-2}"
SEED="${SEED:-42}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

COMMON_ARGS=(
  --data_root "$DATA_ROOT"
  --annotation_xlsx "$ANNOTATION_XLSX"
  --video_column "$VIDEO_COLUMN"
  --split_mode "$SPLIT_MODE"
  --device "$DEVICE"
  --clip_mode offline_only
  --model_id "$MODEL_ID"
  --strict_frozen_clip
  --prompt_set "$PROMPT_SET"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --num_frames "$NUM_FRAMES"
  --adapter_hidden_dim "$ADAPTER_HIDDEN_DIM"
  --adapter_dropout "$ADAPTER_DROPOUT"
  --use_class_weight
  --label_smoothing "$LABEL_SMOOTHING"
  --select_metric weighted_f1
  --use_test_ensemble
  --ensemble_group_size "$ENSEMBLE_GROUP_SIZE"
  --seed "$SEED"
  --feature_cache_dir "$CACHEDIR"
  --checkpoint_output "$CHECKPOINT"
  --output "$OUTPUT"
)

echo "[RUN] $RUN_NAME"
echo "[RUN] log -> $LOGFILE"
echo "[RUN] output -> $OUTPUT"
"$PY" "$SCRIPT" "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$LOGFILE"
echo "[DONE] $RUN_NAME"
