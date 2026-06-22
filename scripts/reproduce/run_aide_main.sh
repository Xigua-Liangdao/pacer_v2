#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_ID="${MODEL_ID:-openai/clip-vit-base-patch32}"
CLIP_MODE="${CLIP_MODE:-auto}"
SEED="${SEED:-42}"
TRAINING_SEED="${TRAINING_SEED:-$SEED}"
EPOCHS="${EPOCHS:-40}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"

AIDE_ROOT="${AIDE_ROOT:-$ROOT/data/AIDE_Dataset}"
AIDE_ANNOTATION_ROOT="${AIDE_ANNOTATION_ROOT:-$AIDE_ROOT/annotation}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/aide/main_seed${SEED}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/aide_features}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"

cd "$ROOT"
"$PYTHON_BIN" aide_clip/src/clip_aide_emotion_train.py \
  --aide_root "$AIDE_ROOT" \
  --annotation_root "$AIDE_ANNOTATION_ROOT" \
  --seed "$SEED" --training_seed "$TRAINING_SEED" \
  --max_sequences "$MAX_SEQUENCES" \
  --strict_frozen_clip on --clip_mode "$CLIP_MODE" --device "$DEVICE" \
  --model_id "$MODEL_ID" \
  --prompt_template "Driver is <LABEL>." --prompt_set driving_7 \
  --num_frames 5 --epochs "$EPOCHS" --batch_size 32 \
  --lr 1.5e-4 --weight_decay 5e-4 --max_grad_norm 1.0 \
  --adapter_hidden_dim 1024 --adapter_dropout 0.2 \
  --pool_adapter_variant legacy --temporal_module none --adapter_mode full \
  --use_prompt_weight on --use_class_temperature on --use_class_bias on \
  --label_smoothing 0.01 --select_metric accuracy \
  --use_test_ensemble on \
  --feature_cache_dir "$CACHE_DIR" \
  --output "$OUT_DIR/result.json" \
  --checkpoint_output "$OUT_DIR/best.ckpt.pt"
