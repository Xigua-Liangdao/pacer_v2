#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
GPU_ID="${GPU_ID:-0}"
MODEL_ID="${MODEL_ID:-openai/clip-vit-base-patch32}"
CLIP_MODE="${CLIP_MODE:-auto}"
SEED="${SEED:-42}"
TRAINING_SEED="${TRAINING_SEED:-$SEED}"
EPOCHS="${EPOCHS:-40}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"

YAWDD_ROOT="${YAWDD_ROOT:-$ROOT/data/yawdd}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/yawdd/main_seed${SEED}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/yawdd_features}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"

cd "$ROOT"
"$PYTHON_BIN" aide_clip/src/clip_yawdd_emotion_train.py \
  --yawdd_root "$YAWDD_ROOT" \
  --label_mode binary --eval_mode single \
  --cv_mode split --split_mode speaker_independent \
  --epochs "$EPOCHS" --max_sequences "$MAX_SEQUENCES" \
  --clip_mode "$CLIP_MODE" --model_id "$MODEL_ID" \
  --prompt_template "The driver looks <LABEL>." --prompt_set yawdd_facial_cues \
  --feature_layout pooled --temporal_head none --temporal_module none \
  --pool_adapter_variant legacy \
  --train_batch_size 64 --extract_batch_size 32 \
  --weight_decay 1e-2 --max_grad_norm 1.0 \
  --adapter_hidden_dim 512 --adapter_dropout 0.3 --adapter_mode full \
  --adapter_use_prompt_weight on --adapter_use_class_temperature on --adapter_use_class_bias on \
  --select_metric weighted_f1 \
  --num_frames 10 --frame_sampling_mode diff_guided \
  --lr 1e-4 --label_smoothing 0.01 \
  --loss_type focal --focal_gamma 2.0 --use_class_weight \
  --disable_test_ensemble \
  --seed "$SEED" --training_seed "$TRAINING_SEED" \
  --feature_cache_dir "$CACHE_DIR" \
  --output "$OUT_DIR/result.json" \
  --checkpoint_output "$OUT_DIR/best.ckpt.pt" \
  --device "$DEVICE" --gpu_id "$GPU_ID"
