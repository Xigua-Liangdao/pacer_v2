#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/ddd}"
DDD_ROOT="${DDD_ROOT:-$PROJECT_ROOT/data/DDD/train_data}"

mkdir -p "$RESULT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
tag="${1:-baseline}"

if [[ $# -gt 0 ]]; then
  shift
fi

safe_tag="$(printf '%s' "$tag" | tr ' /' '__')"
run_stem="ddd_${safe_tag}_${timestamp}"

output_path="$RESULT_DIR/${run_stem}.json"
checkpoint_path="$RESULT_DIR/${run_stem}.pt"
log_path="$RESULT_DIR/${run_stem}.log"

echo "[RUN] project_root=$PROJECT_ROOT"
echo "[RUN] python_bin=$PYTHON_BIN"
echo "[RUN] ddd_root=$DDD_ROOT"
echo "[RUN] output=$output_path"
echo "[RUN] checkpoint=$checkpoint_path"
echo "[RUN] log=$log_path"

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
"$PYTHON_BIN" src/clip_ddd_emotion_train.py \
  --clip_mode offline_only \
  --ddd_root "$DDD_ROOT" \
  --split_mode clip_id \
  --model_id /data1/yanjing/models/clip-vit-base-patch16 \
  --prompt_set ddd_binary_facial_cues \
  --epochs 30 \
  --extract_batch_size 64 \
  --train_batch_size 256 \
  --lr 1.5e-4 \
  --weight_decay 5e-4 \
  --max_grad_norm 1.0 \
  --num_frames 1 \
  --feature_layout pooled \
  --adapter_hidden_dim 256 \
  --adapter_dropout 0.2 \
  --label_smoothing 0.0 \
  --select_metric weighted_f1 \
  --run_zero_shot_eval \
  --checkpoint_output "$checkpoint_path" \
  --log_file "$log_path" \
  --output "$output_path" \
  "$@"