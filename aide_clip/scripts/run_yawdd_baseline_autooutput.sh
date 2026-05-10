#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/yawdd}"
YAWDD_ROOT="${YAWDD_ROOT:-$PROJECT_ROOT/data/yawdd}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$PROJECT_ROOT/cache/yawdd_features}"

mkdir -p "$RESULT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
tag="${1:-baseline}"

if [[ $# -gt 0 ]]; then
  shift
fi

safe_tag="$(printf '%s' "$tag" | tr ' /' '__')"
run_stem="yawdd_${safe_tag}_${timestamp}"

output_path="$RESULT_DIR/${run_stem}.json"
checkpoint_path="$RESULT_DIR/${run_stem}.pt"
log_path="$RESULT_DIR/${run_stem}.log"

echo "[RUN] project_root=$PROJECT_ROOT"
echo "[RUN] python_bin=$PYTHON_BIN"
echo "[RUN] yawdd_root=$YAWDD_ROOT"
echo "[RUN] output=$output_path"
echo "[RUN] checkpoint=$checkpoint_path"
echo "[RUN] log=$log_path"

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
"$PYTHON_BIN" src/clip_yawdd_emotion_train.py \
  --clip_mode offline_only \
  --yawdd_root "$YAWDD_ROOT" \
  --label_mode binary \
  --cv_mode 5fold \
  --fold_idx 0 \
  --model_id /data1/yanjing/models/clip-vit-base-patch16 \
  --prompt_set yawdd_facial_cues \
  --epochs 40 \
  --extract_batch_size 32 \
  --train_batch_size 64 \
  --lr 1.5e-4 \
  --weight_decay 1e-2 \
  --max_grad_norm 1.0 \
  --num_frames 5 \
  --frame_sampling_mode middle_late \
  --sampling_window_start 0.4 \
  --sampling_window_end 0.9 \
  --diff_alpha 0.6 \
  --diff_beta 0.4 \
  --min_gap_ratio 0.08 \
  --score_smooth_window 3 \
  --frame_diff_metric gray_l1 \
  --feature_layout sequence \
  --adapter_hidden_dim 256 \
  --adapter_dropout 0.3 \
  --temporal_head transformer \
  --temporal_num_heads 4 \
  --temporal_num_layers 1 \
  --temporal_pool_mode mean \
  --loss_type focal \
  --focal_gamma 1.0 \
  --label_smoothing 0.1 \
  --select_metric weighted_f1 \
  --use_test_ensemble \
  --ensemble_group_size 2 \
  --disable_class_weight \
  --disable_amp \
  --feature_cache_dir "$FEATURE_CACHE_DIR" \
  --run_zero_shot_eval \
  --checkpoint_output "$checkpoint_path" \
  --log_file "$log_path" \
  --output "$output_path" \
  "$@"