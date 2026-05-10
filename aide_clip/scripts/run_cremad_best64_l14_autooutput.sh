#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/cremad}"
CREMAD_ROOT="${CREMAD_ROOT:-$PROJECT_ROOT/data/crema_d}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$PROJECT_ROOT/cache/cremad_features_vavl_seed42_f16_sequence}"
LOCAL_CLIP_MODEL_DIR="${LOCAL_CLIP_MODEL_DIR:-/data1/yanjing/models/clip-vit-large-patch14}"

mkdir -p "$RESULT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
tag="${1:-best64_l14}"

if [[ $# -gt 0 ]]; then
  shift
fi

safe_tag="$(printf '%s' "$tag" | tr ' /' '__')"
run_stem="cremad_${safe_tag}_${timestamp}"

output_path="$RESULT_DIR/${run_stem}.json"
checkpoint_path="$RESULT_DIR/${run_stem}.pt"
log_path="$RESULT_DIR/${run_stem}.log"

echo "[RUN] project_root=$PROJECT_ROOT"
echo "[RUN] python_bin=$PYTHON_BIN"
echo "[RUN] cremad_root=$CREMAD_ROOT"
echo "[RUN] model_dir=$LOCAL_CLIP_MODEL_DIR"
echo "[RUN] feature_cache_dir=$FEATURE_CACHE_DIR"
echo "[RUN] output=$output_path"
echo "[RUN] checkpoint=$checkpoint_path"
echo "[RUN] log=$log_path"

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
"$PYTHON_BIN" src/clip_cremad_emotion_train.py \
  --clip_mode offline_only \
  --cremad_root "$CREMAD_ROOT" \
  --cv_mode split \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --seed 42 \
  --video_extensions .flv,.mp4 \
  --model_id "$LOCAL_CLIP_MODEL_DIR" \
  --prompt_set cremad_6_facial_cues \
  --epochs 30 \
  --extract_batch_size 32 \
  --train_batch_size 32 \
  --lr 1.5e-4 \
  --weight_decay 5e-4 \
  --max_grad_norm 1.0 \
  --num_frames 16 \
  --frame_sampling_mode uniform \
  --feature_layout sequence \
  --sampling_window_start 0.4 \
  --sampling_window_end 0.9 \
  --diff_alpha 0.6 \
  --diff_beta 0.4 \
  --min_gap_ratio 0.08 \
  --score_smooth_window 3 \
  --frame_diff_metric gray_l1 \
  --ref_frame_ratio 0.1 \
  --adapter_hidden_dim 256 \
  --adapter_dropout 0.2 \
  --adapter_head_type baseline \
  --temporal_head transformer \
  --temporal_num_heads 4 \
  --temporal_num_layers 2 \
  --temporal_pool_mode hybrid \
  --disable_class_weight \
  --label_smoothing 0.03 \
  --loss_type ce \
  --focal_gamma 1.5 \
  --select_metric weighted_f1 \
  --disable_test_ensemble \
  --ensemble_group_size 2 \
  --strict_frozen_clip \
  --disable_global_logit_scale \
  --disable_prompt_weight \
  --disable_class_temperature \
  --disable_class_bias \
  --use_amp \
  --early_stopping_patience 6 \
  --early_stopping_min_delta 1e-4 \
  --feature_cache_dir "$FEATURE_CACHE_DIR" \
  --checkpoint_output "$checkpoint_path" \
  --log_file "$log_path" \
  --output "$output_path" \
  "$@"