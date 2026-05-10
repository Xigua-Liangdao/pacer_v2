#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/ravdess}"
RAVDESS_ROOT="${RAVDESS_ROOT:-$PROJECT_ROOT/data/RAVDESS}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$PROJECT_ROOT/cache/ravdess_features}"
LOCAL_CLIP_MODEL_DIR="${LOCAL_CLIP_MODEL_DIR:-/data1/yanjing/models/clip-vit-large-patch14}"

mkdir -p "$RESULT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
tag="${1:-baseline61}"

if [[ $# -gt 0 ]]; then
  shift
fi

safe_tag="$(printf '%s' "$tag" | tr ' /' '__')"
run_stem="ravdess_${safe_tag}_${timestamp}"

output_path="$RESULT_DIR/${run_stem}.json"
checkpoint_path="$RESULT_DIR/${run_stem}.pt"
log_path="$RESULT_DIR/${run_stem}.log"

echo "[RUN] project_root=$PROJECT_ROOT"
echo "[RUN] python_bin=$PYTHON_BIN"
echo "[RUN] model_dir=$LOCAL_CLIP_MODEL_DIR"
echo "[RUN] output=$output_path"
echo "[RUN] checkpoint=$checkpoint_path"
echo "[RUN] log=$log_path"

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
"$PYTHON_BIN" src/clip_ravdess_emotion_train.py \
  --clip_mode offline_only \
  --experiment_name custom \
  --ravdess_root "$RAVDESS_ROOT" \
  --split_mode benchmark_5fold \
  --benchmark_test_fold 0 \
  --benchmark_val_fold 1 \
  --model_id /data1/yanjing/models/clip-vit-base-patch16 \
  --prompt_set ravdess_8_facial_cues \
  --epochs 80 \
  --extract_batch_size 32 \
  --train_batch_size 32 \
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
  --temporal_pooling mean \
  --loss_type focal \
  --focal_gamma 1.0 \
  --label_smoothing 0.1\
  --select_metric weighted_f1 \
  --use_test_ensemble \
  --ensemble_group_size 2 \
  --strict_frozen_clip \
  --disable_prompt_weight \
  --disable_class_temperature \
  --disable_class_bias \
  --disable_class_weight \
  --disable_amp \
  --lr_scheduler none \
  --early_stopping_patience 0 \
  --early_stopping_min_delta 0.0 \
  --run_zero_shot_eval \
  --report_train_metrics \
  --seed 45 \
  --allowed_modalities 02 \
  --allowed_vocal_channels 01 \
  --allowed_intensities 01,02 \
  --feature_cache_dir "$FEATURE_CACHE_DIR" \
  --checkpoint_output "$checkpoint_path" \
  --log_file "$log_path" \
  --output "$output_path" \
  "$@"
