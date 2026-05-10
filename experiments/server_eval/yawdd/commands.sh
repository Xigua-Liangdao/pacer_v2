#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-7}"
CONDA_ENV="${CONDA_ENV:-mmtl}"

YAWDD_ROOT="${DATA_ROOT:-${YAWDD_ROOT:-}}"
if [[ -z "$YAWDD_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT or YAWDD_ROOT must be set for the YawDD server evaluation template." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-openai/clip-vit-base-patch32}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$PROJECT_ROOT/aide_clip/cache/yawdd_features}"
OUTPUT_JSON="${OUTPUT_JSON:-$RUN_DIR/candidate_raw.json}"
CHECKPOINT_OUTPUT="${CHECKPOINT_OUTPUT:-$RUN_DIR/candidate.ckpt.pt}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-$RUN_DIR/training.log}"

mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[RUN] experiment=yawdd"
echo "[RUN] project_root=$PROJECT_ROOT"
echo "[RUN] conda_env=$CONDA_ENV"
echo "[RUN] gpu_id=$GPU_ID"
echo "[RUN] data_root=$YAWDD_ROOT"
echo "[RUN] output_json=$OUTPUT_JSON"

"$PYTHON_BIN" "$PROJECT_ROOT/aide_clip/src/clip_yawdd_emotion_train.py" \
  --clip_mode offline_only \
  --yawdd_root "$YAWDD_ROOT" \
  --label_mode binary \
  --cv_mode 5fold \
  --fold_idx 0 \
  --model_id "$MODEL_ID" \
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
  --seed "$SEED" \
  --checkpoint_output "$CHECKPOINT_OUTPUT" \
  --log_file "$TRAIN_LOG_FILE" \
  --output "$OUTPUT_JSON"