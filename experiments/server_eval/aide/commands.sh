#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
CONDA_ENV="${CONDA_ENV:-mmtl}"

AIDE_ROOT="${DATA_ROOT:-${AIDE_ROOT:-}}"
if [[ -z "$AIDE_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT or AIDE_ROOT must be set for the AIDE server evaluation template." >&2
  exit 2
fi
AIDE_ANNOTATION_ROOT="${AIDE_ANNOTATION_ROOT:-$AIDE_ROOT/annotation}"
MODEL_ID="${MODEL_ID:-openai/clip-vit-base-patch32}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$PROJECT_ROOT/aide_clip/cache/features}"
OUTPUT_JSON="${OUTPUT_JSON:-$RUN_DIR/candidate_raw.json}"
CHECKPOINT_OUTPUT="${CHECKPOINT_OUTPUT:-$RUN_DIR/candidate.ckpt.pt}"

mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[RUN] experiment=aide"
echo "[RUN] project_root=$PROJECT_ROOT"
echo "[RUN] conda_env=$CONDA_ENV"
echo "[RUN] gpu_id=$GPU_ID"
echo "[RUN] data_root=$AIDE_ROOT"
echo "[RUN] output_json=$OUTPUT_JSON"

"$PYTHON_BIN" "$PROJECT_ROOT/aide_clip/src/clip_aide_emotion_train.py" \
  --aide_root "$AIDE_ROOT" \
  --annotation_root "$AIDE_ANNOTATION_ROOT" \
  --device cuda:0 \
  --clip_mode offline_only \
  --model_id "$MODEL_ID" \
  --strict_frozen_clip on \
  --epochs 40 \
  --batch_size 32 \
  --lr 1.5e-4 \
  --weight_decay 5e-4 \
  --num_frames 5 \
  --label_smoothing 0.03 \
  --select_metric weighted_f1 \
  --seed "$SEED" \
  --feature_cache_dir "$FEATURE_CACHE_DIR" \
  --prompt_set driving_7 \
  --use_class_weight on \
  --use_test_ensemble on \
  --ensemble_group_size 2 \
  --adapter_hidden_dim 2048 \
  --adapter_dropout 0.2 \
  --checkpoint_output "$CHECKPOINT_OUTPUT" \
  --output "$OUTPUT_JSON"