#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
SCRIPT="$ROOT/src/clip_aide_emotion_train.py"
OUTDIR="$ROOT/results/repro"
LOGDIR="$ROOT/logs"
CACHEDIR="${FEATURE_CACHE_DIR:-$ROOT/cache/features}"
mkdir -p "$OUTDIR" "$LOGDIR" "$CACHEDIR"

RUN_NAME="${RUN_NAME:-clip_emotion_strict_repro_c_transformer_try}"
OUTPUT="$OUTDIR/${RUN_NAME}.json"
CHECKPOINT="$OUTDIR/${RUN_NAME}.ckpt.pt"
LOGFILE="$LOGDIR/${RUN_NAME}.log"

for target in "$OUTPUT" "$CHECKPOINT" "$LOGFILE"; do
  if [[ -e "$target" ]]; then
    echo "[ERROR] target already exists: $target" >&2
    echo "[ERROR] set RUN_NAME to a new value to avoid overwriting prior try artifacts." >&2
    exit 1
  fi
done

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

COMMON_ARGS=(
  --device "${DEVICE:-cuda:0}"
  --clip_mode "${CLIP_MODE:-auto}"
  --model_id "${MODEL_ID:-openai/clip-vit-base-patch32}"
  --prompt_template "${PROMPT_TEMPLATE:-Driver is <LABEL>.}"
  --prompt_set "${PROMPT_SET:-driving_7}"
  --epochs "${EPOCHS:-40}"
  --batch_size "${BATCH_SIZE:-32}"
  --lr "${LR:-1.5e-4}"
  --weight_decay "${WEIGHT_DECAY:-5e-4}"
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
  --num_frames "${NUM_FRAMES:-5}"
  --adapter_hidden_dim "${ADAPTER_HIDDEN_DIM:-1024}"
  --adapter_dropout "${ADAPTER_DROPOUT:-0.2}"
  --use_class_weight
  --label_smoothing "${LABEL_SMOOTHING:-0.03}"
  --select_metric "${SELECT_METRIC:-weighted_f1}"
  --use_test_ensemble
  --ensemble_group_size "${ENSEMBLE_GROUP_SIZE:-2}"
  --strict_frozen_clip
  --seed "${SEED:-42}"
  --feature_cache_dir "$CACHEDIR"
  --checkpoint_output "$CHECKPOINT"
  --output "$OUTPUT"
  --temporal_head transformer
)

if [[ -n "${AIDE_ROOT:-}" ]]; then
  COMMON_ARGS+=(--aide_root "$AIDE_ROOT")
fi

if [[ -n "${AIDE_ANNOTATION_ROOT:-}" ]]; then
  COMMON_ARGS+=(--annotation_root "$AIDE_ANNOTATION_ROOT")
fi

if [[ -n "${MAX_SEQUENCES:-}" ]]; then
  COMMON_ARGS+=(--max_sequences "$MAX_SEQUENCES")
fi

echo "[RUN] $RUN_NAME"
echo "[RUN] log -> $LOGFILE"
echo "[RUN] output -> $OUTPUT"
echo "[RUN] checkpoint -> $CHECKPOINT"
"$PY" "$SCRIPT" "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$LOGFILE"
echo "[DONE] $RUN_NAME"