#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
GPU_ID="${GPU_ID:-0}"
SEEDS="${SEEDS:-42,123,2024}"
YAWDD_ROOT="${YAWDD_ROOT:-$ROOT/data/yawdd}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/yawdd/backbone_finetune}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/yawdd_features}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"

cd "$ROOT"
"$PYTHON_BIN" aide_clip/src/yawdd_backbone_finetune.py \
  --yawdd_root "$YAWDD_ROOT" \
  --output_dir "$OUT_DIR" \
  --feature_cache_dir "$CACHE_DIR" \
  --seeds "$SEEDS" \
  --device "$DEVICE" --gpu_id "$GPU_ID"
