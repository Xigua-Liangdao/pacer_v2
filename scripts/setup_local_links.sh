#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_AIDE_CLIP_DATA="/data1/yanjing/talk2bev/aide_clip/data"
SRC_YAWDD_REPO="/data1/yanjing/talk2bev/fatigue-drive-yawning-detection"
SRC_AIDE_DATASET="/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset"
SRC_RESULTS="/data1/yanjing/talk2bev/aide_clip/results"

mkdir -p "$ROOT/aide_clip" "$ROOT/external_data"

ln -sfn "$SRC_AIDE_CLIP_DATA" "$ROOT/aide_clip/data"
ln -sfn "$SRC_YAWDD_REPO" "$ROOT/fatigue-drive-yawning-detection"
ln -sfn "$SRC_AIDE_DATASET" "$ROOT/external_data/AIDE_Dataset"
ln -sfn "$SRC_RESULTS" "$ROOT/external_data/source_results"

echo "[DONE] linked aide_clip/data -> $SRC_AIDE_CLIP_DATA"
echo "[DONE] linked fatigue-drive-yawning-detection -> $SRC_YAWDD_REPO"
echo "[DONE] linked external_data/AIDE_Dataset -> $SRC_AIDE_DATASET"
echo "[DONE] linked external_data/source_results -> $SRC_RESULTS"
