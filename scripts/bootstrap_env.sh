#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PACER_V2_ROOT="$ROOT"
export PYTHON_BIN="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export AIDE_ROOT="${AIDE_ROOT:-/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset}"
export AIDE_ANNOTATION_ROOT="${AIDE_ANNOTATION_ROOT:-$AIDE_ROOT/annotation}"
export YAWDD_ROOT="${YAWDD_ROOT:-/data1/yanjing/talk2bev/fatigue-drive-yawning-detection/extracted_face_multi4}"

echo "PACER_V2_ROOT=$PACER_V2_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "AIDE_ROOT=$AIDE_ROOT"
echo "AIDE_ANNOTATION_ROOT=$AIDE_ANNOTATION_ROOT"
echo "YAWDD_ROOT=$YAWDD_ROOT"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"