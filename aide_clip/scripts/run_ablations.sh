#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
SCRIPT="$ROOT/src/clip_aide_emotion_train.py"
OUTDIR="$ROOT/results/ablations"
LOGDIR="$ROOT/logs/ablations"
mkdir -p "$OUTDIR" "$LOGDIR"

AIDE_ROOT="${AIDE_ROOT:-/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset}"
ANNOTATION_ROOT="${AIDE_ANNOTATION_ROOT:-$AIDE_ROOT/annotation}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_ID="${MODEL_ID:-openai/clip-vit-base-patch32}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

COMMON_ARGS=(
  --aide_root "$AIDE_ROOT"
  --annotation_root "$ANNOTATION_ROOT"
  --device "$DEVICE"
  --clip_mode offline_only
  --model_id "$MODEL_ID"
  --strict_frozen_clip on
  --epochs 40
  --batch_size 32
  --lr 1.5e-4
  --weight_decay 5e-4
  --num_frames 5
  --adapter_hidden_dim 1024
  --adapter_dropout 0.2
  --label_smoothing 0.03
  --select_metric weighted_f1
  --seed 42
  --feature_cache_dir "$ROOT/cache/features"
)

run_case() {
  local name="$1"; shift
  local output="$OUTDIR/${name}.json"
  local log="$LOGDIR/${name}.log"
  echo "[RUN] ${name}"
  "$PY" "$SCRIPT" "${COMMON_ARGS[@]}" "$@" --output "$output" 2>&1 | tee "$log"
  echo "[DONE] ${name} -> ${output}"
}

run_case full_best \
  --prompt_set driving_7 \
  --use_class_weight on \
  --use_test_ensemble on \
  --ensemble_group_size 2

run_case no_test_ensemble \
  --prompt_set driving_7 \
  --use_class_weight on \
  --use_test_ensemble off

run_case no_prompt_weight \
  --prompt_set driving_7 \
  --use_class_weight on \
  --use_test_ensemble on \
  --ensemble_group_size 2 \
  --use_prompt_weight off

run_case no_class_temperature \
  --prompt_set driving_7 \
  --use_class_weight on \
  --use_test_ensemble on \
  --ensemble_group_size 2 \
  --use_class_temperature off

run_case no_class_bias \
  --prompt_set driving_7 \
  --use_class_weight on \
  --use_test_ensemble on \
  --ensemble_group_size 2 \
  --use_class_bias off

run_case single_prompt \
  --prompt_set single \
  --use_class_weight on \
  --use_test_ensemble on \
  --ensemble_group_size 2

run_case no_class_weight \
  --prompt_set driving_7 \
  --use_class_weight off \
  --use_test_ensemble on \
  --ensemble_group_size 2

run_case no_label_smoothing \
  --prompt_set driving_7 \
  --use_class_weight on \
  --use_test_ensemble on \
  --ensemble_group_size 2 \
  --label_smoothing 0.0

run_case one_frame_only \
  --prompt_set driving_7 \
  --use_class_weight on \
  --use_test_ensemble on \
  --ensemble_group_size 2 \
  --num_frames 1

OUTDIR_ENV="$OUTDIR" "$PY" - <<'PY'
import glob
import json
import os

outdir = os.environ["OUTDIR_ENV"]
rows = []
for path in sorted(glob.glob(os.path.join(outdir, "*.json"))):
    if path.endswith("summary.json"):
        continue
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)
    rows.append(
        {
            "name": os.path.splitext(os.path.basename(path))[0],
            "acc": result["test"]["accuracy"],
            "wf1": result["test"]["weighted_f1"],
            "config": result.get("config", {}),
        }
    )

full = next((row for row in rows if row["name"] == "full_best"), None)
if full is not None:
    for row in rows:
        row["delta_acc_vs_full"] = round(row["acc"] - full["acc"], 6)
        row["delta_wf1_vs_full"] = round(row["wf1"] - full["wf1"], 6)

rows.sort(key=lambda x: x["wf1"], reverse=True)
summary = {
    "reference": full,
    "results": rows,
}
summary_path = os.path.join(outdir, "ablation_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[DONE] wrote", summary_path)
PY

echo "[DONE] ablations complete"