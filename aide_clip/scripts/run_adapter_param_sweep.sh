#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
SCRIPT="$ROOT/src/clip_aide_emotion_train.py"
OUTDIR="$ROOT/results/adapter_sweep"
LOGDIR="$ROOT/logs/adapter_sweep"
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
  --label_smoothing 0.03
  --select_metric weighted_f1
  --seed 42
  --feature_cache_dir "$ROOT/cache/features"
  --prompt_set driving_7
    --use_class_weight on
    --use_test_ensemble on
  --ensemble_group_size 2
)

run_case() {
  local name="$1"; shift
  local hidden="$1"; shift
  local dropout="$1"; shift
  local output="$OUTDIR/${name}.json"
  local log="$LOGDIR/${name}.log"
  echo "[RUN] ${name} hidden=${hidden} dropout=${dropout}"
  "$PY" "$SCRIPT" "${COMMON_ARGS[@]}" \
    --adapter_hidden_dim "$hidden" \
    --adapter_dropout "$dropout" \
    --output "$output" 2>&1 | tee "$log"
  echo "[DONE] ${name} -> ${output}"
}

run_case h512_d02 512 0.2
run_case h768_d01 768 0.1
run_case h768_d02 768 0.2
run_case h1024_d01 1024 0.1
run_case h1024_d02 1024 0.2
run_case h1024_d03 1024 0.3
run_case h1536_d02 1536 0.2
run_case h2048_d02 2048 0.2

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
    cfg = result.get("config", {})
    rows.append(
        {
            "name": os.path.splitext(os.path.basename(path))[0],
            "acc": result["test"]["accuracy"],
            "wf1": result["test"]["weighted_f1"],
            "adapter_hidden_dim": cfg.get("adapter_hidden_dim"),
            "adapter_dropout": cfg.get("adapter_dropout"),
            "checkpoint_output": cfg.get("checkpoint_output"),
        }
    )

ref = next((row for row in rows if row["name"] == "h1024_d02"), None)
if ref is not None:
    for row in rows:
        row["delta_acc_vs_h1024_d02"] = round(row["acc"] - ref["acc"], 6)
        row["delta_wf1_vs_h1024_d02"] = round(row["wf1"] - ref["wf1"], 6)

rows.sort(key=lambda x: (x["wf1"], x["acc"]), reverse=True)
summary = {
    "reference": ref,
    "results": rows,
}
summary_path = os.path.join(outdir, "adapter_sweep_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[DONE] wrote", summary_path)
if rows:
    print("[BEST]", rows[0])
PY

echo "[DONE] adapter sweep complete"
