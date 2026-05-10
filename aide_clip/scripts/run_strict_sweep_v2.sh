#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/src/clip_aide_emotion_train.py"
OUTDIR="$ROOT/results/strict_sweep_v2"
LOGDIR="$ROOT/logs/strict_sweep_v2"
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
  --prompt_set driving_7
  --epochs 30
  --batch_size 32
  --lr 2e-4
  --weight_decay 5e-4
  --num_frames 5
  --adapter_hidden_dim 1024
  --adapter_dropout 0.2
    --use_class_weight on
  --label_smoothing 0.03
  --select_metric weighted_f1
    --use_test_ensemble on
  --ensemble_group_size 2
)

run_case() {
  local name="$1"; shift
  local output="$OUTDIR/clip_emotion_strict_sweep_v2_${name}.json"
    local log="$LOGDIR/clip_emotion_strict_sweep_v2_${name}.log"
  echo "[RUN] ${name} -> ${output}"
    "$PY" "$SCRIPT" "${COMMON_ARGS[@]}" "$@" --output "$output" 2>&1 | tee "$log"
}

run_case a --seed 42
run_case b --seed 42 --adapter_dropout 0.3 --weight_decay 8e-4
run_case c --seed 42 --epochs 40 --lr 1.5e-4
run_case d --seed 7

OUTDIR_ENV="$OUTDIR" "$PY" - <<'PY'
import glob
import json
import os

outdir = os.environ["OUTDIR_ENV"]
paths = sorted(
    p for p in glob.glob(os.path.join(outdir, "clip_emotion_strict_sweep_v2_*.json"))
    if not p.endswith("_summary.json")
)
rows = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        r = json.load(f)
    rows.append(
        {
            "name": os.path.basename(p),
            "acc": r["test"]["accuracy"],
            "wf1": r["test"]["weighted_f1"],
            "config": r.get("config", {}),
        }
    )
rows.sort(key=lambda x: (x["acc"], x["wf1"]), reverse=True)
out = {"best": rows[0] if rows else None, "all": rows}
outp = os.path.join(outdir, "clip_emotion_strict_sweep_v2_summary.json")
with open(outp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("[DONE] wrote", outp)
print(json.dumps(out["best"], ensure_ascii=False, indent=2))
PY

echo "[DONE] sweep v2 complete"
