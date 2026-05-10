#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/yanjing/anaconda3/envs/mmtl/bin/python}"
SCRIPT="$ROOT/src/clip_aide_emotion_train.py"

RESULT_DIR="${RESULT_DIR:-$ROOT/results/temporal_pool_ablation}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/temporal_pool_ablation}"
CACHE_DIR="${FEATURE_CACHE_DIR:-$ROOT/cache/features}"
RUN_PREFIX="${RUN_PREFIX:-aide_temporal_pool_h2048_d02cfg}"

AIDE_ROOT="${AIDE_ROOT:-/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset}"
ANNOTATION_ROOT="${AIDE_ANNOTATION_ROOT:-$AIDE_ROOT/annotation}"
DEVICE="${DEVICE:-cuda:0}"
CLIP_MODE="${CLIP_MODE:-offline_only}"
MODEL_ID="${MODEL_ID:-openai/clip-vit-base-patch32}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-Driver is <LABEL>.}"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$CACHE_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

COMMON_ARGS=(
  --aide_root "$AIDE_ROOT"
  --annotation_root "$ANNOTATION_ROOT"
  --device "$DEVICE"
  --clip_mode "$CLIP_MODE"
  --model_id "$MODEL_ID"
  --prompt_template "$PROMPT_TEMPLATE"
  --strict_frozen_clip
  --temporal_head transformer
  --prompt_set driving_7
  --num_frames 5
  --adapter_hidden_dim 2048
  --adapter_dropout 0.2
  --label_smoothing 0.03
  --epochs 40
  --lr 0.00015
    --weight_decay 0.0005
  --max_grad_norm 1.0
  --use_class_weight
  --use_test_ensemble
  --ensemble_group_size 2
  --seed 42
  --select_metric weighted_f1
    --batch_size 32
  --feature_cache_dir "$CACHE_DIR"
)

run_case() {
  local mode="$1"
  local output="$RESULT_DIR/${RUN_PREFIX}_${mode}.json"
  local checkpoint="$RESULT_DIR/${RUN_PREFIX}_${mode}.ckpt.pt"
  local log="$LOG_DIR/${RUN_PREFIX}_${mode}.log"

  for target in "$output" "$checkpoint" "$log"; do
    if [[ -e "$target" ]]; then
      echo "[ERROR] target already exists: $target" >&2
      echo "[ERROR] set RUN_PREFIX, RESULT_DIR, or LOG_DIR to avoid overwriting prior artifacts." >&2
      exit 1
    fi
  done

  echo "[RUN] temporal_pool_mode=$mode"
  echo "[RUN] log -> $log"
  echo "[RUN] output -> $output"
  echo "[RUN] checkpoint -> $checkpoint"

  "$PY" "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --temporal_pool_mode "$mode" \
    --checkpoint_output "$checkpoint" \
    --output "$output" \
    2>&1 | tee "$log"

  echo "[DONE] temporal_pool_mode=$mode"
}

run_case hybrid
run_case cls
run_case mean

RESULT_DIR_ENV="$RESULT_DIR" RUN_PREFIX_ENV="$RUN_PREFIX" "$PY" - <<'PY'
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR_ENV"])
run_prefix = os.environ["RUN_PREFIX_ENV"]
modes = ["hybrid", "cls", "mean"]
rows = []

for mode in modes:
    path = result_dir / f"{run_prefix}_{mode}.json"
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    rows.append(
        {
            "mode": mode,
            "accuracy": result["test"]["accuracy"],
            "weighted_f1": result["test"]["weighted_f1"],
            "output": str(path),
            "checkpoint": result["config"].get("checkpoint_output"),
        }
    )

summary = {"results": rows}
summary_json = result_dir / f"{run_prefix}_summary.json"
summary_csv = result_dir / f"{run_prefix}_summary.csv"
summary_md = result_dir / f"{run_prefix}_summary.md"

with summary_json.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

with summary_csv.open("w", encoding="utf-8") as handle:
    handle.write("mode,test_accuracy,test_weighted_f1,output,checkpoint\n")
    for row in rows:
        handle.write(
            f"{row['mode']},{row['accuracy']:.6f},{row['weighted_f1']:.6f},{row['output']},{row['checkpoint']}\n"
        )

md_lines = [
    "| temporal_pool_mode | test accuracy | test weighted_f1 |",
    "| --- | ---: | ---: |",
]
for row in rows:
    md_lines.append(
        f"| {row['mode']} | {row['accuracy']:.6f} | {row['weighted_f1']:.6f} |"
    )
summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print()
print("Summary:")
print("| temporal_pool_mode | test accuracy | test weighted_f1 |")
print("| --- | ---: | ---: |")
for row in rows:
    print(f"| {row['mode']} | {row['accuracy']:.6f} | {row['weighted_f1']:.6f} |")
print()
print(f"[DONE] wrote {summary_json}")
print(f"[DONE] wrote {summary_csv}")
print(f"[DONE] wrote {summary_md}")
PY

echo "[DONE] temporal pooling ablation complete"