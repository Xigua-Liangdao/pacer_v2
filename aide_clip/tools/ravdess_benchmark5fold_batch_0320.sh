#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/yanjing/talk2bev/aide_clip"
PYTHON="/home/yanjing/anaconda3/envs/mmtl/bin/python"
SCRIPT="$ROOT/src/clip_ravdess_emotion_train.py"
OUT_DIR="$ROOT/results/ravdess/batch_0320_fold0_val1_v2"
SUMMARY_JSONL="$OUT_DIR/summary.jsonl"
mkdir -p "$OUT_DIR"
: > "$SUMMARY_JSONL"

common_args=(
  --clip_mode offline_only
  --split_mode benchmark_5fold
  --benchmark_test_fold 0
  --benchmark_val_fold 1
  --allowed_modalities 02
  --allowed_vocal_channels 01
  --allowed_intensities 01,02
  --video_extensions .mp4
  --adapter_hidden_dim 256
  --disable_prompt_weight
  --disable_class_temperature
  --disable_class_bias
  --run_zero_shot_eval
  --report_train_metrics
)

run_exp() {
  local name="$1"
  shift
  local output="$OUT_DIR/${name}.json"
  local ckpt="$OUT_DIR/${name}.ckpt.pt"
  local log="$OUT_DIR/${name}.log"

  echo "[BATCH] start $name"
  "$PYTHON" "$SCRIPT" \
    "${common_args[@]}" \
    --output "$output" \
    --checkpoint_output "$ckpt" \
    --log_file "$log" \
    "$@"

  "$PYTHON" - <<PY
import json
from pathlib import Path
path = Path(${output@Q})
obj = json.loads(path.read_text(encoding='utf-8'))
row = {
    'name': ${name@Q},
    'prompt_set': obj['config'].get('prompt_set'),
    'loss_type': obj['config'].get('loss_type'),
  'focal_gamma': obj['config'].get('focal_gamma'),
    'label_smoothing': obj['config'].get('label_smoothing'),
    'use_class_weight': obj['config'].get('use_class_weight'),
    'epochs': obj['config'].get('epochs'),
    'val_accuracy': obj['val'].get('accuracy'),
    'val_weighted_f1': obj['val'].get('weighted_f1'),
    'test_accuracy': obj['test'].get('accuracy'),
    'test_weighted_f1': obj['test'].get('weighted_f1'),
    'zero_shot_val_accuracy': (obj.get('zero_shot_val_metrics') or {}).get('accuracy'),
    'zero_shot_val_weighted_f1': (obj.get('zero_shot_val_metrics') or {}).get('weighted_f1'),
    'zero_shot_test_accuracy': (obj.get('zero_shot_test_metrics') or {}).get('accuracy'),
    'zero_shot_test_weighted_f1': (obj.get('zero_shot_test_metrics') or {}).get('weighted_f1'),
}
print(json.dumps(row, ensure_ascii=False))
with open(${SUMMARY_JSONL@Q}, 'a', encoding='utf-8') as f:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
}

run_exp "exp01_generic_ce_ls001" \
  --prompt_set ravdess_8 \
  --loss_type ce \
  --label_smoothing 0.01 \
  --epochs 60

run_exp "exp02_facial_ce_ls001" \
  --prompt_set ravdess_8_facial_cues \
  --loss_type ce \
  --label_smoothing 0.01 \
  --epochs 60

run_exp "exp03_facial_ce_ls000" \
  --prompt_set ravdess_8_facial_cues \
  --loss_type ce \
  --label_smoothing 0.0 \
  --epochs 60

run_exp "exp04_facial_focal15_ls000" \
  --prompt_set ravdess_8_facial_cues \
  --loss_type focal \
  --focal_gamma 1.5 \
  --label_smoothing 0.0 \
  --epochs 80

run_exp "exp05_facial_focal20_ls000" \
  --prompt_set ravdess_8_facial_cues \
  --loss_type focal \
  --focal_gamma 2.0 \
  --label_smoothing 0.0 \
  --epochs 80

run_exp "exp06_facial_focal15_ls000_nocw" \
  --prompt_set ravdess_8_facial_cues \
  --loss_type focal \
  --focal_gamma 1.5 \
  --label_smoothing 0.0 \
  --disable_class_weight \
  --epochs 80

echo "[BATCH] done -> $SUMMARY_JSONL"
