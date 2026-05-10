#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/yanjing/talk2bev/aide_clip"
PYTHON="/home/yanjing/anaconda3/envs/mmtl/bin/python"
SCRIPT="$ROOT/src/clip_ravdess_emotion_train.py"
OUT_DIR="$ROOT/results/ravdess/exp06_variants_0320"
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
  --disable_class_weight
  --loss_type focal
  --focal_gamma 1.5
  --label_smoothing 0.0
  --epochs 80
  --run_zero_shot_eval
  --report_train_metrics
)

run_exp() {
  local name="$1"
  shift
  local output="$OUT_DIR/${name}.json"
  local ckpt="$OUT_DIR/${name}.ckpt.pt"
  local log="$OUT_DIR/${name}.log"

  echo "[VARIANT] start $name"
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
    'frame_sampling_mode': obj['config'].get('frame_sampling_mode'),
    'use_global_logit_scale': obj['config'].get('use_global_logit_scale'),
    'learned_global_logit_scale': obj.get('learned_global_logit_scale'),
    'val_accuracy': obj['val'].get('accuracy'),
    'val_weighted_f1': obj['val'].get('weighted_f1'),
    'test_accuracy': obj['test'].get('accuracy'),
    'test_weighted_f1': obj['test'].get('weighted_f1'),
    'zero_shot_val_accuracy': (obj.get('zero_shot_val_metrics') or {}).get('accuracy'),
    'zero_shot_test_accuracy': (obj.get('zero_shot_test_metrics') or {}).get('accuracy'),
}
print(json.dumps(row, ensure_ascii=False))
with open(${SUMMARY_JSONL@Q}, 'a', encoding='utf-8') as f:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
}

run_exp "expA_pairwise_prompts" \
  --prompt_set ravdess_8_pairwise_cues \
  --frame_sampling_mode uniform \
  --disable_global_logit_scale

run_exp "expC_middlelate_frames" \
  --prompt_set ravdess_8_facial_cues \
  --frame_sampling_mode middle_late \
  --disable_global_logit_scale

run_exp "expD_global_temp" \
  --prompt_set ravdess_8_facial_cues \
  --frame_sampling_mode uniform \
  --use_global_logit_scale

run_exp "expACD_combo" \
  --prompt_set ravdess_8_pairwise_cues \
  --frame_sampling_mode middle_late \
  --use_global_logit_scale

echo "[VARIANT] done -> $SUMMARY_JSONL"
