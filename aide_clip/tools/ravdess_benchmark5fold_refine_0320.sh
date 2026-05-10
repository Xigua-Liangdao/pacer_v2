#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/yanjing/talk2bev/aide_clip"
PYTHON="/home/yanjing/anaconda3/envs/mmtl/bin/python"
SCRIPT="$ROOT/src/clip_ravdess_emotion_train.py"
PHASE1_DIR="$ROOT/results/ravdess/batch_0320_fold0_val1_v2"
OUT_DIR="$ROOT/results/ravdess/batch_0320_fold0_val1_refine"
SUMMARY_JSONL="$OUT_DIR/summary.jsonl"
mkdir -p "$OUT_DIR"
: > "$SUMMARY_JSONL"

while true; do
  if [[ -f "$PHASE1_DIR/summary.jsonl" ]] && [[ $(wc -l < "$PHASE1_DIR/summary.jsonl") -ge 6 ]]; then
    break
  fi
  sleep 60
done

BEST_JSON=$("$PYTHON" - <<'PY'
import json
from pathlib import Path
phase1_dir = Path('/data1/yanjing/talk2bev/aide_clip/results/ravdess/batch_0320_fold0_val1_v2')
best = None
for path in sorted(phase1_dir.glob('exp*.json')):
    obj = json.loads(path.read_text(encoding='utf-8'))
    score = obj['val']['weighted_f1']
    if best is None or score > best[0]:
        best = (score, path)
print(best[1])
PY
)

mapfile -t BEST_ARGS < <("$PYTHON" - <<PY
import json
from pathlib import Path
path = Path(${BEST_JSON@Q})
obj = json.loads(path.read_text(encoding='utf-8'))
config = obj['config']
args = [
    '--clip_mode', 'offline_only',
    '--split_mode', 'benchmark_5fold',
    '--benchmark_test_fold', str(config['benchmark_test_fold']),
    '--benchmark_val_fold', str(config['benchmark_val_fold']),
    '--allowed_modalities', ','.join(config['allowed_modalities']),
    '--allowed_vocal_channels', ','.join(config['allowed_vocal_channels']),
    '--allowed_intensities', ','.join(config['allowed_intensities']),
    '--video_extensions', ','.join(config['video_extensions']),
    '--adapter_hidden_dim', str(config['adapter_hidden_dim']),
    '--prompt_set', str(config['prompt_set']),
    '--loss_type', str(config['loss_type']),
    '--label_smoothing', str(config['label_smoothing']),
    '--epochs', str(config['epochs']),
    '--lr', str(config['lr']),
    '--weight_decay', str(config['weight_decay']),
    '--num_frames', str(config['num_frames']),
    '--run_zero_shot_eval',
    '--report_train_metrics',
]
if not config.get('use_prompt_weight', False):
    args.append('--disable_prompt_weight')
if not config.get('use_class_temperature', False):
    args.append('--disable_class_temperature')
if not config.get('use_class_bias', False):
    args.append('--disable_class_bias')
if not config.get('use_class_weight', True):
    args.append('--disable_class_weight')
if config.get('loss_type') == 'focal':
    args.extend(['--focal_gamma', str(config.get('focal_gamma', 2.0))])
for item in args:
    print(item)
PY
)

run_exp() {
  local name="$1"
  shift
  local output="$OUT_DIR/${name}.json"
  local ckpt="$OUT_DIR/${name}.ckpt.pt"
  local log="$OUT_DIR/${name}.log"

  echo "[REFINE] start $name"
  "$PYTHON" "$SCRIPT" \
    "${BEST_ARGS[@]}" \
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
    'val_accuracy': obj['val'].get('accuracy'),
    'val_weighted_f1': obj['val'].get('weighted_f1'),
    'test_accuracy': obj['test'].get('accuracy'),
    'test_weighted_f1': obj['test'].get('weighted_f1'),
    'num_frames': obj['config'].get('num_frames'),
    'epochs': obj['config'].get('epochs'),
    'lr': obj['config'].get('lr'),
    'weight_decay': obj['config'].get('weight_decay'),
    'label_smoothing': obj['config'].get('label_smoothing'),
}
print(json.dumps(row, ensure_ascii=False))
with open(${SUMMARY_JSONL@Q}, 'a', encoding='utf-8') as f:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
}

run_exp "exp07_best_numframes7" \
  --num_frames 7 \
  --epochs 80

run_exp "exp08_best_numframes7_lr1e4_wd1e4" \
  --num_frames 7 \
  --epochs 120 \
  --lr 1e-4 \
  --weight_decay 1e-4

run_exp "exp09_best_numframes9" \
  --num_frames 9 \
  --epochs 100 \
  --label_smoothing 0.0

echo "[REFINE] done -> $SUMMARY_JSONL"
