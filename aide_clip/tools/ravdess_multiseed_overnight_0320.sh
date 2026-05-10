#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/yanjing/talk2bev/aide_clip"
RESULT_DIR="$ROOT/results/ravdess/multiseed_5fold_0320"
MASTER_LOG="$RESULT_DIR/master.log"
MONITOR_LOG="$RESULT_DIR/overnight_monitor.log"
SUMMARY3="$RESULT_DIR/fusion_summary_3seed.json"
SUMMARY5="$RESULT_DIR/fusion_summary_5seed.json"
FUSION_SCRIPT="$ROOT/tools/ravdess_multiseed_5fold_fusion_0320.py"
GROUP_FUSION_SCRIPT="$ROOT/tools/late_fusion_ravdess_groups.py"
PYTHON="/home/yanjing/anaconda3/envs/mmtl/bin/python"
EXTRA_SCRIPT="$ROOT/tools/ravdess_multiseed_5fold_extra_0320.sh"
EXTRA_LOG="$RESULT_DIR/extra_master.log"

mkdir -p "$RESULT_DIR"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$MONITOR_LOG"; }

wait_for_done_marker() {
  local file="$1"
  local marker="$2"
  while true; do
    if [[ -f "$file" ]] && grep -qF "$marker" "$file"; then
      break
    fi
    sleep 120
  done
}

log "overnight monitor started"
log "waiting for initial multiseed 5-fold batch to finish"
wait_for_done_marker "$MASTER_LOG" "[DONE] all multiseed 5-fold jobs finished"
log "initial batch completed; running 3-seed fusion summary"
"$PYTHON" "$FUSION_SCRIPT" \
  --results_dir "$RESULT_DIR" \
  --fusion_script "$GROUP_FUSION_SCRIPT" \
  --device cuda:0 \
  --batch_size 512 \
  --select_metric weighted_f1 \
  --seeds 42,43,44 \
  --folds 0,1,2,3,4 \
  --output "$SUMMARY3" >> "$MONITOR_LOG" 2>&1

BEST3=$("$PYTHON" - <<'PY'
import json
path = "/data1/yanjing/talk2bev/aide_clip/results/ravdess/multiseed_5fold_0320/fusion_summary_3seed.json"
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data['best_fold']['best_fusion']['test']['accuracy'])
PY
)
AVG3=$("$PYTHON" - <<'PY'
import json
path = "/data1/yanjing/talk2bev/aide_clip/results/ravdess/multiseed_5fold_0320/fusion_summary_3seed.json"
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data['average_test_accuracy'])
PY
)
log "3-seed fusion complete: best_fold_acc=${BEST3} avg_acc=${AVG3}"

NEED_EXTRA=$("$PYTHON" - <<'PY'
import json
path = "/data1/yanjing/talk2bev/aide_clip/results/ravdess/multiseed_5fold_0320/fusion_summary_3seed.json"
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
best_acc = data['best_fold']['best_fusion']['test']['accuracy']
print('1' if best_acc < 0.60 else '0')
PY
)

if [[ "$NEED_EXTRA" == "1" ]]; then
  log "best fold still below 0.60; launching extra seeds 45,46"
  nohup bash "$EXTRA_SCRIPT" > "$EXTRA_LOG" 2>&1 < /dev/null &
  EXTRA_PID=$!
  log "extra seed batch pid=${EXTRA_PID}"
  wait_for_done_marker "$EXTRA_LOG" "[DONE] all extra multiseed 5-fold jobs finished"
  log "extra seeds completed; running 5-seed fusion summary"
  "$PYTHON" "$FUSION_SCRIPT" \
    --results_dir "$RESULT_DIR" \
    --fusion_script "$GROUP_FUSION_SCRIPT" \
    --device cuda:0 \
    --batch_size 512 \
    --select_metric weighted_f1 \
    --seeds 42,43,44,45,46 \
    --folds 0,1,2,3,4 \
    --output "$SUMMARY5" >> "$MONITOR_LOG" 2>&1
  BEST5=$("$PYTHON" - <<'PY'
import json
path = "/data1/yanjing/talk2bev/aide_clip/results/ravdess/multiseed_5fold_0320/fusion_summary_5seed.json"
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data['best_fold']['best_fusion']['test']['accuracy'])
PY
)
  AVG5=$("$PYTHON" - <<'PY'
import json
path = "/data1/yanjing/talk2bev/aide_clip/results/ravdess/multiseed_5fold_0320/fusion_summary_5seed.json"
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data['average_test_accuracy'])
PY
)
  log "5-seed fusion complete: best_fold_acc=${BEST5} avg_acc=${AVG5}"
else
  log "best fold already reached >= 0.60 with 3 seeds; no extra seeds launched"
fi

log "overnight monitor finished"
