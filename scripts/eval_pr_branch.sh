#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BRANCH_NAME=""
PR_NUMBER=""
EXPERIMENT="aide"
CONDA_ENV="mmtl"
BASELINE_PATH=""
METRIC="weighted_f1"
MODE="greater"
GPU_ID="0"
SEED=""
DATA_ROOT=""
COMMENT_RESULT=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_pr_branch.sh --branch <branch_name> [options]
  bash scripts/eval_pr_branch.sh --pr <number> [options]

Options:
  --branch <branch_name>      Git branch name to fetch from origin.
  --pr <number>               GitHub PR number to fetch from origin.
  --experiment <name>         Experiment template to run: aide or yawdd.
  --conda-env <name>          Conda environment name. Default: mmtl.
  --baseline <path>           Baseline raw-result JSON or normalized metrics.json.
  --metric <name>             Metric name to compare. Default: weighted_f1.
  --mode <name>               Comparison mode: greater, greater_equal, less, less_equal.
  --gpu <id>                  GPU id to expose as CUDA_VISIBLE_DEVICES. Default: 0.
  --seed <int>                Optional seed override.
  --data-root <path>          Dataset root consumed by the experiment template.
  --comment                   If --pr is given, try to post compare_report.md back to the PR.
  -h, --help                  Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      BRANCH_NAME="$2"
      shift 2
      ;;
    --pr)
      PR_NUMBER="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --baseline)
      BASELINE_PATH="$2"
      shift 2
      ;;
    --metric)
      METRIC="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --comment)
      COMMENT_RESULT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$BRANCH_NAME" && -n "$PR_NUMBER" ]]; then
  echo "[ERROR] use either --branch or --pr, not both" >&2
  exit 2
fi

if [[ -z "$BRANCH_NAME" && -z "$PR_NUMBER" ]]; then
  echo "[ERROR] one of --branch or --pr is required" >&2
  exit 2
fi

if [[ -z "$BASELINE_PATH" ]]; then
  echo "[ERROR] --baseline is required so pass/fail can be decided" >&2
  exit 2
fi

if [[ "$EXPERIMENT" != "aide" && "$EXPERIMENT" != "yawdd" ]]; then
  echo "[ERROR] unsupported experiment: $EXPERIMENT" >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git is required" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is required to run the server evaluation template" >&2
  exit 2
fi

if ! git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERROR] script must be run inside a git repository" >&2
  exit 2
fi

BASELINE_PATH="$(python3 - <<'PY' "$BASELINE_PATH"
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
)"

if [[ ! -f "$BASELINE_PATH" ]]; then
  echo "[ERROR] baseline file not found: $BASELINE_PATH" >&2
  exit 2
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TARGET_LABEL=""
FETCH_SPEC=""
if [[ -n "$PR_NUMBER" ]]; then
  TARGET_LABEL="pr_${PR_NUMBER}"
  FETCH_SPEC="pull/${PR_NUMBER}/head"
else
  TARGET_LABEL="${BRANCH_NAME}"
  FETCH_SPEC="refs/heads/${BRANCH_NAME}"
fi

SAFE_TARGET="$(printf '%s' "$TARGET_LABEL" | tr '/ ' '__')"
RUN_DIR="$ROOT/results/server_eval/$SAFE_TARGET/$TIMESTAMP"
mkdir -p "$RUN_DIR"

RUN_LOG="$RUN_DIR/run.log"
BASELINE_METRICS="$RUN_DIR/baseline_metrics.json"
CANDIDATE_METRICS="$RUN_DIR/metrics.json"
COMPARE_REPORT="$RUN_DIR/compare_report.md"
RAW_RESULT="$RUN_DIR/candidate_raw.json"
WORKTREE_BASE="$ROOT/.server_eval_worktrees"
WORKTREE_DIR="$WORKTREE_BASE/${SAFE_TARGET}_${TIMESTAMP}"
COMMAND_TEMPLATE="$ROOT/experiments/server_eval/$EXPERIMENT/commands.sh"

if [[ ! -f "$COMMAND_TEMPLATE" ]]; then
  echo "[ERROR] experiment template not found: $COMMAND_TEMPLATE" >&2
  exit 2
fi

cleanup() {
  if [[ -d "$WORKTREE_DIR" ]]; then
    git -C "$ROOT" worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || rm -rf "$WORKTREE_DIR"
  fi
}
trap cleanup EXIT

mkdir -p "$WORKTREE_BASE"

echo "[INFO] repo_root=$ROOT" | tee "$RUN_LOG"
echo "[INFO] run_dir=$RUN_DIR" | tee -a "$RUN_LOG"
echo "[INFO] target=$TARGET_LABEL" | tee -a "$RUN_LOG"
echo "[INFO] experiment=$EXPERIMENT" | tee -a "$RUN_LOG"
echo "[INFO] baseline=$BASELINE_PATH" | tee -a "$RUN_LOG"
echo "[INFO] metric=$METRIC mode=$MODE" | tee -a "$RUN_LOG"

echo "[STEP] fetching origin/$FETCH_SPEC" | tee -a "$RUN_LOG"
git -C "$ROOT" fetch origin "$FETCH_SPEC" >>"$RUN_LOG" 2>&1

echo "[STEP] creating clean worktree $WORKTREE_DIR" | tee -a "$RUN_LOG"
git -C "$ROOT" worktree add --detach "$WORKTREE_DIR" FETCH_HEAD >>"$RUN_LOG" 2>&1

echo "[STEP] normalizing baseline metrics" | tee -a "$RUN_LOG"
python3 "$WORKTREE_DIR/tools/collect_metrics.py" --input "$BASELINE_PATH" --output "$BASELINE_METRICS" >>"$RUN_LOG" 2>&1

echo "[STEP] running experiment template" | tee -a "$RUN_LOG"
export PROJECT_ROOT="$WORKTREE_DIR"
export RUN_DIR
export RUN_LOG
export CONDA_ENV
export GPU_ID
export DATA_ROOT
export OUTPUT_JSON="$RAW_RESULT"
if [[ -n "$SEED" ]]; then
  export SEED
fi

conda run --no-capture-output -n "$CONDA_ENV" bash "$WORKTREE_DIR/experiments/server_eval/$EXPERIMENT/commands.sh" >>"$RUN_LOG" 2>&1

if [[ ! -f "$RAW_RESULT" ]]; then
  echo "[ERROR] candidate raw result was not produced: $RAW_RESULT" | tee -a "$RUN_LOG"
  exit 3
fi

echo "[STEP] normalizing candidate metrics" | tee -a "$RUN_LOG"
python3 "$WORKTREE_DIR/tools/collect_metrics.py" --input "$RAW_RESULT" --output "$CANDIDATE_METRICS" >>"$RUN_LOG" 2>&1

echo "[STEP] comparing candidate against baseline" | tee -a "$RUN_LOG"
set +e
python3 "$WORKTREE_DIR/tools/compare_metrics.py" \
  --baseline "$BASELINE_METRICS" \
  --candidate "$CANDIDATE_METRICS" \
  --metric "$METRIC" \
  --mode "$MODE" \
  --output "$COMPARE_REPORT" >>"$RUN_LOG" 2>&1
COMPARE_EXIT=$?
set -e

if [[ $COMMENT_RESULT -eq 1 && -n "$PR_NUMBER" ]]; then
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    echo "[STEP] posting compare report back to PR #$PR_NUMBER" | tee -a "$RUN_LOG"
    if ! gh pr comment "$PR_NUMBER" --body-file "$COMPARE_REPORT" >>"$RUN_LOG" 2>&1; then
      echo "[WARN] failed to comment on PR #$PR_NUMBER; report remains at $COMPARE_REPORT" | tee -a "$RUN_LOG"
    fi
  else
    echo "[WARN] gh CLI unavailable or unauthenticated; report remains at $COMPARE_REPORT" | tee -a "$RUN_LOG"
  fi
fi

echo "[DONE] raw result: $RAW_RESULT" | tee -a "$RUN_LOG"
echo "[DONE] normalized metrics: $CANDIDATE_METRICS" | tee -a "$RUN_LOG"
echo "[DONE] compare report: $COMPARE_REPORT" | tee -a "$RUN_LOG"

exit "$COMPARE_EXIT"