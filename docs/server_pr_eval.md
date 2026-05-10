# Server PR Evaluation Workflow

## What This Solves

This workflow separates code iteration from expensive evaluation.

- Codex, Copilot, or any cloud coding agent can open a branch or pull request.
- The real evaluation happens on your GPU server, where the dataset, conda environment, cache, and existing experiment layout already exist.
- The server fetches the PR or branch into a clean git worktree, runs a reproducible command template, normalizes metrics, compares against a baseline, and can optionally comment the result back to the GitHub PR.

This avoids running full experiments on the local machine while keeping the evaluation loop reviewable and reproducible.

## Files Added For This Workflow

- `scripts/eval_pr_branch.sh`
  - fetches a PR or branch into a clean worktree and runs the evaluation loop
- `tools/collect_metrics.py`
  - normalizes raw result JSON into a stable `metrics.json`
- `tools/compare_metrics.py`
  - compares baseline and candidate metrics and writes `compare_report.md`
- `experiments/server_eval/aide/commands.sh`
  - server-side AIDE command template
- `experiments/server_eval/yawdd/commands.sh`
  - server-side YawDD command template

## How Codex Or Copilot Should Work With It

1. Create a branch or PR that only changes code and configuration.
2. Do not run full training locally.
3. Push the branch or open a PR.
4. On the server, run `scripts/eval_pr_branch.sh` against that branch or PR.
5. Inspect `results/server_eval/...` for the raw result, normalized metrics, and comparison report.
6. If `--comment` is enabled and `gh` is authenticated, the report can be posted back to the PR automatically.

## Existing Experiment Entry Points In This Repo

AIDE:

- `aide_clip/src/clip_aide_emotion_train.py`
- `aide_clip/scripts/run_adapter_param_sweep.sh`
- `aide_clip/scripts/run_ablations.sh`

YawDD:

- `aide_clip/src/clip_yawdd_emotion_train.py`
- `aide_clip/scripts/run_yawdd_baseline_autooutput.sh`
- YawDD benchmark snapshots also exist under `benchmarks/`

Current AIDE raw results normally contain top-level `val` and `test` sections with fields such as `accuracy` and `weighted_f1`.

Current YawDD raw results may contain:

- `test.weighted_f1`
- `test.accuracy`
- `test_metrics.f1`
- `aggregate.*.mean`
- `zero_shot`

`tools/collect_metrics.py` is tolerant to those differences and derives `macro_f1`, `uar`, and `war` from confusion matrices when possible.

## Manual Server Usage

Evaluate a PR:

```bash
bash scripts/eval_pr_branch.sh \
  --pr 123 \
  --experiment aide \
  --conda-env mmtl \
  --data-root /path/to/AIDE_Dataset \
  --baseline benchmarks/AIDE_qcpa_full_20260504_020149.json \
  --metric weighted_f1 \
  --gpu 0
```

Evaluate a branch:

```bash
bash scripts/eval_pr_branch.sh \
  --branch feature/better-cda \
  --experiment yawdd \
  --conda-env mmtl \
  --data-root /path/to/extracted_face_multi4 \
  --baseline benchmarks/A0_pacer_baseline.json \
  --metric weighted_f1 \
  --gpu 0
```

Evaluate a PR and post the result back to GitHub:

```bash
bash scripts/eval_pr_branch.sh \
  --pr 123 \
  --experiment aide \
  --conda-env mmtl \
  --data-root /path/to/AIDE_Dataset \
  --baseline benchmarks/AIDE_qcpa_full_20260504_020149.json \
  --metric weighted_f1 \
  --gpu 0 \
  --comment
```

## Expected Output Layout

Each run writes to a timestamped folder:

- `results/server_eval/<branch_or_pr>/<timestamp>/run.log`
- `results/server_eval/<branch_or_pr>/<timestamp>/candidate_raw.json`
- `results/server_eval/<branch_or_pr>/<timestamp>/metrics.json`
- `results/server_eval/<branch_or_pr>/<timestamp>/baseline_metrics.json`
- `results/server_eval/<branch_or_pr>/<timestamp>/compare_report.md`

The script exits with code `0` only when the comparison passes.

## How Pass Or Fail Is Decided

`tools/compare_metrics.py` compares:

- baseline value
- candidate value
- absolute delta
- relative delta when baseline is nonzero

Default rule:

- metric: `weighted_f1`
- mode: `greater`

That means the candidate must have strictly larger `weighted_f1` than the baseline.

You can change the rule with `--metric` and `--mode`.

## Debugging Common Failures

GitHub fetch failure:

- verify the server can access `origin`
- verify the branch exists or the PR number is valid
- verify the checked-out repo remote still points to the intended GitHub repository

Conda environment missing:

- verify `conda` is on `PATH`
- verify the requested env name exists
- run `conda env list` on the server

Dataset path missing:

- pass `--data-root <path>` explicitly
- for AIDE, point to the dataset root that contains `annotation/`
- for YawDD, point to the extracted face sequence root used by your server setup

Metrics file missing:

- inspect `run.log`
- verify the experiment template produced `candidate_raw.json`
- verify the training command did not crash before writing output

Candidate below baseline:

- this is an expected failure mode
- read `compare_report.md` and `metrics.json`
- the script exits nonzero so CI or cron wrappers can detect the regression

`gh` CLI not authenticated:

- run `gh auth status`
- if authentication is unavailable, the evaluation still completes and prints the report path instead of failing the whole run

## Safety Constraints

- do not run full experiments on the local machine
- do not download datasets in this workflow
- do not delete or overwrite historical result files
- do not commit credentials, tokens, server IPs, or private absolute paths into the repository
- keep experiment-specific paths configurable through CLI arguments or environment variables