# Codex Main Prompt

You are working inside the `pacer_v2` research workspace on the same machine as the original AIDE and YawDD experiments.

## Mission

Continuously improve the current PACER-style causal CLIP pipeline.

Primary target:

- improve AIDE performance beyond the current strongest reference in `benchmarks/adapter_sweep_summary.json`
- the practical score to beat is the AIDE full-model result with accuracy `0.802744` and weighted_f1 `0.798728`

Secondary target:

- improve YawDD as much as possible without sacrificing AIDE progress
- use `benchmarks/causal_v2_final_report.md` and the copied PACER v1 snapshots as the YawDD reference set

## Environment You Must Use

- preferred Python interpreter: `/home/yanjing/anaconda3/envs/mmtl/bin/python`
- always keep these variables enabled unless a concrete reason requires otherwise:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- local AIDE data path: `/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset`
- local AIDE annotation path: `/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset/annotation`
- local YawDD extracted faces path: `/data1/yanjing/talk2bev/fatigue-drive-yawning-detection/extracted_face_multi4`

Before running experiments, assume this repo has already been prepared with:

```bash
bash scripts/setup_local_links.sh
source scripts/bootstrap_env.sh
```

## Code Areas That Matter Most

- `aide_clip/src/clip_cremad_emotion_train.py`
  - shared CCL, CFA, CDA, v2 counterfactual logic
- `aide_clip/src/clip_aide_emotion_train.py`
  - AIDE training loop, group construction, pooled vs transformer branch
- `aide_clip/src/clip_yawdd_emotion_train.py`
  - YawDD training loop and PACER-style switches
- `aide_clip/tools/run_pacer_phase2.py`
  - PACER v2 block sweeps and final report generation
- `aide_clip/src/qcpa.py`
  - QCPA classification head used by current AIDE variants

## Allowed Scope

You may:

- rewrite CCL, CFA, CDA implementations
- add new causal regularizers or replace the existing ones
- change scheduling, weighting, gating, or sampling strategies
- change pooled vs transformer behavior if justified by results
- add new scripts, manifests, and result summaries
- keep PACER legacy baselines intact while creating new experiment outputs elsewhere

You should not:

- delete benchmark snapshots in `benchmarks/`
- overwrite historical results unless explicitly asked
- move local datasets out of their current source locations

## Operating Rules

1. Start from the smallest experiment that can falsify your hypothesis.
2. Prefer AIDE first. Use YawDD as a secondary validation axis.
3. Keep a written experiment trail: config, hypothesis, output path, and result delta.
4. Do not claim a module is better unless it beats the benchmark with a clear margin or at least consistent repeatability.
5. If a full multi-module combination underperforms a simpler variant, treat that as signal rather than forcing the full model.

## Suggested Iteration Plan

1. Read `docs/workspace_map.md` and `docs/workflows.md`.
2. Reconfirm the benchmark targets from the files in `benchmarks/`.
3. Pick one causal component to modify first, usually in `clip_cremad_emotion_train.py`.
4. Run a fast AIDE check with one seed and a unique output location under `codex_runs/`.
5. If the result is promising, run the relevant YawDD baseline or PACER phase2 subset.
6. Summarize the delta against:
   - `benchmarks/adapter_sweep_summary.json`
   - `benchmarks/causal_v2_final_report.md`
7. Iterate.

## Concrete Baselines To Compare Against

- AIDE full adapter sweep best: `benchmarks/adapter_sweep_summary.json`
- AIDE transformer reproduction snapshot: `benchmarks/clip_emotion_strict_repro_c_transformer_try_v2.json`
- AIDE pooled causal snapshot: `benchmarks/AIDE_qcpa_full_20260504_020149.json`
- YawDD PACER v1 baseline: `benchmarks/A0_pacer_baseline.json`
- YawDD PACER v1 strong subset model: `benchmarks/A3_cda_only.json`
- YawDD PACER v1 full model: `benchmarks/B4_all_three.json`
- YawDD PACER v2 report: `benchmarks/causal_v2_final_report.md`

## Reporting Format

For every meaningful change, report:

- hypothesis
- files changed
- exact command run
- output path
- AIDE delta
- YawDD delta if evaluated
- whether to keep, revert, or continue iterating

## Preferred Output Layout For New Runs

Use unique folders under `codex_runs/`, for example:

```bash
codex_runs/20260510_cda_rewrite_v1/
codex_runs/20260510_ccl_margin_v2/
```

Within each folder, keep:

- `notes.md`
- `commands.sh`
- copied JSON summaries or symlinks to produced outputs

Your job is not to preserve the current causal components. Your job is to beat the current AIDE benchmark and then push YawDD upward without breaking reproducibility.
