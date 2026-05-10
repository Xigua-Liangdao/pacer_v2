# pacer_v2 Workspace

This workspace is a clean experiment shell around the local AIDE and YawDD PACER-style CLIP pipelines.

It is designed for two simultaneous goals:

1. local end-to-end experimentation with the full datasets available on this machine
2. a lightweight GitHub repository that contains code, workflow, baselines, and an explicit Codex prompt without trying to version 60G+ of local data

## What Is Included

- `aide_clip/`
  - copied runnable code from the current local implementation
  - includes `src/`, `scripts/`, `tools/`, `docs/`, and `requirements.txt`
- `benchmarks/`
  - lightweight baseline snapshots copied from the current workspace
  - includes PACER v1 summary, PACER v2 final report, AIDE best summaries, and representative JSON outputs
- `vendor/fatigue-drive-yawning-detection/`
  - lightweight copy of the auxiliary YawDD repository code without the large extracted-face and dataset payloads
- `docs/`
  - workspace map and workflow notes
- `prompts/codex_main_prompt.md`
  - the main prompt to hand to Codex for iterative improvement work
- `scripts/setup_local_links.sh`
  - creates local symlinks to datasets and large result trees on this machine
- `scripts/bootstrap_env.sh`
  - exports the environment variables used by the main experiment scripts

## Important Constraint

The source machine currently has about 60G under `talk2bev/aide_clip/data` and about 115G in the broader source repository. Those assets are intentionally not committed here. This repository is meant to track code and lightweight experiment metadata only.

## Quick Start

Run these commands after entering this directory:

```bash
bash scripts/setup_local_links.sh
source scripts/bootstrap_env.sh
```

That creates the following local-only links:

- `aide_clip/data -> /data1/yanjing/talk2bev/aide_clip/data`
- `fatigue-drive-yawning-detection -> /data1/yanjing/talk2bev/fatigue-drive-yawning-detection`
- `external_data/AIDE_Dataset -> /data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset`
- `external_data/source_results -> /data1/yanjing/talk2bev/aide_clip/results`

## Default Runtime Environment

- primary experiment Python: `/home/yanjing/anaconda3/envs/mmtl/bin/python`
- fallback VS Code Python currently points at base conda, but the PACER sweep scripts already assume the `mmtl` interpreter
- offline flags should remain enabled on this machine:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`

## Current Reference Targets

Primary AIDE target:

- best current full AIDE model from adapter sweep:
  - accuracy `0.802744`
  - weighted_f1 `0.798728`
  - summary: `benchmarks/adapter_sweep_summary.json`

Secondary YawDD target:

- PACER v2 report currently recommends `D3_cda_only`
- summary: `benchmarks/causal_v2_final_report.md`

Legacy PACER reference:

- baseline and v1 ablations are summarized in `benchmarks/causal_v1_summary.md`

## Suggested First Runs

```bash
source scripts/bootstrap_env.sh
bash aide_clip/scripts/run_adapter_param_sweep.sh
bash aide_clip/scripts/run_ablations.sh
/home/yanjing/anaconda3/envs/mmtl/bin/python aide_clip/tools/run_pacer_phase2.py --blocks all
```

## Where To Start Reading

- `docs/workspace_map.md`
- `docs/workflows.md`
- `prompts/codex_main_prompt.md`
