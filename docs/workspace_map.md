# Workspace Map

## Core Training Code

- `aide_clip/src/clip_aide_emotion_train.py`
  - main AIDE CLIP emotion training entrypoint
- `aide_clip/src/clip_yawdd_emotion_train.py`
  - main YawDD CLIP drowsiness training entrypoint
- `aide_clip/src/clip_cremad_emotion_train.py`
  - shared implementation for CCL, CFA, CDA, v2 counterfactual utilities, and adapter logit helpers
- `aide_clip/src/qcpa.py`
  - QCPA head used by current AIDE pipeline variants

## Experiment Drivers

- `aide_clip/scripts/run_adapter_param_sweep.sh`
  - strongest current AIDE full-model reference
- `aide_clip/scripts/run_ablations.sh`
  - component ablation on AIDE
- `aide_clip/scripts/run_aide_best_transformer_try.sh`
  - AIDE transformer-branch reproduction run
- `aide_clip/scripts/run_yawdd_baseline_autooutput.sh`
  - YawDD baseline driver
- `aide_clip/tools/run_pacer_phase2.py`
  - PACER v2 block sweep driver for YawDD and AIDE cross-dataset analysis

## Auxiliary Source

- `vendor/fatigue-drive-yawning-detection/`
  - lightweight copy of the original YawDD-side utility repo without heavy datasets or model weights

## Lightweight Baseline Snapshots In This Repo

- `benchmarks/causal_v1_summary.md`
- `benchmarks/A0_pacer_baseline.json`
- `benchmarks/A3_cda_only.json`
- `benchmarks/B4_all_three.json`
- `benchmarks/causal_v2_final_report.md`
- `benchmarks/adapter_sweep_summary.json`
- `benchmarks/ablation_summary.json`
- `benchmarks/clip_emotion_strict_repro_c_transformer_try_v2.json`
- `benchmarks/AIDE_qcpa_full_20260504_020149.json`

## Local-Only Assets After Running setup_local_links.sh

- `aide_clip/data`
- `fatigue-drive-yawning-detection`
- `external_data/AIDE_Dataset`
- `external_data/source_results`

## Main Objective

Treat AIDE as the primary score target and YawDD as the secondary target.

- beat the current best AIDE adapter-sweep model if possible
- improve or at least retain strong YawDD behavior
- keep legacy PACER baselines reproducible for comparison
