# Workflows

## 1. Local Setup

```bash
bash scripts/setup_local_links.sh
source scripts/bootstrap_env.sh
```

## 2. AIDE Baselines

Adapter sweep:

```bash
source scripts/bootstrap_env.sh
bash aide_clip/scripts/run_adapter_param_sweep.sh
```

Ablations:

```bash
source scripts/bootstrap_env.sh
bash aide_clip/scripts/run_ablations.sh
```

Transformer reproduction on AIDE:

```bash
source scripts/bootstrap_env.sh
bash aide_clip/scripts/run_aide_best_transformer_try.sh
```

## 3. YawDD Baseline

```bash
source scripts/bootstrap_env.sh
bash aide_clip/scripts/run_yawdd_baseline_autooutput.sh
```

## 4. PACER v2 Sweep Blocks

Run all blocks:

```bash
source scripts/bootstrap_env.sh
/home/yanjing/anaconda3/envs/mmtl/bin/python aide_clip/tools/run_pacer_phase2.py --blocks all
```

Run a subset:

```bash
source scripts/bootstrap_env.sh
/home/yanjing/anaconda3/envs/mmtl/bin/python aide_clip/tools/run_pacer_phase2.py --blocks block2 block3
```

Available blocks:

- `block2`
- `block3`
- `block4`
- `block5`
- `final_report`
- `all`

## 5. Practical Iteration Strategy

Use this order when modifying causal components:

1. fast single-run smoke checks on AIDE with one seed
2. compare against `benchmarks/adapter_sweep_summary.json`
3. only when AIDE looks promising, re-run YawDD and PACER block subsets
4. reserve full `block5` cross-dataset runs for candidates that already improved AIDE or clearly stabilized YawDD

## 6. Output Hygiene

Do not overwrite the copied benchmark snapshots.

Write new experiments under a fresh local-only folder, for example:

```bash
mkdir -p codex_runs/20260510_trial01
```

Store per-run outputs there or pass unique `RUN_NAME` and `RESULT_DIR` values to scripts.
