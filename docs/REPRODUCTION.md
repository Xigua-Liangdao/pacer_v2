# Reproduction notes

The scripts in `scripts/reproduce/` are written as readable command records. They avoid machine-specific absolute paths and can be configured through environment variables.

Common variables:

```bash
export DEVICE=cuda:0
export MODEL_ID=openai/clip-vit-base-patch32
export CLIP_MODE=auto
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
```

For offline model loading:

```bash
export CLIP_MODE=offline_only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache/hub
```

Recommended order:

1. `pytest -q`
2. `bash scripts/reproduce/run_aide_main.sh`
3. `bash scripts/reproduce/run_yawdd_main.sh`
4. `bash scripts/reproduce/run_yawdd_table3_baselines.sh`
5. optional: `bash scripts/reproduce/run_yawdd_backbone_finetune.sh`

The full multi-seed run is compute-heavy. For a quick smoke run, set:

```bash
EPOCHS=1 MAX_SEQUENCES=64 bash scripts/reproduce/run_aide_main.sh
```

and similarly for YawDD:

```bash
EPOCHS=1 MAX_SEQUENCES=64 bash scripts/reproduce/run_yawdd_main.sh
```
