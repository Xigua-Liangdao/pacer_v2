# Dataset layout and protocols

This repository does not redistribute AIDE or YawDD. Download the datasets from their official sources and place them under `data/`, or provide paths through environment variables.

## AIDE

Expected layout:

```text
data/AIDE_Dataset/
  annotation/
    0001.json
    ...
  0001/
    incarframes/
      0.jpg
      1.jpg
      ...
  ...
```

Default environment:

```bash
export AIDE_ROOT=$PWD/data/AIDE_Dataset
export AIDE_ANNOTATION_ROOT=$AIDE_ROOT/annotation
```

Protocol used in the paper:

- task: driver emotion recognition;
- classes: Anxiety, Peace, Weariness, Happiness, Anger;
- split: stratified 65/15/20 train/validation/test;
- view: in-cabin RGB frames only;
- frame sampling: uniform sampling from each clip;
- main model frames: `T=5`;
- model selection: validation accuracy.

## YawDD

Expected layout:

```text
data/yawdd/
  Mirror/
    Female_mirror/
      1-FemaleNoGlasses-Normal.avi
      ...
    Male_mirror/
      ...
  Dash/
    ...
```

Default environment:

```bash
export YAWDD_ROOT=$PWD/data/yawdd
```

Protocol used in the paper:

- task: binary drowsiness detection;
- labels: `drowsy`, `notdrowsy`;
- unit of prediction: one label per video;
- split: speaker-independent train/validation/test split;
- frame sampling: `T=10` difference-guided frames;
- backbone: CLIP ViT-B/32;
- model selection: validation weighted F1.

The YawDD protocol is intentionally stricter than frame-level or speaker-overlapping setups. Results are therefore not directly comparable to prior YawDD papers that use a different split.

## CLIP weights

The scripts use `openai/clip-vit-base-patch32` through Hugging Face Transformers. If running offline, pre-download the model and set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache/hub
```

If running online, leave `CLIP_MODE=auto` in the reproduction scripts.
