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

- task: five-way driver affective-state recognition;
- classes: Anxiety, Peace, Weariness, Happiness, Anger;
- split: stratified 65/15/20 train/validation/test video-level protocol;
- view: in-cabin RGB frames only;
- frame sampling: `T=3` uniformly sampled frames;
- backbone: CLIP ViT-B/32;
- CLIP image/text encoders: frozen in the main model;
- prompt templates: `P=7`;
- adapter hidden size: 1024;
- dropout: 0.2;
- loss: cross entropy;
- label smoothing: 0.01;
- learning rate: 1.5e-4;
- weight decay: 5e-4;
- batch size: 32;
- epochs: 40;
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

- task: binary yawning-based drowsiness-cue detection;
- public-facing labels: non-yawning, yawning cue;
- internal code mapping: `drowsy` = yawning-cue positive class; `notdrowsy` = non-yawning class;
- unit of prediction: one label per video;
- split: driver-disjoint train/validation/test split;
- drivers: 34/6/7 train/validation/test;
- videos: 208/49/62 train/validation/test;
- test support: 41 non-yawning videos and 21 yawning-cue videos;
- frame sampling: `T=10` difference-guided frames;
- backbone: CLIP ViT-B/32;
- CLIP image/text encoders: frozen in the main model;
- prompt templates: `P=5` `yawdd_facial_cues` templates per class;
- adapter hidden size: 512;
- dropout: 0.3;
- loss: focal loss;
- focal gamma: 2.0;
- label smoothing: 0.01;
- learning rate: 1.0e-4;
- weight decay: 1e-2;
- batch size: 16;
- epochs: 40;
- model selection: validation weighted F1.

YawDD is treated as a yawning-based drowsiness-related vigilance-cue benchmark, not as a full physiological drowsiness-label dataset.

The YawDD setting is not directly comparable to frame-level or driver-overlapping YawDD protocols, and it should be interpreted as yawning-cue recognition rather than comprehensive drowsiness-state estimation.

## CLIP weights

The scripts use `openai/clip-vit-base-patch32` through Hugging Face Transformers. If running offline, pre-download the model and set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache/hub
```

If running online, leave `CLIP_MODE=auto` in the reproduction scripts.
