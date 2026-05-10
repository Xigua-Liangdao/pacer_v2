# RAVDESS 61.2 Baseline

This file records the exact locked baseline corresponding to:

- test weighted_f1 = 0.612489
- test accuracy = 0.613333

Canonical result artifact:

- results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.json

## Environment

- Python env: `/home/yanjing/anaconda3/envs/mmtl/bin/python`
- Working directory: `/data1/yanjing/talk2bev/aide_clip`
- Recommended offline flags for reproducibility on this machine:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`

## Exact Baseline Settings

### Task and data

- dataset: `RAVDESS`
- task: `emotion`
- split_mode: `benchmark_5fold`
- benchmark_test_fold: `0`
- benchmark_val_fold: `1`
- train actors: `1,4,8,9,10,11,12,17,19,20,21,22,23,24`
- val actors: `3,6,7,13,18`
- test actors: `2,5,14,15,16`
- allowed_modalities: `02`
- allowed_vocal_channels: `01`
- allowed_intensities: `01,02`
- video_extensions: `.mp4`
- total samples: `1440`
- train samples: `840`
- val samples: `300`
- test samples: `300`

### Backbone and prompts

- model_id: `openai/clip-vit-base-patch32`
- clip_mode: `offline_only` recommended on this machine
- prompt_template: `The person looks <LABEL>.`
- prompt_set: `ravdess_8_facial_cues`
- prompts_per_class: `7`
- total_text_prompts: `56`

### Optimization

- epochs: `80`
- extract_batch_size: `32`
- train_batch_size: `32`
- batch_size: `32`
- lr: `1.5e-4`
- weight_decay: `5e-4`
- max_grad_norm: `1.0`
- optimizer: `AdamW`
- lr_scheduler: `none`
- disable_scheduler: `false`
- early_stopping_patience: `0`
- early_stopping_min_delta: `0.0`
- select_metric: `weighted_f1`
- seed: `45`

### Feature extraction and temporal settings

- num_frames: `5`
- frame_sampling_mode: `middle_late`
- sampling_window_start: `0.4`
- sampling_window_end: `0.9`
- diff_alpha: `0.6`
- diff_beta: `0.4`
- min_gap_ratio: `0.08`
- score_smooth_window: `3`
- frame_diff_metric: `gray_l1`
- feature_layout: `sequence`
- temporal_head: `transformer`
- temporal_num_heads: `4`
- temporal_num_layers: `1`
- temporal_pooling: `mean`

### Adapter and loss

- strict_frozen_clip: `true`
- unfreeze_last_visual_block: `false`
- adapter_hidden_dim: `256`
- adapter_dropout: `0.2`
- loss_type: `focal`
- focal_gamma: `1.5`
- label_smoothing: `0.0`
- use_class_weight: `false`
- use_prompt_weight: `false`
- use_global_logit_scale: `false`
- use_class_temperature: `false`
- use_class_bias: `false`
- use_amp: `false`
- use_test_ensemble: `true`
- ensemble_group_size: `2`

### Cache and outputs

- feature_cache_dir: `/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features`
- resolved train cache:
  `/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features/ravdess_train_openai_clip-vit-base-patch32_f5_middle_late_a9ce3494_n840_3f76c24be37e054a_sequence.pt`
- resolved val cache:
  `/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features/ravdess_val_openai_clip-vit-base-patch32_f5_middle_late_a9ce3494_n300_a046aa41eef89309_sequence.pt`
- resolved test cache:
  `/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features/ravdess_test_openai_clip-vit-base-patch32_f5_middle_late_a9ce3494_n300_64a031450f5b0f49_sequence.pt`
- checkpoint_output:
  `results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.ckpt.pt`
- log_file:
  `results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.log`
- output:
  `results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.json`

## Terminal Launch Command

Run this from `/data1/yanjing/talk2bev/aide_clip`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/yanjing/anaconda3/envs/mmtl/bin/python src/clip_ravdess_emotion_train.py \
  --clip_mode offline_only \
  --experiment_name custom \
  --ravdess_root /data1/yanjing/talk2bev/aide_clip/data/RAVDESS \
  --split_mode benchmark_5fold \
  --benchmark_test_fold 0 \
  --benchmark_val_fold 1 \
  --model_id openai/clip-vit-base-patch32 \
  --prompt_set ravdess_8_facial_cues \
  --epochs 80 \
  --extract_batch_size 32 \
  --train_batch_size 32 \
  --lr 1.5e-4 \
  --weight_decay 5e-4 \
  --max_grad_norm 1.0 \
  --num_frames 5 \
  --frame_sampling_mode middle_late \
  --sampling_window_start 0.4 \
  --sampling_window_end 0.9 \
  --diff_alpha 0.6 \
  --diff_beta 0.4 \
  --min_gap_ratio 0.08 \
  --score_smooth_window 3 \
  --frame_diff_metric gray_l1 \
  --feature_layout sequence \
  --adapter_hidden_dim 256 \
  --adapter_dropout 0.2 \
  --temporal_head transformer \
  --temporal_num_heads 4 \
  --temporal_num_layers 1 \
  --temporal_pooling mean \
  --loss_type focal \
  --focal_gamma 1.5 \
  --label_smoothing 0.0 \
  --select_metric weighted_f1 \
  --use_test_ensemble \
  --ensemble_group_size 2 \
  --strict_frozen_clip \
  --disable_prompt_weight \
  --disable_class_temperature \
  --disable_class_bias \
  --disable_class_weight \
  --disable_amp \
  --lr_scheduler none \
  --early_stopping_patience 0 \
  --early_stopping_min_delta 0.0 \
  --run_zero_shot_eval \
  --report_train_metrics \
  --seed 45 \
  --allowed_modalities 02 \
  --allowed_vocal_channels 01 \
  --allowed_intensities 01,02 \
  --feature_cache_dir /data1/yanjing/talk2bev/aide_clip/cache/ravdess_features \
  --checkpoint_output results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.ckpt.pt \
  --log_file results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.log \
  --output results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.json
```

## One-line Command

```bash
cd /data1/yanjing/talk2bev/aide_clip && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/yanjing/anaconda3/envs/mmtl/bin/python src/clip_ravdess_emotion_train.py --clip_mode offline_only --experiment_name custom --ravdess_root /data1/yanjing/talk2bev/aide_clip/data/RAVDESS --split_mode benchmark_5fold --benchmark_test_fold 0 --benchmark_val_fold 1 --model_id openai/clip-vit-base-patch32 --prompt_set ravdess_8_facial_cues --epochs 80 --extract_batch_size 32 --train_batch_size 32 --lr 1.5e-4 --weight_decay 5e-4 --max_grad_norm 1.0 --num_frames 5 --frame_sampling_mode middle_late --sampling_window_start 0.4 --sampling_window_end 0.9 --diff_alpha 0.6 --diff_beta 0.4 --min_gap_ratio 0.08 --score_smooth_window 3 --frame_diff_metric gray_l1 --feature_layout sequence --adapter_hidden_dim 256 --adapter_dropout 0.2 --temporal_head transformer --temporal_num_heads 4 --temporal_num_layers 1 --temporal_pooling mean --loss_type focal --focal_gamma 1.5 --label_smoothing 0.0 --select_metric weighted_f1 --use_test_ensemble --ensemble_group_size 2 --strict_frozen_clip --disable_prompt_weight --disable_class_temperature --disable_class_bias --disable_class_weight --disable_amp --lr_scheduler none --early_stopping_patience 0 --early_stopping_min_delta 0.0 --run_zero_shot_eval --report_train_metrics --seed 45 --allowed_modalities 02 --allowed_vocal_channels 01 --allowed_intensities 01,02 --feature_cache_dir /data1/yanjing/talk2bev/aide_clip/cache/ravdess_features --checkpoint_output results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.ckpt.pt --log_file results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.log --output results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.json
```

## Auto-output Script

Script path:

- scripts/run_ravdess_baseline61_autooutput.sh

This script keeps the locked baseline flags but automatically creates unique file names for:

- `--output`
- `--checkpoint_output`
- `--log_file`

Usage:

```bash
cd /data1/yanjing/talk2bev/aide_clip
bash scripts/run_ravdess_baseline61_autooutput.sh
```

Custom tag:

```bash
cd /data1/yanjing/talk2bev/aide_clip
bash scripts/run_ravdess_baseline61_autooutput.sh lr1e4_try1
```

Custom tag plus extra override flags:

```bash
cd /data1/yanjing/talk2bev/aide_clip
bash scripts/run_ravdess_baseline61_autooutput.sh lr1e4_try1 --lr 1e-4 --seed 46
```

The generated files will look like:

- `results/ravdess/ravdess_baseline61_20260321_230000.json`
- `results/ravdess/ravdess_baseline61_20260321_230000.pt`
- `results/ravdess/ravdess_baseline61_20260321_230000.log`

## Notes for Manual Fine-tuning

- Keep this file as the source of truth for the locked baseline.
- When doing one-factor tuning, copy the command above and change only the target flags.
- If you want a new run without overwriting the baseline artifact, change these three arguments together:
  - `--output`
  - `--checkpoint_output`
  - `--log_file`