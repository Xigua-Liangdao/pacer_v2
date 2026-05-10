# RAVDESS Best Locked Config

## Best Result

- Best test weighted_f1 found in the current repository: `0.612489`
- Best test accuracy: `0.613333`
- Canonical result files:
  - `results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.json`
  - `results/ravdess/clip_ravdess_benchmark_fold0_f5_middlelate_temporal_transformer_l1_facialcfg_meanpool_v1.json`

Note: I scanned the current `results/ravdess` directory and did not find a RAVDESS run with test weighted_f1 `0.615xxx`. The highest saved result currently present is the config below.

## Core Config

| Key | Value |
| --- | --- |
| dataset | `RAVDESS` |
| task | `emotion` |
| execution_mode | `strict_frozen_clip_adapter` |
| model_id | `openai/clip-vit-base-patch32` |
| prompt_template | `The person looks <LABEL>.` |
| prompt_set | `ravdess_8_facial_cues` |
| split_strategy | `benchmark_5fold` |
| benchmark_test_fold | `0` |
| benchmark_val_fold | `1` |
| epochs | `80` |
| train_batch_size | `32` |
| extract_batch_size | `32` |
| lr | `1.5e-4` |
| weight_decay | `5e-4` |
| max_grad_norm | `1.0` |
| loss_type | `focal` |
| focal_gamma | `1.5` |
| label_smoothing | `0.0` |
| use_class_weight | `false` |
| select_metric | `weighted_f1` |
| use_test_ensemble | `true` |
| ensemble_group_size | `2` |
| use_amp | `false` |
| lr_scheduler | `none` |
| early_stopping_patience | `0` |
| seed | `45` |

## Data Filtering

| Key | Value |
| --- | --- |
| ravdess_root | `/data1/yanjing/talk2bev/aide_clip/data/RAVDESS` |
| benchmark_video_list | `/data1/yanjing/talk2bev/MMEmotionRecognition-main/data/ravdess_videos.csv` |
| allowed_modalities | `02` |
| allowed_vocal_channels | `01` |
| allowed_intensities | `01,02` |
| video_extensions | `.mp4` |

## Frame / Feature Config

| Key | Value |
| --- | --- |
| num_frames | `5` |
| frame_sampling_mode | `middle_late` |
| sampling_window_start | `0.4` |
| sampling_window_end | `0.9` |
| diff_alpha | `0.6` |
| diff_beta | `0.4` |
| min_gap_ratio | `0.08` |
| score_smooth_window | `3` |
| frame_diff_metric | `gray_l1` |
| feature_layout | `sequence` |
| feature_cache_dir | `/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features` |

Resolved cache files recorded in the canonical result:

- train: `/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features/ravdess_train_openai_clip-vit-base-patch32_f5_middle_late_a9ce3494_n840_3f76c24be37e054a_sequence.pt`
- val: `/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features/ravdess_val_openai_clip-vit-base-patch32_f5_middle_late_a9ce3494_n300_a046aa41eef89309_sequence.pt`
- test: `/data1/yanjing/talk2bev/aide_clip/cache/ravdess_features/ravdess_test_openai_clip-vit-base-patch32_f5_middle_late_a9ce3494_n300_64a031450f5b0f49_sequence.pt`

## Adapter / Temporal Config

| Key | Value |
| --- | --- |
| adapter_hidden_dim | `256` |
| adapter_dropout | `0.2` |
| temporal_head | `transformer` |
| temporal_num_heads | `4` |
| temporal_num_layers | `1` |
| temporal_pooling | `mean` |

## Frozen CLIP Behavior

| Key | Value |
| --- | --- |
| strict_frozen_clip | `true` |
| unfreeze_last_visual_block | `false` |
| visual_block_lr_scale | `0.1` |
| use_global_logit_scale | `false` |
| use_prompt_weight | `false` |
| use_class_temperature | `false` |
| use_class_bias | `false` |
| strict_seed_control | `false` |

## Prompt Setup

- prompts_per_class: `7`
- total_text_prompts: `56`
- label space:
  - `01 -> neutral`
  - `02 -> calm`
  - `03 -> happy`
  - `04 -> sad`
  - `05 -> angry`
  - `06 -> fearful`
  - `07 -> disgust`
  - `08 -> surprised`

## Dataset Sizes For This Run

| Split | Size |
| --- | --- |
| train | `840` |
| val | `300` |
| test | `300` |
| total | `1440` |

Actors recorded in the canonical artifact:

- train actors: `1,4,8,9,10,11,12,17,19,20,21,22,23,24`
- val actors: `3,6,7,13,18`
- test actors: `2,5,14,15,16`

## Canonical Launch Command

```bash
cd /data1/yanjing/talk2bev/aide_clip
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/yanjing/anaconda3/envs/mmtl/bin/python src/clip_ravdess_emotion_train.py \
  --dataset RAVDESS \
  --ravdess_root /data1/yanjing/talk2bev/aide_clip/data/RAVDESS \
  --benchmark_video_list /data1/yanjing/talk2bev/MMEmotionRecognition-main/data/ravdess_videos.csv \
  --split_mode benchmark_5fold \
  --benchmark_test_fold 0 \
  --benchmark_val_fold 1 \
  --allowed_modalities 02 \
  --allowed_vocal_channels 01 \
  --allowed_intensities 01,02 \
  --model_id openai/clip-vit-base-patch32 \
  --prompt_set ravdess_8_facial_cues \
  --execution_mode strict_frozen_clip_adapter \
  --strict_frozen_clip \
  --feature_layout sequence \
  --num_frames 5 \
  --frame_sampling_mode middle_late \
  --sampling_window_start 0.4 \
  --sampling_window_end 0.9 \
  --adapter_hidden_dim 256 \
  --adapter_dropout 0.2 \
  --temporal_head transformer \
  --temporal_num_heads 4 \
  --temporal_num_layers 1 \
  --temporal_pooling mean \
  --epochs 80 \
  --train_batch_size 32 \
  --extract_batch_size 32 \
  --lr 1.5e-4 \
  --weight_decay 5e-4 \
  --loss_type focal \
  --focal_gamma 1.5 \
  --label_smoothing 0.0 \
  --max_grad_norm 1.0 \
  --lr_scheduler none \
  --select_metric weighted_f1 \
  --use_test_ensemble \
  --ensemble_group_size 2 \
  --seed 45 \
  --feature_cache_dir /data1/yanjing/talk2bev/aide_clip/cache/ravdess_features \
  --checkpoint_output results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.ckpt.pt \
  --log_file results/ravdess/diag_ravdess_meanpool_exp1_baseline_vloss.log
```