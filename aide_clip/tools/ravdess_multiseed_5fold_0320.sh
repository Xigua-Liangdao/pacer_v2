#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/yanjing/talk2bev/aide_clip"
PYTHON="/home/yanjing/anaconda3/envs/mmtl/bin/python"
OUT_DIR="$ROOT/results/ravdess/multiseed_5fold_0320"
mkdir -p "$OUT_DIR"

SEEDS=(42 43 44)
FOLDS=(0 1 2 3 4)
GPUS=(0 1 4 5)
NUM_GPUS=${#GPUS[@]}

run_job() {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[RUN] gpu=${gpu} name=${name}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/src/clip_ravdess_emotion_train.py" "$@" \
    --clip_mode offline_only \
    --device cuda:0 \
    --split_mode benchmark_5fold \
    --allowed_modalities 02 \
    --allowed_vocal_channels 01 \
    --allowed_intensities 01,02 \
    --video_extensions .mp4 \
    --adapter_hidden_dim 256 \
    --disable_prompt_weight \
    --disable_class_temperature \
    --disable_class_bias \
    --disable_class_weight \
    --loss_type focal \
    --focal_gamma 1.5 \
    --label_smoothing 0.0 \
    --epochs 80 \
    --select_metric weighted_f1 \
    --num_frames 5 \
    --frame_sampling_mode middle_late \
    --run_zero_shot_eval \
    --report_train_metrics \
    --output "$OUT_DIR/${name}.json" \
    --checkpoint_output "$OUT_DIR/${name}.ckpt.pt" \
    --log_file "$OUT_DIR/${name}.log" \
    > "$OUT_DIR/${name}.stdout" 2>&1
}

commands=()
for fold in "${FOLDS[@]}"; do
  val_fold=$(( (fold + 1) % 5 ))
  for seed in "${SEEDS[@]}"; do
    commands+=("run_job GPU expC_fold${fold}_seed${seed} --seed ${seed} --benchmark_test_fold ${fold} --benchmark_val_fold ${val_fold} --prompt_set ravdess_8_facial_cues")
    commands+=("run_job GPU auto_fold${fold}_seed${seed} --seed ${seed} --benchmark_test_fold ${fold} --benchmark_val_fold ${val_fold} --prompt_set ravdess_8_auto_selected --auto_prompt_k 4 --auto_prompt_refine_passes 3 --auto_prompt_top_pairs 8")
  done
done

echo "[INFO] total_jobs=${#commands[@]} gpus=${GPUS[*]}"

for worker_idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker_idx]}"
  (
    idx=$worker_idx
    while [ "$idx" -lt "${#commands[@]}" ]; do
      cmd="${commands[$idx]}"
      cmd="${cmd/GPU/$gpu}"
      eval "$cmd"
      idx=$((idx + NUM_GPUS))
    done
  ) &
done

wait

echo "[DONE] all multiseed 5-fold jobs finished"
