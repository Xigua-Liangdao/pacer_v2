# AIDE CLIP Emotion Experiments

这个目录是从原来的混合代码里拆出来的独立版本，只保留 AIDE 上的 CLIP 情绪分类实验。

## 当前保留内容

- `src/clip_aide_emotion_train.py`
  - AIDE 情绪分类训练脚本
  - 支持 `strict_frozen_clip` 方案
  - 支持多 prompt、class weight、label smoothing、test-time ensemble
- `scripts/run_strict_sweep_v2.sh`
  - 复现最新一版 `strict sweep v2`
- `scripts/run_ablations.sh`
  - 对主要小组件做 ablation study
- `scripts/run_adapter_param_sweep.sh`
  - 对 adapter 的 `hidden_dim` / `dropout` 做参数搜索
- `results/strict_sweep_v2/clip_emotion_strict_sweep_v2_summary.json`
  - 当前最好结果汇总

## 当前最好 full model

当前按最新记录来看，**full model 最高的是 adapter sweep 里的 `h2048_d02`**：

- accuracy: `0.802744`
- weighted_f1: `0.798728`
- 结果文件：`aide_clip/results/adapter_sweep/h2048_d02.json`
- checkpoint：`aide_clip/results/adapter_sweep/h2048_d02.ckpt.pt`
- 关键配置：
  - `strict_frozen_clip=on`
  - `prompt_set=driving_7`
  - `num_frames=5`
  - `use_class_weight=on`
  - `label_smoothing=0.03`
  - `use_test_ensemble=on`
  - `ensemble_group_size=2`
  - `adapter_hidden_dim=2048`
  - `adapter_dropout=0.2`
  - `epochs=40`
  - `batch_size=32`
  - `lr=1.5e-4`
  - `weight_decay=5e-4`
  - `seed=42`

对应汇总文件：

- `aide_clip/results/adapter_sweep/adapter_sweep_summary.json`
- `aide_clip/results/adapter_sweep/adapter_sweep_table.csv`
- `aide_clip/results/adapter_sweep/adapter_sweep_table.md`

## 历史 strict sweep v2 最好结果

当前最好配置来自 `clip_emotion_strict_sweep_v2_c.json`：

- accuracy: `0.790738`
- weighted_f1: `0.789839`
- 关键配置：
  - `epochs=40`
  - `batch_size=32`
  - `lr=1.5e-4`
  - `weight_decay=5e-4`
  - `num_frames=5`
  - `adapter_hidden_dim=1024`
  - `adapter_dropout=0.2`
  - `use_class_weight=on`
  - `label_smoothing=0.03`
  - `use_test_ensemble=on`
  - `ensemble_group_size=2`
  - `strict_frozen_clip=on`
  - `seed=42`

这个结果是早先 `strict_sweep_v2` 的最好记录，保存在：

- `aide_clip/results/strict_sweep_v2/clip_emotion_strict_sweep_v2_summary.json`

## 当前 ablation 基线（full_best）

当前 ablation 表中的完整基线 `full_best`：

- accuracy: `0.799314`
- weighted_f1: `0.794231`
- 结果文件：`aide_clip/results/ablations/full_best.json`
- checkpoint：`aide_clip/results/ablations/full_best.ckpt.pt`

对应汇总与表格：

- `aide_clip/results/ablations/ablation_summary.json`
- `aide_clip/results/ablations/ablation_table.csv`
- `aide_clip/results/ablations/ablation_table.md`

## 实验记录总览

目前目录里保留的主要实验记录如下：

- `results/strict_sweep_v2/`
  - 历史 `strict sweep v2` 结果与汇总
- `results/repro/`
  - 最佳历史配置的复现实验结果、checkpoint、params manifest
- `results/ablations/`
  - 小组件 ablation 的逐项结果、checkpoint、汇总表
- `results/adapter_sweep/`
  - adapter 参数搜索结果、checkpoint、汇总表
- `logs/strict_sweep_v2/`
  - strict sweep 日志
- `logs/ablations/`
  - ablation 日志
- `logs/adapter_sweep/`
  - adapter 参数搜索日志

## 依赖

建议先自行安装与 CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
pip install -r aide_clip/requirements.txt
```

## 数据默认路径

脚本默认读取：

- `AIDE_ROOT=/data1/yanjing/datasets/AIDE/extracted/AIDE_Dataset`
- `AIDE_ANNOTATION_ROOT=$AIDE_ROOT/annotation`

也可以在运行前覆盖环境变量：

```bash
export AIDE_ROOT=/path/to/AIDE_Dataset
export AIDE_ANNOTATION_ROOT=/path/to/AIDE_Dataset/annotation
```

## 运行 strict sweep v2

```bash
bash aide_clip/scripts/run_strict_sweep_v2.sh
```

可选环境变量：

- `PYTHON_BIN`
- `DEVICE`
- `MODEL_ID`
- `HF_HUB_OFFLINE`
- `TRANSFORMERS_OFFLINE`

结果会写到：

- `aide_clip/results/strict_sweep_v2/`
- `aide_clip/logs/strict_sweep_v2/`

## MOSI / MOSEI 准备

当前 `aide_clip` 的主训练脚本仍然是 AIDE 专用；如果要迁移到 MOSI / MOSEI，建议先下载 MMSA 提供的 processed features，再做我们自己的 visual+text 重组。

已提供的最小入口：

- 下载脚本：`aide_clip/scripts/download_msa_features.sh`
- 官方 `.csd` 直链下载：`aide_clip/scripts/download_cmu_sdk_vt.sh`
- 结构检查脚本：`aide_clip/src/inspect_msa_pickle.py`
- manifest 转换脚本：`aide_clip/src/prepare_msa_dataset.py`
- MOSI/MOSEI 训练脚本：`aide_clip/src/clip_msa_sentiment_train.py`
- 迁移说明：`aide_clip/docs/mosi_mosei_migration.md`

默认只下载 MMSA 的 processed features，不下载 raw videos。

示例：

```bash
bash aide_clip/scripts/download_msa_features.sh mosei aligned
bash aide_clip/scripts/download_msa_features.sh mosi aligned

# 如果 Google Drive 路径不稳定，可直接走官方 CSD 直链
bash aide_clip/scripts/download_cmu_sdk_vt.sh mosei
bash aide_clip/scripts/download_cmu_sdk_vt.sh mosi

# 下载完成后先检查 pkl 结构
/data1/yanjing/AGDiff/bin/python aide_clip/src/inspect_msa_pickle.py \
  --input aide_clip/data/mosei/aligned_50.pkl

# 再把 MMSA pkl 转成我们自己的统一 manifest
/data1/yanjing/AGDiff/bin/python aide_clip/src/prepare_msa_dataset.py \
  --dataset mosei \
  --input aide_clip/data/mosei/aligned_50.pkl \
  --output aide_clip/data/mosei/manifest_ternary.json

# 直接在 MOSI / MOSEI processed features 上训练 CLIP-style text+vision baseline
/data1/yanjing/AGDiff/bin/python aide_clip/src/clip_msa_sentiment_train.py \
  --dataset mosi \
  --input aide_clip/data/mosi/aligned_50.pkl \
  --output aide_clip/results/mosi/mosi_clip_tv_ternary.json
```

## 手动下载

如果当前机器连不上外网，你可以先手动下载，再把文件放到固定位置。

### 路线 A：MMSA processed features

1. 打开 MMSA README 中提供的 Google Drive dataset folder。
2. 进入：`MOSEI/Processed/` 或 `MOSI/Processed/`
3. 下载你要的文件：
   - `aligned_50.pkl`
   - 或 `unaligned_50.pkl`
4. 放到：
   - `aide_clip/data/mosei/aligned_50.pkl`
   - `aide_clip/data/mosi/aligned_50.pkl`

推荐先下：

- `MOSEI/Processed/aligned_50.pkl`
- `MOSI/Processed/aligned_50.pkl`

### 路线 B：官方 SDK / CSD

如果你更想保留官方格式，就把 `.csd` 文件放到：

- `aide_clip/data/mosei/csd/`
- `aide_clip/data/mosi/csd/`

MOSEI 当前最小子集：

- `CMU_MOSEI_TimestampedWords.csd`
- `CMU_MOSEI_VisualOpenFace2.csd`
- `CMU_MOSEI_LabelsSentiment.csd`
- `CMU_MOSEI_LabelsEmotions.csd`

MOSI 当前最小子集：

- `CMU_MOSI_TimestampedWords.csd`
- `CMU_MOSI_OpenFace2.csd`
- `CMU_MOSI_Opinion_Labels.csd`

## MELD 准备

如果改走有 raw 视频的情绪数据集，当前已经补了 MELD 的最小入口：

- 下载脚本：`aide_clip/scripts/download_meld_raw.sh`
- 训练脚本：`aide_clip/src/clip_meld_emotion_train.py`

MELD 这条线默认按官方命名规则把 csv 和原始视频 clips 对齐：

- 视频文件名：`diaX_uttY.mp4`
- `X` 分别对应 csv 里的 `Dialogue_ID` 和 `Utterance_ID`

训练脚本会：

1. 递归扫描 `video_root` 下的 `.mp4`
2. 用 `train/dev/test` 三个 csv 构造样本
3. 对每个视频均匀抽多帧
4. 用冻结 CLIP 图像编码器提特征
5. 用多 prompt 文本标签做分类

示例：

```bash
bash aide_clip/scripts/download_meld_raw.sh

/home/yanjing/anaconda3/envs/mmtl/bin/python aide_clip/src/clip_meld_emotion_train.py \
  --video_root aide_clip/data/meld/raw \
  --train_csv aide_clip/data/meld/train_sent_emo.csv \
  --dev_csv aide_clip/data/meld/dev_sent_emo.csv \
  --test_csv aide_clip/data/meld/test_sent_emo.csv \
  --device cuda:0 \
  --clip_mode auto \
  --num_frames 5 \
  --prompt_set meld_scene_9 \
  --epochs 20 \
  --batch_size 16 \
  --use_class_weight on \
  --use_test_ensemble on \
  --feature_cache_dir aide_clip/cache/meld_features \
  --output aide_clip/results/meld/meld_clip_video.json
```

每次运行也会自动保存一个 checkpoint，默认与结果 JSON 同目录，文件名形如：

- `*.ckpt.pt`

## 运行 ablation study

```bash
bash aide_clip/scripts/run_ablations.sh
```

结果会写到：

- `aide_clip/results/ablations/`
- `aide_clip/results/ablations/ablation_summary.json`
- `aide_clip/logs/ablations/`

对应每个 ablation case 也会自动保存自己的：

- 结果 JSON
- checkpoint (`*.ckpt.pt`)
- 运行日志 (`*.log`)

## 运行 adapter 参数搜索

```bash
bash aide_clip/scripts/run_adapter_param_sweep.sh
```

结果会写到：

- `aide_clip/results/adapter_sweep/`
- `aide_clip/results/adapter_sweep/adapter_sweep_summary.json`
- `aide_clip/results/adapter_sweep/adapter_sweep_table.csv`
- `aide_clip/results/adapter_sweep/adapter_sweep_table.md`
- `aide_clip/logs/adapter_sweep/`

每个 case 也会自动保存：

- 结果 JSON
- checkpoint (`*.ckpt.pt`)
- 运行日志 (`*.log`)

## 说明

这里不再包含自动 `git add` / `git commit` / `git push` 逻辑，也不再依赖 `talk2bev/tools` 或 `talk2bev/outputs` 目录。
