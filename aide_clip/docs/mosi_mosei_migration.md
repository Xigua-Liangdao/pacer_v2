# MOSI / MOSEI 迁移说明

## 目标

把当前 `aide_clip` 的冻结 CLIP 情绪识别方案迁移到常见多模态 benchmark：

- `MOSEI`
- `MOSI`
- `IEMOCAP`（后续，需官方 release form）

## 当前策略

优先顺序：

1. `MOSEI`
2. `MOSI`
3. `IEMOCAP`

原因：

- `MOSEI/MOSI` 可直接通过官方 SDK 或 MMSA processed features 获取
- `IEMOCAP` 需要先提交官方 release form，不适合卡住当前主线

## 设定提醒

这些 benchmark 的社区标准设定通常是：

- `text + audio + vision`

而我们当前的 `aide_clip` 是：

- 冻结 CLIP
- visual + text
- CLIP-style matching / adapter classification

所以后续对外表述必须明确：

1. 我们使用的是 benchmark 的 `visual + text` 子设定
2. 不直接与 full tri-modal 方法做“同设定”对比
3. 更准确的说法是 `CLIP-style multimodal emotion/sentiment understanding`

## 下载策略

今晚优先使用 MMSA 的 processed features：

- 不需要 raw videos
- 直接带 train / valid / test split
- 方便先做 baseline 和表格

已提供脚本：

- `aide_clip/scripts/download_msa_features.sh`

脚本默认行为：

1. 从 MMSA 的 Google Drive folder 拉取 processed features
2. 默认只处理 `aligned_50.pkl`
3. 自动校验 SHA-256
4. 落盘到：
   - `aide_clip/data/mosei/`
   - `aide_clip/data/mosi/`

## 这一步完成后要做什么

下载完成后，下一步是做一个统一 dataset adapter，把 MMSA 的 pickle 格式整理成当前 CLIP 管线能吃的 sample 列表。核心要解决：

1. `text` 直接使用 utterance / raw_text
2. `vision` 需要决定是：
   - 直接把预提取视觉特征接到 adapter
   - 还是回到原始视频帧做 CLIP image encoder
3. 标签要统一：
   - `MOSI/MOSEI` 默认是 sentiment regression label
   - 如果做分类，需要先定义 label mapping / binning

## MAG 预处理对齐

如果要参考 `BERT_multimodal_transformer` 的预处理，核心不是简单地读一个 `pkl`，而是把每个 sample 组织成：

- `((words, visual, acoustic), label, segment)`

其中：

1. `words` 是词序列
2. `visual` / `acoustic` 是按词对齐的时序特征
3. 进入 BERT 前，再做 subword tokenization
4. 如果一个词被切成多个 subword，就把该词对应的 `visual/audio` 特征复制多份

当前已经补了一个转换脚本：

- `aide_clip/src/export_mmsa_to_mag.py`

它会把 MMSA 的 `aligned_50.pkl` 导出成 MAG 风格样本格式。

### 重要提醒

你现在手上的 MMSA `aligned_50.pkl`，特征维度是：

- MOSI: `audio=5`, `vision=20`

而 MAG 官方 README 里默认的维度是：

- MOSI: `audio=74`, `vision=47`

这意味着：

1. 当前文件可以参考 MAG 的“对齐方式”
2. 但不能直接声称和 MAG 论文输入特征完全相同
3. 如果想最大程度逼近论文结果，仍然需要 MAG 官方提供的 `mosi.pkl` / `mosei.pkl`

## 建议的最小迁移路线

第一阶段：

1. 先用 `raw_text + vision features` 做一个 CLIP-style baseline
2. 暂时不碰 audio
3. 先把表头和 split 打通

第二阶段：

1. 再决定是否把 raw video 帧抽出来，回到真正的 CLIP image encoder
2. 再补 audio 或做更完整对比

## IEMOCAP

IEMOCAP 官方入口需要先提交 release form：

- 官方 release 页面：`https://sail.usc.edu/iemocap/iemocap_release.htm`

当前建议：

1. 先并行提交申请
2. 不阻塞 MOSI / MOSEI 的主线推进