# AIDE CLIP 组件说明与公式整理

这份文档整理当前 `aide_clip` 里用到的主要设计、小组件、对应公式，以及前面问到的几个概念区别。

对应实现主文件：
- [aide_clip/src/clip_aide_emotion_train.py](aide_clip/src/clip_aide_emotion_train.py)

---

## 1. 整体流程

当前最主要的方案是 `strict_frozen_clip`：

1. 用冻结的 CLIP 提取图像特征。
2. 用冻结的 CLIP 提取文本 prompt 特征。
3. 只训练一个轻量 `adapter`。
4. 用 adapter 输出的图像特征去和文本特征做相似度匹配。
5. 在相似度之上再叠加若干小组件：
   - `Prompt Weight`
   - `Class Temp.`
   - `Class Bias`
   - `Test Ensemble`
   - `Class Weight`
   - `Label Smoothing`

---

## 2. 基础符号

记：

- 图像序列为 $V$
- 从序列中采样的帧为 $\{I_1, I_2, \dots, I_T\}$
- `num_frames = T`
- CLIP 图像编码器输出单帧特征为 $f_i \in \mathbb{R}^d$
- 类别数为 $C$
- 每个类别的 prompt 数为 $P$
- 第 $c$ 类第 $p$ 条 prompt 的文本特征为 $t_{c,p} \in \mathbb{R}^d$
- adapter 输出后的图像特征为 $z \in \mathbb{R}^d$

这里在当前实现中通常有：

- $d = 512$
- `EMOTION_LABELS = [Anxiety, Peace, Weariness, Happiness, Anger]`
- 即 $C = 5$

---

## 3. Frames 是什么

### 3.1 含义

`Frames` 表示：**每个视频序列取多少帧参与分类**。

实现见：
- [sample_frame_paths()](aide_clip/src/clip_aide_emotion_train.py#L45-L58)
- [extract_image_features()](aide_clip/src/clip_aide_emotion_train.py#L218-L244)

### 3.2 做法

如果 `num_frames <= 1`：
- 只取中间帧。

如果 `num_frames > 1`：
- 在整段序列里均匀采样多帧。

单帧图像特征先归一化，再做平均池化：

$$
\hat f_i = \frac{f_i}{\lVert f_i \rVert_2}
$$

$$
\bar f = \frac{1}{T} \sum_{i=1}^{T} \hat f_i
$$

然后再归一化得到最终图像输入特征：

$$
x = \frac{\bar f}{\lVert \bar f \rVert_2}
$$

### 3.3 直观作用

- 多帧：更稳，能利用时序片段中的更多信息。
- 单帧：更快，但更容易漏掉关键表情或姿态线索。

---

## 4. Prompts 是什么

### 4.1 含义

`Prompts` 表示：**每个类别用几种文本模板描述**。

实现见：
- [build_prompt_templates()](aide_clip/src/clip_aide_emotion_train.py#L179-L202)
- [build_class_prompts()](aide_clip/src/clip_aide_emotion_train.py#L204-L206)

例如 `driving_7` 会为每个类别生成 7 条不同描述。

### 4.2 公式

若第 $c$ 类有 $P$ 条 prompt，则文本特征集合为：

$$
\{t_{c,1}, t_{c,2}, \dots, t_{c,P}\}
$$

每条文本特征都由冻结的 CLIP 文本编码器提取，并做 $L_2$ 归一化。

### 4.3 直观作用

因为同一个类别可以被不同文字表达：
- 有的 prompt 更贴近 CLIP 语义空间
- 有的 prompt 更适合驾驶舱场景
- 多 prompt 往往比单 prompt 更稳

---

## 5. Adapter 是什么

### 5.1 含义

在 `strict_frozen_clip` 模式下，CLIP 本体不训练，只训练一个小型图像 adapter。

实现见：
- [ClipImageAdapter](aide_clip/src/clip_aide_emotion_train.py#L265-L395)

### 5.2 结构

当前 adapter 结构是：

- `input_proj`: `Linear(d, h)`
- `net`: `LayerNorm(h) -> Linear(h, h) -> GELU -> Dropout -> Linear(h, h)`
- `out_proj`: `Linear(h, d)`

其中：
- $d = 512$
- $h = adapter\_hidden\_dim$

### 5.3 公式

输入图像特征为 $x$，先映射到 hidden 空间：

$$
b = W_{in}x + b_{in}
$$

经过中间网络得到残差项：

$$
\Delta = \mathrm{MLP}(b)
$$

做残差融合：

$$
f = b + \Delta
$$

再投影回 CLIP 空间：

$$
u = W_{out} f + b_{out}
$$

最后做归一化：

$$
z = \frac{u}{\lVert u \rVert_2}
$$

其中 $z$ 就是 adapter 输出后的图像特征。

### 5.4 直观作用

adapter 的本质是：
- 不改 CLIP 本体
- 只学习一个轻量映射
- 让视觉特征更贴合当前任务的数据分布

---

## 6. Prompt Weight 是什么

### 6.1 含义

`Prompt Weight` 表示：**同一类别下多条 prompt 不等权，而是学习每条 prompt 的重要性**。

实现见：
- 参数定义：[ClipImageAdapter.__init__()](aide_clip/src/clip_aide_emotion_train.py#L287-L295)
- 使用位置：[logits()](aide_clip/src/clip_aide_emotion_train.py#L351-L366)

### 6.2 为什么要做

同一类下的多条 prompt 质量未必相同：
- 有些 prompt 更自然
- 有些 prompt 更符合 CLIP 预训练分布
- 有些 prompt 可能更噪声

如果直接平均，相当于默认每条 prompt 一样重要。这个假设通常过强。

### 6.3 公式

先计算图像特征与每个类别、每条 prompt 的相似度：

$$
\mathrm{sim}_{c,p} = z^\top t_{c,p}
$$

代码里相当于：
- `sim` 的形状是 `(batch, class, prompt)`

对于每个类别 $c$，学习一组参数 $a_c \in \mathbb{R}^P$，即 `prompt_weight_logits`。

通过 softmax 得到权重：

$$
w_{c,p} = \frac{\exp(a_{c,p})}{\sum_{j=1}^{P} \exp(a_{c,j})}
$$

然后做加权和：

$$
s_c = \sum_{p=1}^{P} w_{c,p} \cdot \mathrm{sim}_{c,p}
$$

如果关闭 `Prompt Weight`，就退化成简单平均：

$$
s_c = \frac{1}{P}\sum_{p=1}^{P} \mathrm{sim}_{c,p}
$$

### 6.4 直观理解

它是在学：
- 同样属于 `Peace` 的 7 条 prompt 里，哪几条更靠谱
- 不是人工拍脑袋平均，而是让模型自己学

---

## 7. Class Temp. 是什么

### 7.1 含义

`Class Temp.` 是：**给每个类别单独一个可学习的缩放系数**。

实现见：
- 参数定义：[ClipImageAdapter.__init__()](aide_clip/src/clip_aide_emotion_train.py#L287-L295)
- 使用位置：[logits()](aide_clip/src/clip_aide_emotion_train.py#L365-L374)

### 7.2 为什么要做

不同类别的 logit 分布可能不一致：
- 有些类别天然分数偏高
- 有些类别天然分数偏低
- 有些类别需要更“锐利”的决策边界

所以允许每个类别单独调节分数尺度。

### 7.3 公式

先有类别相似度 $s_c$。

全局缩放是：

$$
\alpha = \exp(g)
$$

其中 $g$ 对应 `logit_scale`。

类别级缩放是：

$$
\tau_c = \mathrm{clip}(\exp(q_c), 0.5, 2.5)
$$

其中 $q_c$ 对应 `class_logit_scale`。

则缩放后的类别分数为：

$$
\tilde s_c = \alpha \cdot s_c \cdot \tau_c
$$

如果关闭 `Class Temp.`：

$$
\tau_c = 1
$$

### 7.4 直观理解

- `Prompt Weight`：决定同类 prompt 怎么加权
- `Class Temp.`：决定每个类别分数放大还是缩小

它更像分类校准，而不是语义建模本身。

---

## 8. Class Bias 是什么

### 8.1 含义

`Class Bias` 是：**给每个类别再加一个可学习常数偏置项**。

实现见：
- 参数定义：[ClipImageAdapter.__init__()](aide_clip/src/clip_aide_emotion_train.py#L287-L295)
- 使用位置：[logits()](aide_clip/src/clip_aide_emotion_train.py#L365-L374)

### 8.2 为什么要做

即使相似度算得差不多，不同类别的整体输出分布也可能有系统偏移：
- 某些类别整体被压低
- 某些类别整体被抬高

`Class Bias` 允许模型直接学习这种类别先验偏移。

### 8.3 公式

记类别偏置为 $b_c$，则最终 logit 可写为：

$$
\ell_c = \alpha \cdot s_c \cdot \tau_c + b_c
$$

若关闭 `Class Bias`：

$$
b_c = 0
$$

### 8.4 直观理解

- `Class Temp.`：乘法缩放
- `Class Bias`：加法平移

也可以理解成：
- `temp` 调斜率
- `bias` 调截距

---

## 9. Test Ensemble 是什么

### 9.1 含义

`Test Ensemble` 是：**测试时把 prompt 分组，多次打分，再做投票/融合**。

实现见：
- [build_prompt_group_indices()](aide_clip/src/clip_aide_emotion_train.py#L209-L216)
- [grouped_logits()](aide_clip/src/clip_aide_emotion_train.py#L376-L395)
- [predict_emotion_from_features()](aide_clip/src/clip_aide_emotion_train.py#L398-L435)

### 9.2 为什么要做

多 prompt 一次性全部聚合，有时会被某些 prompt 干扰。

测试时把它们按组分开：
- 每组独立给出预测
- 再做组间投票
- 平票时看总分

通常会更稳。

### 9.3 公式

假设一类有 $P$ 条 prompt，按组大小 $G$ 切成若干组：

$$
\mathcal{G}_1, \mathcal{G}_2, \dots, \mathcal{G}_K
$$

对第 $k$ 组，类别 $c$ 的组分数：

$$
s_c^{(k)} = \frac{1}{|\mathcal{G}_k|} \sum_{p \in \mathcal{G}_k} \mathrm{sim}_{c,p}
$$

再加缩放和偏置：

$$
\ell_c^{(k)} = \alpha \cdot s_c^{(k)} \cdot \tau_c + b_c
$$

每组给一个预测：

$$
\hat y^{(k)} = \arg\max_c \ell_c^{(k)}
$$

最终用多数投票：

$$
\hat y = \mathrm{majority\_vote}(\hat y^{(1)}, \dots, \hat y^{(K)})
$$

如果平票，则选择总分更大的类别：

$$
\sum_{k=1}^{K} \ell_c^{(k)}
$$

### 9.4 直观理解

它不是多模型集成，
而是**同一个模型在测试阶段对多组 prompt 做小集成**。

---

## 10. Class Weight 是什么

### 10.1 含义

`Class Weight` 是：**训练时在 loss 里对不同类别施加不同权重**。

实现见：
- [train_strict_frozen_clip()](aide_clip/src/clip_aide_emotion_train.py#L519-L531)

### 10.2 为什么要做

数据类别可能不平衡。常见目标是：
- 稀少类别不要被忽视
- 少数类分错时惩罚更大

### 10.3 当前代码的实际公式

代码先统计训练集每类样本数：

$$
n_c = \text{count of class } c
$$

总样本数：

$$
N = \sum_{c=1}^{C} n_c
$$

先构造未归一化权重：

$$
\tilde w_c = \frac{N}{\max(n_c, 1)}
$$

再除以平均值做归一化：

$$
w_c = \frac{\tilde w_c}{\frac{1}{C}\sum_{j=1}^{C} \tilde w_j}
$$

因此：
- 类别越稀少，$n_c$ 越小，$w_c$ 越大
- 训练时分错这类样本，惩罚越重

最终交叉熵可理解为：

$$
\mathcal{L}_{CE} = - w_y \log p(y)
$$

这里 $y$ 是真实标签。

### 10.4 直观理解

`Class Weight` 管的是：
- **训练时怎么罚**

而不是：
- 最终输出分数怎么平移

---

## 11. Class Bias 和 Class Weight 的区别

这是两个完全不同层面的组件。

### `Class Weight`

- 用在训练 loss 里
- 作用在真实类别的损失权重上
- 目的是缓解类别不均衡
- 改的是“怎么学”

### `Class Bias`

- 用在最终 logit 上
- 给每个类别加一个可学习偏置
- 目的是校准输出分布
- 改的是“怎么判”

### 对比表

| 组件 | 作用位置 | 数学形式 | 主要目的 |
| --- | --- | --- | --- |
| `Class Weight` | loss | $-w_y \log p(y)$ | 训练时更重视少数类 |
| `Class Bias` | logit | $\ell_c = \cdots + b_c$ | 输出层类别校准 |

---

## 12. Label Smoothing 是什么

### 12.1 含义

`Label Smoothing` 是：**不要把真实标签看成绝对 100% 的 one-hot，而是给一点平滑**。

实现见：
- [train_strict_frozen_clip()](aide_clip/src/clip_aide_emotion_train.py#L528-L531)

### 12.2 为什么要做

它可以：
- 降低过拟合
- 缓解模型过度自信
- 让输出更平滑一些

### 12.3 公式

如果平滑系数是 $\varepsilon$，类别数是 $C$，则真实标签分布由 one-hot 变成：

$$
q_c =
\begin{cases}
1-\varepsilon, & c = y \\
\frac{\varepsilon}{C-1}, & c \neq y
\end{cases}
$$

交叉熵目标从 one-hot 变成平滑分布。

---

## 13. Global Logit Scale 是什么

除了上面的开关，adapter 里还有一个 `logit_scale`。

实现见：
- [ClipImageAdapter.__init__()](aide_clip/src/clip_aide_emotion_train.py#L287-L295)
- [logits()](aide_clip/src/clip_aide_emotion_train.py#L365-L374)

对应公式：

$$
\alpha = \exp(g)
$$

最终会乘在整个分数上：

$$
\ell_c = \alpha \cdot s_c \cdot \tau_c + b_c
$$

它的作用是控制整体 logit 尺度。

---

## 14. 当前 full model 的完整打分公式

综合上面几个组件，当前 full model 的类别 $c$ 最终分数可以写成：

### 第一步：多帧聚合图像特征

$$
x = \mathrm{Norm}\left( \frac{1}{T} \sum_{i=1}^{T} \mathrm{Norm}(f_i) \right)
$$

### 第二步：adapter 映射

$$
z = \mathrm{Norm}\left(W_{out}(W_{in}x + \mathrm{MLP}(W_{in}x))\right)
$$

### 第三步：图文相似度

$$
\mathrm{sim}_{c,p} = z^\top t_{c,p}
$$

### 第四步：prompt 加权聚合

$$
s_c = \sum_{p=1}^{P} w_{c,p} \cdot \mathrm{sim}_{c,p}
$$

其中：

$$
w_{c,p} = \mathrm{softmax}(a_c)_p
$$

### 第五步：类别缩放与偏置

$$
\ell_c = \alpha \cdot s_c \cdot \tau_c + b_c
$$

其中：

$$
\alpha = \exp(g), \quad \tau_c = \mathrm{clip}(\exp(q_c), 0.5, 2.5)
$$

### 第六步：softmax 分类

$$
p(c \mid x) = \frac{\exp(\ell_c)}{\sum_{j=1}^{C} \exp(\ell_j)}
$$

训练时如果启用 `Class Weight` 和 `Label Smoothing`，就在交叉熵里进一步加入：
- 类别加权
- 标签平滑

---

## 15. 当前做过的主要 ablation 对应关系

在当前实验里，这些组件对应的 ablation 如下：

| 组件 | ablation case | 含义 |
| --- | --- | --- |
| `Frames` | `one_frame_only` | 从多帧改成单帧 |
| `Prompts` | `single_prompt` | 从多 prompt 改成单 prompt |
| `Prompt Weight` | `no_prompt_weight` | prompt 不再学习权重，改成均值 |
| `Class Temp.` | `no_class_temperature` | 去掉类别级温度缩放 |
| `Class Bias` | `no_class_bias` | 去掉类别偏置 |
| `Class Weight` | `no_class_weight` | 去掉类别损失加权 |
| `Label Smoothing` | `no_label_smoothing` | 去掉标签平滑 |
| `Test Ensemble` | `no_test_ensemble` | 去掉测试时 prompt 分组集成 |

对应表格：
- [aide_clip/results/ablations/ablation_table.md](aide_clip/results/ablations/ablation_table.md)
- [aide_clip/results/ablations/ablation_table.csv](aide_clip/results/ablations/ablation_table.csv)

---

## 16. 一句话总结

当前这套方法可以概括成：

> 用冻结的 CLIP 提供稳定图文表征，用一个轻量 adapter 做任务适配，再通过多 prompt、prompt 加权、类别温度、类别偏置、类别重加权和测试时集成来提高分类性能与稳定性。
