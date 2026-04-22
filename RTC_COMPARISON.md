# `training-no-dit` 版本的 Training-Time RTC 与 Inference-Time RTC 区别

本文档专门说明 `Isaac-GR00T-rtc-training-no-dit` 里的 **training-time RTC**，与 `Isaac-GR00T-rtc-inference-only` 里的 **inference-time RTC** 有什么不同。

这两者最容易混淆的一点是：

**training-time RTC 去掉的是“推理时的 guidance / inpainting 算法开销”，不是去掉“推理时需要把已知 prefix 传给模型”这件事。**

所以你在 `training-no-dit` 版本里看到 `rtc_freeze_steps`、`frozen_prefix`、`fixed_delay_steps` 这些东西仍然存在，是正常的。

## 1. 一句话区别

- `inference-time RTC`：**主要在推理时补救延迟**。训练基本不变，推理时用 guidance / soft mask / hard replacement 强行把新 chunk 接到旧 chunk 上。
- `training-time RTC`：**主要在训练时把延迟问题教给模型**。训练时随机模拟 prefix delay，推理时不再做 guidance，只做 prefix overwrite 条件采样。

## 2. 论文来源

| 方法 | 对应论文 | 核心思想 |
| :--- | :--- | :--- |
| Inference-Time RTC | `Real-Time Execution of Action Chunking Flow Policies` | 推理时对重叠区做 inpainting / guidance，修正 chunk 边界 |
| Training-Time RTC | `Training-Time Action Conditioning for Efficient Real-Time Chunking` | 训练时模拟延迟，让模型学会“给定 prefix，补全 suffix” |

## 3. 这两个方法分别把 RTC 放在哪里

### Inference-Time RTC

RTC 主要加在 **推理采样过程** 里：

1. 从上一 chunk 取出 overlap 作为 `frozen_prefix`
2. 在每个 denoising step 中：
   - 计算当前预测和 `frozen_prefix` 的误差
   - 用 soft mask 加权
   - 用 guidance / VJP 把整条轨迹往正确边界拉
   - 再对前若干步做 hard replacement

所以它的关键工作量发生在 **推理阶段**。

### 你的 `training-no-dit` 版本 Training-Time RTC

RTC 主要加在 **训练目标** 和 **专用采样路径** 里：

1. 训练时随机采样一个延迟 `L`
2. 把前 `L` 个 action token 当作已知 prefix
3. prefix token：
   - noised action 直接设成 clean action
   - `action encoder` 的 timestep 设成 `t = 1`
4. loss 只算 postfix
5. 推理时只做：
   - 传入 `frozen_prefix`
   - 每一步采样前做 prefix overwrite
   - `action encoder` 对 prefix token 继续用 `t = 1`

在你现在这版 `training-no-dit` 里，训练延迟采样还支持：

- `uniform`
- `normal`
- `exponential`

如果真实部署延迟有明显的最小值、最大值，并且大多数样本围绕某个平均值波动，那么通常 `normal` 会比 `exponential` 更贴近真实情况。

所以它把大部分“处理延迟”的能力前移到了 **训练阶段**。

## 4. 核心差异总表

| 维度 | `training-no-dit` 版本 Training-Time RTC | Inference-Time RTC |
| :--- | :--- | :--- |
| 主要作用阶段 | 训练阶段为主，推理阶段只保留轻量条件采样 | 推理阶段为主 |
| 训练是否改动 | 是。prefix-conditioned loss | 通常否 |
| 推理是否要传 `frozen_prefix` | 是 | 是 |
| 推理是否要估计 delay | 是 | 是 |
| 推理是否有 guidance | 否 | 是 |
| 推理是否有 soft mask | 否 | 是 |
| 推理是否有 `beta` / `mask_decay` | 否 | 是 |
| 推理是否需要梯度 | 否，`torch.inference_mode()` 即可 | guidance 模式下通常需要 `torch.enable_grad()` / `torch.no_grad()` |
| 推理额外开销 | 低 | 高 |
| 边界连续性的主要来源 | 训练中学到的 prefix conditioning + 推理时 prefix overwrite | 推理时 guidance + hard replacement |
| 是否改 `DiT` 主体 | 这个 `no-dit` 版本不改 | 不需要改 |
| timestep 设计 | 仅 `action encoder` 做 per-token timestep，`DiT` 仍是全局 timestep | 通常全局 timestep |

## 5. 为什么 Training-Time RTC 在推理时还需要 `rtc_freeze_steps`

这是最关键也最容易误解的一点。

### 不是因为它还在做 inference-time RTC guidance

在你的 `training-no-dit` 版本里，`rtc_freeze_steps` **已经不是** `beta` / `mask_decay` 那种 inference-time RTC 参数了。

它现在本质上是：

**“本次推理最多把多少 overlap 作为 prefix 传给模型”**

也就是说，它控制的是 **prefix handover**，不是 guidance 强度。

### 为什么必须还要有这个东西

因为 training-time RTC 学到的是：

> “如果前面有一段 prefix 已经确定了，我就根据这段 prefix 去补后面的 suffix。”

那么到了真实推理时，模型必须知道：

- 哪些动作已经属于上一 chunk 的延迟区
- 哪些动作是这一次需要新生成的

如果你在推理时完全不传 prefix，那么模型看到的就只是“从纯噪声开始生成一整段 chunk”，这已经不是 training-time RTC 想要的推理条件了。

所以：

- `training-time RTC` 不需要 inference-time RTC 的 guidance
- 但 **仍然需要 prefix 作为条件**

### 你这版里它的真实含义

在 `training-no-dit` 版本中：

- `--rtc-freeze-steps 0`
  - 不传 prefix
  - 退化成普通 chunk 推理
- `--rtc-freeze-steps -1`
  - 把当前可用 overlap 全部传给模型
- `--rtc-freeze-steps K`
  - 最多只传前 `K` 步 overlap

然后模型内部再结合运行时估计的 `fixed_delay_steps`，决定真正参与 conditioning 的 prefix 长度。

所以这个参数名虽然还叫 `freeze_steps`，但在这份分支里更接近：

- `prefix_steps_cap`
- `max_prefix_overlap`

## 6. 你的 `no-dit` 版本到底改了什么

你的 `training-no-dit` 版本不是“完整版 training-time RTC”，而是一个 **保守版**：

- 已实现：
  - training-time RTC loss
  - training-time RTC sampling path
  - `action encoder` 的 per-token timestep
- 没有实现：
  - `DiT` 主体里的 per-token timestep conditioning

所以它的定位更准确地说是：

**“只在 action encoder 侧接入 training-time RTC 条件信号，不改 DiT 主体”的折中版本。**

这带来的效果是：

- 比纯 inference-time RTC 更省推理开销
- 比“连 DiT 一起改”的完整版 training-time RTC 更保守
- 与原始预训练结构更接近

## 7. 推理路径上的直观区别

### Inference-Time RTC 的推理思路

可以理解为：

> “模型原本不会自动接边，我在推理时每一步都去纠正它。”

因此会出现：

- `beta`
- `mask_decay`
- soft mask
- VJP guidance
- hard replacement

### 你的 `training-no-dit` 版本推理思路

可以理解为：

> “模型在训练时已经学过 prefix-conditioned completion，所以推理时我只要告诉它 prefix 是什么。”

因此只剩：

- `frozen_prefix`
- `fixed_delay_steps`
- prefix overwrite
- prefix token 的 `t = 1`

没有：

- `beta`
- `mask_decay`
- guidance
- soft mask

## 8. 两者的优缺点

### `training-no-dit` 版本 Training-Time RTC

优点：

- 推理开销更低
- 推理路径更简单
- 不需要 guidance 反传
- 更适合实时部署

缺点：

- 训练时必须显式引入 delay 模拟
- 运行效果更依赖训练分布是否覆盖真实延迟
- 你这个 `no-dit` 版本没有改 `DiT` 主体，上限会低于“全模型都支持 per-token timestep”的版本

### Inference-Time RTC

优点：

- 对已有模型更直接
- 即使训练时没做 training-time RTC，也能在推理阶段补救 chunk 边界
- 边界控制更强

缺点：

- 推理更慢
- 实现更复杂
- guidance 模式下需要额外梯度计算

## 9. 什么时候该用哪一个

- 如果你最关心 **实时性和部署开销**：
  优先用你的 `training-no-dit` 版本 training-time RTC。

- 如果你手头已经有一个没做 training-time RTC 的模型，但又想在推理时尽量把边界接平：
  更适合 inference-time RTC。

- 如果你愿意改训练流程，并且目标是长期部署：
  training-time RTC 更像主路线。

## 10. 最后一句结论

对于你的 `training-no-dit` 分支，可以把它理解成：

> **“用训练把 RTC 学进去，用推理时的 prefix handover 把条件喂进去，但不再在推理阶段做 inference-time guidance。”**

所以你看到推理代码里仍然有 `rtc_freeze_steps`，不是因为它还在跑 inference-time RTC，而是因为：

**training-time RTC 仍然需要在推理时告诉模型：哪一段 prefix 已经确定。**
