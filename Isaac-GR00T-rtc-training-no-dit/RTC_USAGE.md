# Training-Time RTC 使用说明

本文档对应 `Isaac-GR00T-rtc-training-no-dit`。

这份分支只保留了 **training-time RTC**：
- 训练时：使用 prefix-conditioned loss。
- 推理时：使用 prefix overwrite sampling path。
- `action encoder` 使用 per-token timestep。
- `DiT` 主体仍然使用全局 timestep，**没有**接入 per-token timestep。
- **不包含** inference-time RTC 的 `beta` / `mask_decay` / guidance / soft mask 逻辑。

## 1. 实现概览

当前这版的 training-time RTC 闭环如下：

1. 训练阶段从 `L ~ delay_distribution` 采样一个延迟步数。
2. 前 `L` 个 action token 视为已知 prefix：
   - 这些 token 的 noised action 直接设为 clean action。
   - 这些 token 在 `action encoder` 中使用 `t = 1` 的 per-token timestep。
   - loss 只计算 postfix。
3. 推理阶段把上一个 chunk 未执行完的尾部作为 `frozen_prefix` 传给模型。
4. 每一步采样前，都先把 committed prefix 覆写到当前轨迹，再做一步 denoise。
5. `action encoder` 在 prefix token 上继续使用 `t = 1`，其余 token 使用当前采样步的全局 timestep。

## 2. 如何训练

训练入口是 `gr00t/experiment/launch_finetune.py`，常见启动脚本是 `examples/FRANKA/finetune_franka_ngpus.sh`。

### 开启 training-time RTC

只要把 `--training-rtc-max-latency` 设成大于 0 的整数即可：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 --master_port=29500 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path /path/to/base-model \
  --dataset-path /path/to/dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/FRANKA/modality.py \
  --num-gpus 4 \
  --output-dir /path/to/output \
  --training-rtc-max-latency 24
```

关键参数：

| 参数 | 含义 | 建议 |
| :--- | :--- | :--- |
| `--training-rtc-max-latency` | 最大模拟延迟步数的**上界**，采样区间是 `[min_latency, max_latency)` | 若想覆盖最大延迟 `D_max`，通常设为 `ceil(D_max / control_period) + 1`，或再留 1 到 2 步安全余量 |
| `--training-rtc-min-latency` | 最小模拟延迟步数的下界 | 若已知部署延迟不会低于某值，建议显式设置 |
| `--training-rtc-delay-distribution` | 延迟采样分布，支持 `uniform` / `normal` / `exponential` | 你的场景更推荐 `normal` |
| `--training-rtc-delay-exponential-temperature` | 指数分布温度，仅在 `exponential` 下生效 | 默认 `1.0` |
| `--training-rtc-delay-normal-mean` | 截断正态分布均值，仅在 `normal` 下生效 | 建议设为真实平均延迟对应的步数 |
| `--training-rtc-delay-normal-std` | 截断正态分布标准差，仅在 `normal` 下生效 | 建议设为能覆盖常见波动范围的步数标准差 |

示例：

```bash
--training-rtc-max-latency 24 \
--training-rtc-min-latency 0 \
--training-rtc-delay-distribution uniform
```

如果真实部署时延迟大多集中在某个均值附近，更推荐 `normal`：

```bash
--training-rtc-min-latency 6 \
--training-rtc-max-latency 22 \
--training-rtc-delay-distribution normal \
--training-rtc-delay-normal-mean 12 \
--training-rtc-delay-normal-std 3.5
```

如果你想让训练更偏向短延迟样本，再考虑 `exponential`：

```bash
--training-rtc-max-latency 24 \
--training-rtc-delay-distribution exponential \
--training-rtc-delay-exponential-temperature 1.5
```

### 关闭 training-time RTC

把 `--training-rtc-max-latency` 设为 `0` 即可：

```bash
--training-rtc-max-latency 0
```

### 训练参数怎么选

- 控制周期为 `control_period` 秒、最大端到端推理延迟约为 `max_inference_delay` 秒时，推荐：
  `training_rtc_max_latency = ceil(max_inference_delay / control_period) + 1`
- 如果已知最小端到端推理延迟约为 `min_inference_delay` 秒，推荐：
  `training_rtc_min_latency = round(min_inference_delay / control_period)`
- 例如：最低 `100ms`、最高 `350ms`、控制周期 `17ms`：
  - 最低延迟约 `100 / 17 ~= 6` 步，可设 `--training-rtc-min-latency 6`
  - 最高延迟约 `350 / 17 ~= 21` 步，若严格贴合上界可设 `--training-rtc-max-latency 22`
  - 若想留安全余量，可设 `--training-rtc-max-latency 24`
  - 若总体延迟集中在 `200ms` 左右，即约 `12` 步，更推荐：

```bash
--training-rtc-min-latency 6 \
--training-rtc-max-latency 22 \
--training-rtc-delay-distribution normal \
--training-rtc-delay-normal-mean 12 \
--training-rtc-delay-normal-std 3.5
```

- `uniform` 适合你只知道上下界、不清楚内部形状的时候。
- `normal` 适合你这种“有明确最小值、最大值，而且大多数样本集中在某个平均值附近”的情况。
- `exponential` 更适合“大多数时候都很短延迟，只偶尔拖长”的场景，不太适合你当前这个延迟分布。

## 3. 如何推理

推理客户端是 `franka_buffer_rtc_threshold.py`。

这份分支里的推理不是 inference-time RTC guidance，而是 **training-time RTC 对应的 sampling path**：
- 客户端把上一 chunk 的剩余 overlap 作为 `frozen_prefix` 传给模型。
- 模型每一步采样前先做 prefix overwrite。
- 实际参与 conditioning 的 prefix 长度由 `fixed_delay_steps` 决定。

### 开启 training-time RTC 推理

默认只要 `--rtc-freeze-steps != 0` 就会启用这条路径。

推荐用法：

```bash
python franka_buffer_rtc_threshold.py \
  --rtc-freeze-steps -1 \
  --rtc-hard-freeze-mode auto \
  --rtc-warmup-delay-steps 10
```

含义：

- `--rtc-freeze-steps -1`
  - 把上一 chunk 当前还没执行完的全部 overlap 都传给模型。
  - 模型内部再结合 `fixed_delay_steps`，只把前一部分当成 committed prefix。
- `--rtc-hard-freeze-mode auto`
  - 根据历史运行时延迟估计本次 `fixed_delay_steps`。
- `--rtc-warmup-delay-steps 10`
  - 在延迟历史还不稳定时，先用这个固定值做 warmup。

### 关闭 training-time RTC 推理

```bash
python franka_buffer_rtc_threshold.py --rtc-freeze-steps 0
```

这会退化成标准 chunk 推理，不向模型传任何 prefix。

### 推理参数说明

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--rtc-freeze-steps` | `-1` | 传给模型的 overlap 上限。`-1` 表示传全部剩余 overlap，`0` 表示禁用 RTC，正整数表示最多传前 `K` 步。 |
| `--rtc-warmup-delay-steps` | `10` | 延迟历史不稳定时使用的固定 delay 步数。 |
| `--rtc-hard-freeze-mode` | `auto` | `auto` 表示使用历史延迟估计 `fixed_delay_steps`，`off` 表示固定为 0。 |

### 一个重要细节

`--rtc-freeze-steps` 不是模型最终一定会“冻结”的步数。

当前实现里：
- 客户端先把 overlap 作为 `frozen_prefix` 传入。
- 真实参与 training-time RTC conditioning 的 prefix 长度是：
  `min(available_overlap, rtc_freeze_steps上限, estimated_delay_steps)`

所以更推荐：
- `--rtc-freeze-steps -1`
- `--rtc-hard-freeze-mode auto`

这样实际 prefix 长度主要由运行时估计的 delay 决定。

## 4. 推荐配置

### 推荐训练配置

```bash
--training-rtc-max-latency 24 \
--training-rtc-min-latency 6 \
--training-rtc-delay-distribution normal \
--training-rtc-delay-normal-mean 12 \
--training-rtc-delay-normal-std 3.5
```

### 推荐推理配置

```bash
python franka_buffer_rtc_threshold.py \
  --rtc-freeze-steps -1 \
  --rtc-hard-freeze-mode auto \
  --rtc-warmup-delay-steps 10
```

## 5. 当前版本的边界

- 这份分支只在 `action encoder` 做 per-token timestep。
- `DiT` 主体仍然是全局 timestep，这是 `training-no-dit` 版本的有意设计。
- 因此它比“同时改 DiT 主体”的版本更保守，也更接近原始预训练结构。
