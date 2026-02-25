# Real-Time Chunking (RTC) 配置指南

RTC (Real-Time Chunking) 是一种用于解决机器人控制延迟的技术，本文档说明如何在 Isaac-GR00T 训练和推理过程中开启、关闭及配置 RTC。

---

## 1. 训练时RTC (Training-Time RTC)

在微调脚本（如 `examples/FRANKA/finetune_franka_ngpus.sh`）中，通过传递命令行参数来控制是否启用 RTC 训练模式。

只需一个参数 `--training-rtc-max-latency` 即可控制训练阶段的 RTC 开关：

### 开启 RTC
设置 `--training-rtc-max-latency` 为大于 0 的整数：

```bash
# ... 其他参数 ...
--training-rtc-max-latency 24 \
# ...
```

*   `--training-rtc-max-latency <INT>`: 最大模拟延迟步数。**值 > 0 时自动启用 RTC**。
    *   **建议值**: `ceil(max_inference_delay / control_period)`。
    *   例如：推理延迟 350ms，控制周期 16ms，则 350/16 ≈ 22，设置 24 作为安全余量。

### 关闭 RTC
**移除** `--training-rtc-max-latency` 参数（默认值为 0），或显式设置为 0：

```bash
--training-rtc-max-latency 0
```

---

## 2. 推理时RTC (Inference-Time )

在推理客户端脚本（`franka_buffer_rtc_threshold.py`）中，RTC 的行为完全通过命令行参数动态控制，**无需修改代码**。

### 开启 RTC (默认行为)
默认情况下，脚本已启用 RTC（默认 `rtc-freeze-steps=15`）。

```bash
python franka_buffer_rtc_threshold.py
```

### 关闭 RTC
设置 `--rtc-freeze-steps 0` 即可完全禁用 RTC，退化为标准的 Chunk 推理模式（无平滑过渡）。

```bash
python franka_buffer_rtc_threshold.py --rtc-freeze-steps 0
```

### 关键参数配置

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--rtc-freeze-steps` | 15 | RTC 冻结步数。<br>• `>0`: 启用 RTC 并冻结指定步数。<br>• `0`: **禁用 RTC**。<br>• `-1`: 冻结所有剩余步数 (Strongest constraint)。<br>**推荐设置**: `H - s` (Chunk长度 - 每次推理消耗步数)。 |
| `--rtc-beta` | 5.0 | 引导权重裁剪值 (β)。控制新旧轨迹融合的强度。<br>• 值越大，对历史轨迹的约束越强。<br>• 设为 0 则退化为仅 Hard Replacement。 |
| `--rtc-mask-decay` | 2.0 | Soft mask 指数衰减率 (Eq. 5)。控制冻结区域内各步权重的衰减速度。 |

### 常见用法示例

**1. 标准 RTC 模式 (推荐)**
适用于一般延迟情况。
```bash
python franka_buffer_rtc_threshold.py --rtc-freeze-steps 15 --rtc-beta 5.0
```

**2. 强约束模式 (冻结更多步数)**
适用于对平滑性要求极高或延迟较大的场景。
```bash
python franka_buffer_rtc_threshold.py --rtc-freeze-steps 25
```

**3. 禁用 RTC (仅做 Buffer 推理)**
适用于调试或不需要平滑过渡的场景。
```bash
python franka_buffer_rtc_threshold.py --rtc-freeze-steps 0
```
