# Isaac-GR00T RTC 实现说明

本仓库包含两个主要文件夹，分别对应 **Isaac-GR00T** 模型的两种 Real-Time Chunking (RTC) 实现方式。RTC 技术旨在解决机器人控制中的推理延迟问题，通过在训练或推理阶段引入对延迟的处理，实现更平滑、更鲁棒的控制。

## 目录结构

*   **`Isaac-GR00T-rtc-full/`**: 完整版 RTC 实现。支持 **Training-Time RTC**（训练时模拟延迟）和 **Inference-Time RTC**（推理时处理延迟）。
*   **`Isaac-GR00T-rtc-inference-only/`**: 仅推理版 RTC 实现。仅支持 **Inference-Time RTC**，模型训练部分保持原样，仅在推理阶段应用 RTC 策略。

---

## 1. Isaac-GR00T-rtc-full (完整版)

该文件夹下的代码实现了训练端和推理端的RTC（训练端和推理端的 RTC 二选一使用即可）。在训练阶段，模型会显式地模拟推理延迟，从而学习到在延迟存在的情况下如何进行鲁棒的预测。

### 主要功能
*   **Training-Time RTC**: 通过 `training_rtc_max_latency` 参数，在训练过程中随机采样延迟步数，模拟真实的推理滞后。
*   **Inference-Time RTC**: 提供带有 RTC 策略（如冻结步数、权重衰减）的推理脚本。

### 修改的关键文件 (为了实现 RTC)

以下文件相较于原始代码库进行了修改或新增，以支持 RTC 功能：

1.  **`gr00t/configs/finetune_config.py`**
    *   新增 `training_rtc_max_latency` 配置参数，用于控制训练时的最大模拟延迟。

2.  **`gr00t/configs/model/gr00t_n1d6.py`**
    *   在模型配置中新增 `training_rtc_max_latency` 字段。

3.  **`gr00t/experiment/launch_finetune.py`**
    *   修改了训练启动逻辑，将配置中的 `training_rtc_max_latency` 参数传递给模型。

4.  **`gr00t/model/gr00t_n1d6/gr00t_n1d6.py`**
    *   **训练部分 (`forward` 函数)**: 增加了基于 `training_rtc_max_latency` 的逻辑，在计算 Loss 前对 Input Action 进行相应的时间步偏移（模拟延迟）。
    *   **推理部分 (`get_action` 函数)**: 增加了对 `rtc_params` 的支持，允许在生成动作时应用 RTC 混合策略（如 Beta 权重融合、Mask 衰减）。

5.  **`franka_buffer_rtc_threshold.py`** (位于根目录)
    *   新增/修改的推理脚本，实现了具体的 RTC 推理循环，包括 Buffer 管理、Chunk 拼接和 RTC 参数控制（`--rtc-freeze-steps`, `--rtc-beta` 等）。

6.  **`gr00t/policy/rtc_policy.py`**
    *   新增文件，提供了 `RTCPolicyWrapper`，用于封装策略以支持 RTC 参数的传递。

---

## 2. Isaac-GR00T-rtc-inference-only (仅推理版)

该文件夹下的代码仅在推理阶段实现了 RTC。这意味着你可以使用标准的（未经过 RTC 专门训练的）模型权重，直接通过 RTC 推理策略来改善实时控制性能。

### 主要功能
*   **Inference-Time RTC Only**: 仅在推理时生效，不需要重新训练模型。
*   **兼容性**: 适用于任何标准的 Isaac-GR00T 模型权重。

### 修改的关键文件 (为了实现 RTC)

1.  **`gr00t/model/gr00t_n1d6/gr00t_n1d6.py`**
    *   **仅修改推理部分 (`get_action` 函数)**: 增加了对 `rtc_params` 的支持，允许推理时应用 RTC 策略。**注意：** 此处未包含训练时的延迟模拟逻辑。

2.  **`franka_buffer_rtc_threshold.py`** (位于根目录)
    *   同完整版，实现了支持 RTC 的推理循环和参数控制。

3.  **`gr00t/policy/rtc_policy.py`**
    *   同完整版，提供了 `RTCPolicyWrapper`。

---

