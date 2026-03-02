# GR00T N1.6 Inference-Time RTC (Real-Time Chunking) Technical Reference

This document provides a technical overview of the Inference-Time RTC implementation in GR00T N1.6, highlighting the mathematical formulation and its relationship to the standard Flow Matching framework.

## Overview

GR00T N1.6 utilizes a **Flow Matching** diffusion policy with a Transformer-based (DiT) backbone. The Inference-Time RTC algorithm is implemented as a guidance-based trajectory modification process that ensures smooth transitions between consecutive action chunks.

By leveraging the differentiability of the DiT backbone and the In-Context Learning capabilities of the Attention mechanism, we achieve robust and physically consistent trajectory stitching.

## Mathematical Formulation

The core update rule for the velocity field in GR00T N1.6's Inference-Time RTC is defined as follows:

$$
\mathbf{v}_{\text{GR00T}}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) = \mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) - w(\tau) \cdot \nabla_{\mathbf{A}_t^\tau} \mathcal{L}
$$

Where:
- $\mathbf{v}_{\text{GR00T}}$ is the modified velocity field used for denoising.
- $\mathbf{v}$ is the original velocity field predicted by the model.
- $\mathbf{A}_t^\tau$ is the noisy action at time step $\tau$.
- $\mathbf{o}_t$ is the observation context.
- $\tau \in [0, 1]$ is the diffusion time step ($0$ = noise, $1$ = data).

### Key Components

#### 1. Predicted Clean Data ($\widehat{\mathbf{A}}_t^1$)
Based on the Flow Matching ODE assumption, we estimate the clean data at $\tau=1$ by linearly extrapolating along the current velocity direction:

$$
\widehat{\mathbf{A}}_t^1 = \mathbf{A}_t^\tau + (1 - \tau)\mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau)
$$

This prediction serves as the basis for calculating the guidance loss.

#### 2. Guidance Loss ($\mathcal{L}$)
We define a weighted Mean Squared Error (MSE) loss between the predicted clean prefix and the target frozen prefix $\mathbf{Y}$:

$$
\mathcal{L} = \frac{1}{2} (\widehat{\mathbf{A}}_t^1 - \mathbf{Y})^\top \text{diag}(\mathbf{W}) (\widehat{\mathbf{A}}_t^1 - \mathbf{Y})
$$

- $\mathbf{Y}$: The frozen prefix (unexecuted tail of the previous action chunk).
- $\mathbf{W}$: A diagonal weight matrix (Soft Mask) that enforces stronger constraints near the stitching boundary.

#### 3. Gradient Term ($\nabla_{\mathbf{A}_t^\tau} \mathcal{L}$)
We compute the gradient of the loss with respect to the **input noisy action** $\mathbf{A}_t^\tau$ using automatic differentiation (Backpropagation):

$$
\nabla_{\mathbf{A}_t^\tau} \mathcal{L} = (\widehat{\mathbf{A}}_t^1 - \mathbf{Y})^\top \text{diag}(\mathbf{W}) \frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}
$$

**Note:** In implementation, we do not explicitly construct the Jacobian matrix $\frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}$. Instead, we compute the vector-Jacobian product directly via `torch.autograd.grad`, which is computationally efficient (equivalent to one backward pass).

**Equivalence Proof:**
Let the error term be:

$$
\mathbf{e} = \widehat{\mathbf{A}}_t^1 - \mathbf{Y}
$$

The loss is defined as:

$$
\mathcal{L} = \frac{1}{2} \mathbf{e}^\top \operatorname{diag}(\mathbf{W}) \mathbf{e}
$$

By the chain rule, the gradient of $\mathcal{L}$ with respect to the input $\mathbf{A}_t^\tau$ is:

$$
\nabla_{\mathbf{A}_t^\tau} \mathcal{L}
=
\frac{\partial \mathcal{L}}{\partial \widehat{\mathbf{A}}_t^1}
\cdot
\frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}
$$

The first term is the weighted error vector:

$$
\frac{\partial \mathcal{L}}{\partial \widehat{\mathbf{A}}_t^1}
=
\operatorname{diag}(\mathbf{W})
(\widehat{\mathbf{A}}_t^1 - \mathbf{Y})
$$

The second term is the Jacobian matrix:

$$
\mathbf{J}
=
\frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}
$$

Therefore, the full gradient becomes:

$$
\nabla_{\mathbf{A}_t^\tau} \mathcal{L}
=
\operatorname{diag}(\mathbf{W})
(\widehat{\mathbf{A}}_t^1 - \mathbf{Y})
\cdot
\mathbf{J}
$$

Thus, the gradient computed by autograd is mathematically identical to the matrix product used in the RTC formula.

#### 5. Mapping to Standard RTC Formula

The GR00T N1.6 implementation directly maps to the standard Inference-Time RTC update rule:

$$
\mathbf{v}_{\text{guide}} = \mathbf{v} + \min \left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right) (\mathbf{Y} - \widehat{\mathbf{A}}_t^1)^\top \text{diag}(\mathbf{W}) \frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}
$$

| Standard RTC Term | GR00T Implementation Code | Note |
| :--- | :--- | :--- |
| $\mathbf{v}$ | `pred_velocity` | Original model prediction. |
| $\min (\dots)$ | `w = min(rtc_beta, (1-tau)/tau)` | Simplified weight schedule. |
| $(\mathbf{Y} - \widehat{\mathbf{A}}_t^1)^\top \text{diag}(\mathbf{W}) \frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}$ | `-1.0 * grad` | `grad = torch.autograd.grad(loss, actions_g)[0]` |

**Code Snippet (`gr00t_n1d6.py` L521-538):**
```python
# 1. Compute Predicted Clean Data (A_hat)
x0_hat = actions_g + (1.0 - tau) * pred_velocity

# 2. Compute Guidance Loss (MSE with Soft Mask)
# diff = (Y - A_hat)
diff = x0_hat[:, :K, :] - frozen_prefix[:, :K, :]
loss = (soft_mask * diff.pow(2)).sum()

# 3. Compute Gradient Term (via Autograd)
grad = torch.autograd.grad(loss, actions_g)[0]

# 4. Compute Weight Schedule (w)
if tau < 1e-8:
    w = rtc_beta
else:
    w = min(rtc_beta, (1.0 - tau) / tau)

# 5. Update Velocity (v_guide = v - w * grad)
actions = actions + dt * pred_velocity.detach() - w * grad.detach()
```

**Why `-1.0 * grad`?**
The standard formula adds a term proportional to $(\mathbf{Y} - \widehat{\mathbf{A}})$. This direction minimizes the error $(\mathbf{Y} - \widehat{\mathbf{A}})^2$.
In our code, `grad` is the gradient of the loss $\nabla \mathcal{L}$. Since we want to **minimize** the loss, we move in the **negative** gradient direction.
Thus: $+ (\mathbf{Y} - \widehat{\mathbf{A}}) \dots \equiv - \nabla \mathcal{L}$.

#### 6. Weight Schedule ($w(\tau)$)
The guidance strength is dynamically adjusted during the diffusion process:

$$
w(\tau) = \min \left(\beta, \frac{1-\tau}{\tau}\right)
$$

- **Early stage ($\tau \to 0$)**: The term $\frac{1-\tau}{\tau}$ approaches infinity, so we clamp it to a maximum value $\beta$ (default `rtc_beta=5.0`) to prevent gradient explosion.
- **Late stage ($\tau \to 1$)**: The guidance strength naturally decays to 0, allowing the model to refine local details without interference.

#### 5. Soft Mask ($\mathbf{W}$)
The weight matrix $\mathbf{W}$ uses an exponential decay schedule to prioritize continuity at the immediate next step:

$$
\mathbf{W}_k = \exp\left(-\frac{\lambda \cdot k}{\max(K-1, 1)}\right), \quad k \in [0, K-1]
$$

- $K$: Length of the frozen prefix.
- $\lambda$: Decay rate (default `rtc_mask_decay=2.0`).

## Comparison with Standard RTC

While derived from the same principles as [Standard Inference-Time RTC](https://arxiv.org/abs/2506.07339), GR00T N1.6 introduces engineering optimizations:

| Component | Standard RTC | GR00T N1.6 | Rationale |
| :--- | :--- | :--- | :--- |
| **Backbone** | Agnostic (CNN/MLP) | **DiT (Transformer)** | Leverages global attention for In-Context trajectory adaptation. |
| **Weight Schedule** | $\min (\beta, \frac{1-\tau}{\tau \cdot r_\tau^2})$ | $\min (\beta, \frac{1-\tau}{\tau})$ | Simplifies calculation while maintaining the monotonic decay property required for stability. |
| **Guidance Calculation** | Explicit Jacobian term | **Autograd of Loss** | Efficient implementation using PyTorch's automatic differentiation engine. |

### Technical Analysis of Divergence

The primary mathematical divergence lies in the omission of the normalization term $r_\tau^2$ in the guidance weight schedule.

In the standard RTC formulation, $r_\tau^2 = \frac{(1-\tau)^2}{\tau^2 + (1-\tau)^2}$ normalizes the guidance scale based on the variance of the Brownian bridge.

**Why GR00T N1.6 simplifies this:**
1.  **Engineering Stability**: The simplified $\frac{1-\tau}{\tau}$ term captures the essential monotonic decay property (strong guidance at $\tau \approx 0$, weak guidance at $\tau \approx 1$).
2.  **Architecture Compatibility**: DiT architectures with global attention are more sensitive to input perturbations. The simplified schedule provides a smoother gradient signal that works well with the Transformer's inductive bias, avoiding potential instability from the more complex $r_\tau^2$ term in high-dimensional action spaces.
3.  **Step-wise vs. Chunk-wise Dependency**: Standard RTC assumes step-wise denoising independence. GR00T's DiT backbone introduces inter-token dependencies via attention. This means the gradient update on the prefix inherently influences the suffix generation through the attention mechanism (Implicit Guidance), reducing the need for the strictly derived theoretical weight.

---
*Implementation Reference: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` (Methods: `get_action_with_features`, `_denoise_step_forward`)*
