# GR00T N1.6 Inference-Time RTC (Real-Time Chunking) Technical Reference

This document provides a technical overview of the Inference-Time RTC implementation in GR00T N1.6, highlighting the mathematical formulation and its relationship to the standard Flow Matching framework.

## Overview

GR00T N1.6 uses a **Flow Matching** diffusion policy with a Transformer-based (DiT) backbone. Its Inference-Time RTC implementation combines:

- **Guidance-based RTC** over the whole frozen prefix when `beta > 0`
- **Delay-aware hard replacement** over the first `fixed_delay_steps` actions
- **Soft-mask weighting** over the remaining overlap region

This design preserves exact continuity for the already-inevitable delay region while still allowing smoother adaptation near the chunk boundary.

## Mathematical Formulation

The core update rule for the velocity field in GR00T N1.6's Inference-Time RTC is defined as follows:

$$
\mathbf{v}_{\text{GR00T}}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) = \mathbf{v}(\mathbf{A}_t^\tau, \mathbf{o}_t, \tau) + w(\tau) \cdot \mathbf{g}
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

This prediction serves as the basis for calculating the RTC guidance term.

#### 2. Weighted Error Term
The implementation forms the soft-masked prefix error

$$
\mathbf{e} = (\mathbf{Y} - \widehat{\mathbf{A}}_t^1) \odot \mathbf{W}
$$

where:

- $\mathbf{Y}$ is the frozen prefix (the unexecuted tail of the previous chunk)
- $\mathbf{W}$ is an exponential soft mask
- the first `fixed_delay_steps` entries have weight $1$, and later entries decay exponentially

Concretely, for prefix index $i$:

$$
W_i = \exp\left(-\lambda \cdot \frac{\max(i-d, 0)}{\max(K-d-1, 1)}\right)
$$

with:

- $\lambda =$ `mask_decay`
- $d =$ `fixed_delay_steps`
- $K =$ frozen-prefix length

#### 3. Vector-Jacobian Product Term
Instead of first constructing a scalar MSE loss and then differentiating it, the implementation directly computes the vector-Jacobian product used in the RTC paper:

$$
\mathbf{g} = \left(\frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}\right)^\top \mathbf{e}
$$

via `torch.autograd.grad(outputs=x0_hat, inputs=actions_g, grad_outputs=error_padded)`.

This is mathematically equivalent to the guidance term in the paper and avoids explicitly materializing the Jacobian.

#### 4. Mapping to Standard RTC Formula

The GR00T N1.6 implementation follows the standard Inference-Time RTC update rule:

$$
\mathbf{v}_{\text{guide}} = \mathbf{v} + \min \left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right) (\mathbf{Y} - \widehat{\mathbf{A}}_t^1)^\top \text{diag}(\mathbf{W}) \frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}
$$

| Standard RTC Term | GR00T Implementation Code | Note |
| :--- | :--- | :--- |
| $\mathbf{v}$ | `pred_velocity` | Original model prediction. |
| $\min (\dots)$ | `w = min(rtc_beta, raw_w.item())` | Uses the full $r_\tau^2$ normalization term. |
| $\left(\frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}\right)^\top ((\mathbf{Y} - \widehat{\mathbf{A}}_t^1)\odot \mathbf{W})$ | `g = torch.autograd.grad(..., grad_outputs=error_padded)[0]` | Direct VJP implementation. |

**Code Snippet (`gr00t_n1d6.py`):**
```python
# 1. Compute Predicted Clean Data (A_hat)
x0_hat = actions_g + (1.0 - tau) * pred_velocity

# 2. Compute weighted prefix error
error = (frozen_prefix[:, :K, :] - x0_hat[:, :K, :]) * soft_mask
error_padded = torch.zeros_like(x0_hat)
error_padded[:, :K, :] = error

# 3. Compute the VJP term directly
g = torch.autograd.grad(
    outputs=x0_hat,
    inputs=actions_g,
    grad_outputs=error_padded.detach(),
)[0]

# 4. Compute Weight Schedule (w)
tau_t = torch.tensor(tau, dtype=torch.float64)
one_minus_tau = 1.0 - tau_t
r_tau_sq = one_minus_tau ** 2 / (tau_t ** 2 + one_minus_tau ** 2)
raw_w = torch.nan_to_num(
    one_minus_tau / (tau_t * r_tau_sq),
    posinf=torch.tensor(rtc_beta, dtype=torch.float64),
)
w = min(rtc_beta, raw_w.item())

# 5. Euler update with RTC guidance
actions = actions + dt * (pred_velocity.detach() + w * g.detach())

# 6. Hard replacement only for the inevitable delay region
replace_k = min(K, fixed_delay_steps)
if replace_k > 0:
    actions[:, :replace_k, :] = (
        (1.0 - tau_next) * noise_frozen[:, :replace_k, :]
        + tau_next * frozen_prefix[:, :replace_k, :]
    )
```

**Why `+ w * g`?**
Here `g` is already defined using the paper's sign convention:

$$
\mathbf{g} = \left(\frac{\partial \widehat{\mathbf{A}}_t^1}{\partial \mathbf{A}_t^\tau}\right)^\top ((\mathbf{Y} - \widehat{\mathbf{A}}_t^1)\odot\mathbf{W})
$$

So the implementation adds `+ w * g` directly, rather than subtracting the gradient of a separately defined scalar loss.

**Physical Meaning of `dt`**:
In the Euler integration step, both the base velocity field `pred_velocity` and the guidance correction `w * g` are treated as velocity components. Therefore, they must both be multiplied by the time step `dt` to compute the displacement.
$$ \mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \cdot (\mathbf{v}_{\text{base}} + \mathbf{v}_{\text{guide}}) $$
The previous implementation omitted `dt` for the guidance term, effectively applying an impulse $1/\Delta t$ times stronger than intended. The corrected formula ensures physical consistency.

## Comparison with Standard RTC

The current implementation is largely aligned with [Standard Inference-Time RTC](https://arxiv.org/abs/2506.07339) in its core guidance equation, but adds a deployment-oriented delay model:

| Component | Standard RTC | GR00T N1.6 | Note |
| :--- | :--- | :--- | :--- |
| **Backbone** | Agnostic (CNN/MLP) | **DiT (Transformer)** | Same RTC logic, different policy backbone. |
| **Weight Schedule** | $\min (\beta, \frac{1-\tau}{\tau \cdot r_\tau^2})$ | Same | No simplification in the current code. |
| **Guidance Calculation** | Jacobian / VJP form | Direct `autograd.grad(..., grad_outputs=...)` | Efficient VJP realization of the same term. |
| **Hard Freezing** | Prefix inpainting formulation | Hard replacement only on first `fixed_delay_steps` tokens | Encodes the inevitable execution delay during deployment. |
| **Prefix Weighting** | Soft weighting near boundary | Delay-aware exponential mask | First `d` steps are weight 1, later overlap decays smoothly. |

### Deployment-Specific Extension

The main practical extension beyond the textbook RTC formulation is the split between:

1. **Hard-frozen delay region**: the first `fixed_delay_steps` actions are overwritten with the OT interpolant at every denoising step.
2. **Soft-guided overlap region**: the remaining frozen prefix is not hard-clamped, but receives exponentially decayed guidance.

This reflects real control latency more faithfully: actions that are already unavoidable should be preserved exactly, while later overlap steps can still adapt for smoother stitching.

---
*Implementation Reference: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py` (Methods: `get_action_with_features`, `_denoise_step_forward`)*
