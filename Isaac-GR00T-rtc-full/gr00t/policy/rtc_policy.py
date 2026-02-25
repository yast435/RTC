"""Real-Time Chunking (RTC) Policy Wrapper.

This module provides ``RTCPolicyWrapper``, a drop-in wrapper that adds
Real-Time Chunking to any existing GR00T policy.  RTC ensures smooth,
temporally-consistent transitions between consecutive action chunks by
constraining the beginning of each new chunk to match the unexecuted tail
of the previous chunk (the "frozen prefix").

Reference
---------
Kevin Black, Manuel Y. Galliker, Sergey Levine.
"Real-Time Execution of Action Chunking Flow Policies."
NeurIPS 2025.  https://arxiv.org/abs/2506.07339

Usage
-----
::

    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.rtc_policy import RTCPolicyWrapper

    base_policy = Gr00tSimPolicyWrapper(
        Gr00tPolicy(embodiment_tag=tag, model_path=path, device=0)
    )

    # Wrap with RTC.  ``execution_horizon`` should equal the number of
    # action steps executed between consecutive policy calls (i.e. the
    # ``n_action_steps`` parameter of ``MultiStepWrapper``).
    policy = RTCPolicyWrapper(base_policy, execution_horizon=8)

    # Then use ``policy.get_action(obs)`` exactly as before – the wrapper
    # automatically manages the frozen-prefix handover between chunks.
"""

from typing import Any

import numpy as np

from .policy import BasePolicy, PolicyWrapper


class RTCPolicyWrapper(PolicyWrapper):
    """Wrapper that adds Real-Time Chunking (RTC) to any GR00T policy.

    Between consecutive ``get_action`` calls the wrapper:

    1. Saves the **normalised** action prediction from the previous call.
    2. On the next call, extracts the *tail* of the previous prediction
       (the portion that was **not** executed) and passes it to the model
       as a ``frozen_prefix`` via the ``options`` dict.
    3. The model's denoising loop constrains its first *K* actions to
       reproduce that tail, guaranteeing temporal continuity at chunk
       boundaries.

    Parameters
    ----------
    policy : BasePolicy
        The inner policy (e.g. ``Gr00tSimPolicyWrapper``) to wrap.
    execution_horizon : int
        Number of action steps consumed by the environment between two
        consecutive ``get_action`` calls.  This is typically equal to
        ``MultiStepConfig.n_action_steps``.
    """

    def __init__(
        self,
        policy: BasePolicy,
        execution_horizon: int,
        rtc_beta: float = 5.0,
        rtc_mask_decay: float = 2.0,
    ):
        super().__init__(policy, strict=False)
        self.execution_horizon = execution_horizon
        self.rtc_beta = rtc_beta
        self.rtc_mask_decay = rtc_mask_decay

        # Internal state – cleared on ``reset()``.
        self._prev_normalized_pred: np.ndarray | None = None  # (B, H, D)

    # -- validation delegates to inner policy ----------------------------------

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    # -- core logic ------------------------------------------------------------

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute action with automatic frozen-prefix management.

        The wrapper transparently builds the ``options["rtc"]`` dict so
        that downstream code (``Gr00tPolicy._get_action``) receives the
        correct ``frozen_prefix`` tensor.
        """
        if options is None:
            options = {}

        # Build frozen prefix from the *unexecuted tail* of the previous chunk.
        if self._prev_normalized_pred is not None:
            s = self.execution_horizon
            tail = self._prev_normalized_pred[:, s:, :]  # (B, H-s, D)
            if tail.shape[1] > 0:
                options.setdefault("rtc", {})
                options["rtc"]["enabled"] = True
                options["rtc"]["frozen_prefix"] = tail
                options["rtc"]["beta"] = self.rtc_beta
                options["rtc"]["mask_decay"] = self.rtc_mask_decay

        # Forward to the inner policy (which forwards to the model).
        action, info = self.policy.get_action(observation, options)

        # Cache normalised prediction for the next call.
        if "normalized_action_pred" in info:
            self._prev_normalized_pred = info["normalized_action_pred"]
        else:
            # Safety: if the inner policy doesn't return normalised preds,
            # we cannot build a frozen prefix – just clear the cache.
            self._prev_normalized_pred = None

        return action, info

    # -- lifecycle -------------------------------------------------------------

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset internal RTC state and forward to the inner policy."""
        self._prev_normalized_pred = None
        return super().reset(options)
