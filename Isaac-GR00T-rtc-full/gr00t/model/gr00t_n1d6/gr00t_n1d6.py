from typing import Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    def _denoise_step_forward(
        self,
        actions: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        embodiment_id: torch.Tensor,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        """Run a single denoising-step forward pass and return the predicted velocity.

        This helper is shared by the standard (no-grad) path and the
        guidance-based (grad-enabled) path, avoiding code duplication.
        """
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=actions.device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -self.action_horizon :]
        return pred_velocity

    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        frozen_prefix: torch.Tensor | None = None,
        rtc_params: dict | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process with optional RTC.

        Real-Time Chunking (RTC) enables smooth transitions between consecutive
        action chunks by constraining the beginning of the new chunk to match
        the tail of the previous chunk (frozen_prefix).

        Two RTC modes are supported (controlled via *rtc_params*):

        1. **Guidance-based inpainting** (``beta > 0``, default):
           At each denoising step the predicted clean sample x̂₀ is compared
           against the frozen prefix, and the gradient of a soft-masked MSE
           loss w.r.t. the noised actions is used to steer the *entire* chunk
           toward consistency.  A hard replacement is still applied afterwards
           to guarantee an exact match on the frozen portion.
           This is the full algorithm described in the RTC paper (Algorithm 1).

        2. **Replacement-only** (``beta == 0``):
           The frozen portion is replaced with the correct OT interpolant
           at each step (Diffuser-style).  Cheaper but produces slightly
           less smooth transitions at chunk boundaries.

        Reference
        ---------
        Kevin Black, Manuel Y. Galliker, Sergey Levine.
        "Real-Time Execution of Action Chunking Flow Policies."
        NeurIPS 2025.  https://arxiv.org/abs/2506.07339

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
            frozen_prefix: [B, K, action_dim] – the first K actions of this
                chunk are constrained to equal these values (in normalised
                action space).  K may be smaller than action_horizon.
                Pass None to disable RTC (standard generation).
            rtc_params: Optional dict with RTC hyper-parameters:
                - beta (float, default 5.0): guidance weight clipping value.
                    Set to 0 to fall back to replacement-only mode.
                - mask_decay (float, default 2.0): exponential-decay rate
                    for the soft mask over the frozen prefix.
        """
        # ── RTC hyper-parameters ────────────────────────────────────────────
        if rtc_params is None:
            rtc_params = {}
        rtc_beta: float = rtc_params.get("beta", 5.0)
        rtc_mask_decay: float = rtc_params.get("mask_decay", 2.0)

        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # ── RTC: prepare frozen prefix ──────────────────────────────────────
        K = 0
        noise_frozen = None
        use_guidance = False
        if frozen_prefix is not None:
            if frozen_prefix.dim() == 2:
                frozen_prefix = frozen_prefix.unsqueeze(0)  # (1, K, D)
            if frozen_prefix.shape[0] == 1 and batch_size > 1:
                frozen_prefix = frozen_prefix.expand(batch_size, -1, -1)
            K = min(frozen_prefix.shape[1], actions.shape[1])
            # Pad frozen_prefix action_dim to match model's max_action_dim
            if frozen_prefix.shape[-1] < self.action_dim:
                pad = torch.zeros(
                    *frozen_prefix.shape[:-1],
                    self.action_dim - frozen_prefix.shape[-1],
                    dtype=frozen_prefix.dtype,
                    device=frozen_prefix.device,
                )
                frozen_prefix = torch.cat([frozen_prefix, pad], dim=-1)
            # Save the initial noise for the frozen portion so we can
            # reconstruct the correct OT interpolant at every τ.
            noise_frozen = actions[:, :K, :].clone()
            use_guidance = K > 0 and rtc_beta > 0

        # ── Pre-compute soft mask for guidance (Eq. 5 in RTC paper) ─────────
        # Exponential decay over the frozen prefix: actions near the start
        # (about to be executed) get the strongest guidance; actions near the
        # boundary with the free region get weaker guidance for a smoother
        # transition.
        soft_mask = None
        if use_guidance and K > 0:
            idx = torch.arange(K, device=device, dtype=dtype)
            soft_mask = torch.exp(
                -rtc_mask_decay * idx / max(K - 1, 1)
            )  # shape (K,)
            soft_mask = soft_mask[None, :, None]  # (1, K, 1) for broadcast

        # ── Denoising loop (Euler integration of flow matching ODE) ─────────
        for t in range(self.num_inference_timesteps):
            tau = t / float(self.num_inference_timesteps)       # 0, 1/N, …
            tau_next = (t + 1) / float(self.num_inference_timesteps)
            t_discretized = int(tau * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )

            if use_guidance:
                # ── Guidance-based RTC (Algorithm 1 from the paper) ─────────
                # We need gradients w.r.t. *actions* to compute the guidance
                # signal, so run the forward pass inside ``enable_grad()``.
                with torch.enable_grad():
                    actions_g = actions.detach().clone().requires_grad_(True)

                    pred_velocity = self._denoise_step_forward(
                        actions_g, timesteps_tensor, embodiment_id,
                        state_features, vl_embeds, backbone_output,
                    )

                    # Predicted clean sample: x̂₀ = x_τ + (1 − τ) · v
                    x0_hat = actions_g + (1.0 - tau) * pred_velocity

                    # Soft-masked MSE loss on the frozen portion
                    diff = x0_hat[:, :K, :] - frozen_prefix[:, :K, :]
                    loss = (soft_mask * diff.pow(2)).sum()

                    # Gradient of guidance loss w.r.t. input actions
                    grad = torch.autograd.grad(loss, actions_g)[0]

                # ── Guidance weight (Eq. 2 from RTC paper) ──────────────────
                # w(τ) = min(β, (1−τ) / (τ · r_τ²))
                # where r_τ² = (1−τ)² / ((1−τ)² + τ²)
                # so 1/r_τ² = ((1−τ)² + τ²) / (1−τ)²
                # Full: w(τ) = min(β, (1−τ)/τ · ((1−τ)² + τ²) / (1−τ)²)
                tau_t = torch.as_tensor(tau, device=device, dtype=dtype)
                one_minus_tau = 1.0 - tau_t
                # inv_r2 = 1/r_τ²  (correction factor from optimal transport)
                inv_r2 = torch.nan_to_num(
                    (one_minus_tau ** 2 + tau_t ** 2) / (one_minus_tau ** 2),
                    posinf=rtc_beta,
                )
                c = torch.nan_to_num(
                    one_minus_tau / tau_t,
                    posinf=rtc_beta,
                )
                w = torch.nan_to_num(c * inv_r2, posinf=rtc_beta)
                w = torch.minimum(w, torch.as_tensor(rtc_beta, device=device, dtype=dtype)).item()

                # Euler step + guidance correction (single step!)
                actions = actions + dt * pred_velocity.detach() - w * grad.detach()
            else:
                # ── Standard Euler step (no guidance) ───────────────────────
                with torch.no_grad():
                    pred_velocity = self._denoise_step_forward(
                        actions, timesteps_tensor, embodiment_id,
                        state_features, vl_embeds, backbone_output,
                    )
                actions = actions + dt * pred_velocity

            # ── RTC hard replacement: overwrite frozen portion with OT
            #    interpolant to *guarantee* exact convergence at τ=1. ────────
            if K > 0 and noise_frozen is not None:
                actions[:, :K, :] = (
                    (1.0 - tau_next) * noise_frozen
                    + tau_next * frozen_prefix[:, :K, :]
                )

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        frozen_prefix: torch.Tensor | None = None,
        rtc_params: dict | None = None,
    ) -> BatchFeature:
        """Generate actions via backbone features → encode → denoise.

        This is the main inference entry point for the action head.

        Args:
            backbone_output: Output from the backbone model
            action_input: Action-side inputs (state, embodiment_id, …)
            frozen_prefix: Optional frozen prefix for RTC (see get_action_with_features)
            rtc_params: Optional RTC hyper-parameters (see get_action_with_features)
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            frozen_prefix=frozen_prefix,
            rtc_params=rtc_params,
        )


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(
        self,
        inputs: dict,
        frozen_prefix: torch.Tensor | None = None,
        rtc_params: dict | None = None,
    ) -> BatchFeature:
        """Generate actions using the complete model.

        Args:
            inputs: Dictionary containing backbone and action inputs.
            frozen_prefix: Optional frozen prefix for RTC – see
                ``Gr00tN1d6ActionHead.get_action_with_features`` for details.
            rtc_params: Optional RTC hyper-parameters – see
                ``Gr00tN1d6ActionHead.get_action_with_features`` for details.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs,
            frozen_prefix=frozen_prefix,
            rtc_params=rtc_params,
        )

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
