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

    def _sample_training_rtc_delay(
        self,
        batch_size: int,
        min_latency: int,
        max_latency_exclusive: int,
        device,
    ) -> torch.Tensor:
        if max_latency_exclusive <= 0:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        max_valid_latency = max_latency_exclusive - 1
        min_latency = max(0, min(min_latency, max_valid_latency))
        if min_latency == max_valid_latency:
            return torch.full((batch_size,), min_latency, dtype=torch.long, device=device)

        distribution = getattr(self.config, "training_rtc_delay_distribution", "uniform")
        if distribution == "uniform":
            return torch.randint(min_latency, max_latency_exclusive, (batch_size,), device=device)

        if distribution == "exponential":
            temperature = max(
                float(getattr(self.config, "training_rtc_delay_exponential_temperature", 1.0)),
                1e-6,
            )
            delay_values = torch.arange(
                min_latency, max_latency_exclusive, device=device, dtype=torch.float32
            )
            probs = torch.exp(-temperature * (delay_values - float(min_latency)))
            probs = probs / probs.sum()
            return torch.multinomial(probs, batch_size, replacement=True).to(dtype=torch.long)

        if distribution == "normal":
            delay_values = torch.arange(
                min_latency, max_latency_exclusive, device=device, dtype=torch.float32
            )
            default_mean = 0.5 * (min_latency + max_valid_latency)
            mean = float(getattr(self.config, "training_rtc_delay_normal_mean", -1.0))
            if mean < 0:
                mean = default_mean

            default_std = max((max_valid_latency - min_latency + 1) / 6.0, 1.0)
            std = max(
                float(getattr(self.config, "training_rtc_delay_normal_std", -1.0)),
                0.0,
            )
            if std == 0.0:
                std = default_std

            logits = -0.5 * ((delay_values - mean) / std) ** 2
            probs = torch.exp(logits - logits.max())
            probs = probs / probs.sum()
            return torch.multinomial(probs, batch_size, replacement=True).to(dtype=torch.long)

        raise ValueError(
            f"Unsupported training RTC delay distribution: {distribution}. "
            "Expected one of {'uniform', 'exponential', 'normal'}."
        )

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
        B, T_act, _ = actions.shape
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(B, device=actions.device, dtype=actions.dtype)  # (B,)

        # velocity target is always action - noise, independent of t
        velocity = actions - noise

        # ── Training-Time RTC ────────────────────────────────────────────────
        # Reference: Algorithm 1 in https://arxiv.org/abs/2512.05964
        #
        # Core idea: simulate inference delay by treating the first L actions
        # as a known "prefix" (already executed).  For prefix tokens we set
        # time = 1.0 (= clean data in flow matching) so that x_t = action.
        # Loss is only computed on the remaining "postfix" tokens.
        #
        # Two timestep tensors are maintained:
        #   t_encoder  – per-token (B, T), fed to action_encoder so each token
        #                gets the correct sinusoidal time embedding.
        #   t_dit      – scalar (B,), fed to DiT's AdaLayerNorm (global cond.).
        # ─────────────────────────────────────────────────────────────────────
        t_encoder = None  # Will be set to (B, T) if RTC is active
        action_mask = action_input.action_mask

        if self.training and getattr(self.config, "training_rtc_max_latency", 0) > 0:
            max_latency = min(
                getattr(self.config, "training_rtc_max_latency", 8), T_act
            )
            min_latency = max(0, int(getattr(self.config, "training_rtc_min_latency", 0)))
            min_latency = min(min_latency, max(max_latency - 1, 0))
            if max_latency <= 0:
                # max_latency=0 means no prefix → fall through to standard path
                max_latency = 0
            # Sample L per configured delay distribution.
            L = self._sample_training_rtc_delay(B, min_latency, max_latency, device)

            # prefix_mask: (B, T), True for positions < L (frozen prefix)
            seq_idx = torch.arange(T_act, device=device)[None, :]  # (1, T)
            prefix_mask = seq_idx < L[:, None]                      # (B, T)
            prefix_mask_3d = prefix_mask.unsqueeze(-1)              # (B, T, 1)

            # Per-token time: prefix → 1.0 (clean), suffix → sampled t
            # Paper: time = jnp.where(prefix_mask, 1.0, time[:, None])
            t_per_token = torch.where(
                prefix_mask,
                torch.ones(1, device=device, dtype=t.dtype),
                t[:, None].expand(-1, T_act),
            )  # (B, T)

            # Noisy trajectory with per-token time
            # Paper: x_t = time[:,:,None] * action_chunk + (1 - time[:,:,None]) * noise
            t_expanded = t_per_token.unsqueeze(-1)  # (B, T, 1)
            noisy_trajectory = t_expanded * actions + (1.0 - t_expanded) * noise

            # Loss only on postfix. Keep the masked tensor local so we do not
            # mutate the input batch in place.
            action_mask = action_mask * (~prefix_mask_3d)

            # Discretized per-token time for action_encoder
            t_encoder = (t_per_token * self.num_timestep_buckets).long()  # (B, T)
        else:
            # Standard (non-RTC): scalar time broadcast
            noisy_trajectory = t[:, None, None] * actions + (1.0 - t[:, None, None]) * noise

        # DiT global timestep – always scalar (B,)
        t_dit = (t * self.num_timestep_buckets).long()

        # Action encoder timestep – per-token (B, T) when RTC, else scalar (B,)
        if t_encoder is None:
            t_encoder = t_dit

        action_features = self.action_encoder(noisy_trajectory, t_encoder, embodiment_id)

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
                timestep=t_dit,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_dit,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
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
        action_timesteps_tensor: torch.Tensor,
        dit_timesteps_tensor: torch.Tensor,
        embodiment_id: torch.Tensor,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        """Run a single denoising-step forward pass and return the predicted velocity.

        This helper is shared by the standard (no-grad) path and the
        training-time RTC sampling path.
        """
        action_features = self.action_encoder(actions, action_timesteps_tensor, embodiment_id)
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
                timestep=dit_timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=dit_timesteps_tensor,
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
        Generate actions using Training-Time RTC sampling.

        Reference
        ---------
        Kevin Black, Allen Z. Ren, Michael Equi, Sergey Levine.
        "Training-Time Action Conditioning for Efficient Real-Time Chunking."
        arXiv 2025.  https://arxiv.org/abs/2512.05964

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
            frozen_prefix: Optional prefix actions in normalised action space.
            rtc_params: Optional dict with RTC hyper-parameters:
                - fixed_delay_steps (int, default 0): number of prefix steps
                    to condition on during sampling.
        """
        if rtc_params is None:
            rtc_params = {}
        fixed_delay_steps: int = rtc_params.get("fixed_delay_steps", 0)

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

        K = 0
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
        prefix_steps = min(K, max(fixed_delay_steps, 0))

        # Training-Time RTC sampling: overwrite the committed prefix before
        # each denoising step. The action encoder receives per-token timesteps,
        # while the DiT body still receives a single global timestep.
        for t in range(self.num_inference_timesteps):
            tau = t / float(self.num_inference_timesteps)       # 0, 1/N, …
            t_discretized = int(tau * self.num_timestep_buckets)
            dit_timestep = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                device=device,
                dtype=torch.long,
            )
            action_timestep = dit_timestep
            if prefix_steps > 0:
                action_timestep = torch.full(
                    (batch_size, self.config.action_horizon),
                    fill_value=t_discretized,
                    device=device,
                    dtype=torch.long,
                )
                action_timestep[:, :prefix_steps] = self.num_timestep_buckets

            with torch.no_grad():
                actions_model = actions
                if prefix_steps > 0:
                    actions_model = actions_model.clone()
                    actions_model[:, :prefix_steps, :] = frozen_prefix[:, :prefix_steps, :]
                pred_velocity = self._denoise_step_forward(
                    actions_model,
                    action_timestep,
                    dit_timestep,
                    embodiment_id,
                    state_features,
                    vl_embeds,
                    backbone_output,
                )
            actions = actions_model + dt * pred_velocity
            if prefix_steps > 0:
                actions[:, :prefix_steps, :] = frozen_prefix[:, :prefix_steps, :]

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

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


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
