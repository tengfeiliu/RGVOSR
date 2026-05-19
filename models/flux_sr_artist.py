import importlib
import os
from pathlib import Path

import torch
import torch.nn as nn

from models.degradation_vector_encoder import DegradationVectorEncoder
from models.lr_condition_encoder import LRConditionEncoder
from models.visual_condition_adapter import VisualConditionAdapter
from rg_flux_fm import convert_sigma_to_flux_timestep


def _cfg(config, path, default=None):
    current = config
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def _dtype_from_config(value):
    if value in {torch.float32, torch.float16, torch.bfloat16}:
        return value
    value = str(value or "bf16").lower()
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    return latent_image_ids.reshape(height * width, 3)


def _module_device(module):
    return next(module.parameters()).device


def _pack_latents(latents):
    bsz, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"FLUX packed latents require even H/W, got {height}x{width}")
    latents = latents.view(bsz, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(bsz, (height // 2) * (width // 2), channels * 4)


def _unpack_latents(tokens, height, width, latent_channels):
    bsz = tokens.shape[0]
    tokens = tokens.view(bsz, height // 2, width // 2, latent_channels, 2, 2)
    tokens = tokens.permute(0, 3, 1, 4, 2, 5)
    return tokens.reshape(bsz, latent_channels, height, width)


def _import_hf_deepspeed_config():
    try:
        from transformers.integrations import HfDeepSpeedConfig
    except ImportError:
        try:
            from transformers.integrations.deepspeed import HfDeepSpeedConfig
        except ImportError as exc:
            raise ImportError("DeepSpeed ZeRO-3 loading requires transformers with HfDeepSpeedConfig.") from exc
    return HfDeepSpeedConfig


def _clear_hf_deepspeed_config():
    for module_name in ("transformers.integrations.deepspeed", "transformers.deepspeed"):
        try:
            deepspeed_module = importlib.import_module(module_name)
        except ImportError:
            continue
        unset_config = getattr(deepspeed_module, "unset_hf_deepspeed_config", None)
        if callable(unset_config):
            unset_config()
        if hasattr(deepspeed_module, "_hf_deepspeed_config_weak_ref"):
            setattr(deepspeed_module, "_hf_deepspeed_config_weak_ref", None)


def _clip_position_embedding_length(text_pipeline):
    text_encoder = getattr(text_pipeline, "text_encoder", None)
    text_model = getattr(text_encoder, "text_model", None)
    embeddings = getattr(text_model, "embeddings", None)
    position_embedding = getattr(embeddings, "position_embedding", None)
    weight = getattr(position_embedding, "weight", None)
    if weight is None:
        return None
    return int(weight.shape[0])


class FluxSRArtist(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flux_model_path = _cfg(config, "model.flux_model_path", _cfg(config, "flux_model_path"))
        if not self.flux_model_path:
            raise ValueError("model.flux_model_path is required")

        self.weight_dtype = _dtype_from_config(_cfg(config, "model.dtype", "bf16"))
        self.lr_cond_mode = _cfg(config, "condition.lr_cond_mode", "latent_adapter")
        self.guidance_scale = float(_cfg(config, "model.guidance_scale", 1.0))
        self.timestep_mode = _cfg(config, "flow_matching.timestep_conversion", "sigma")
        self.use_degradation_vector = bool(_cfg(config, "condition.use_degradation_vector", True))
        self.use_visual_semantic_tokens = bool(_cfg(config, "condition.use_visual_semantic_tokens", False))
        self.use_lora = bool(_cfg(config, "model.use_lora", True))
        self.text_encoder_device = str(_cfg(config, "model.text_encoder_device", "cpu"))
        self.text_encoder_dtype = _dtype_from_config(_cfg(config, "model.text_encoder_dtype", "fp32"))
        self.vae_device = str(_cfg(config, "model.vae_device", "cpu"))
        default_vae_dtype = "fp32" if self.vae_device.lower() == "cpu" else _cfg(config, "model.dtype", "bf16")
        self.vae_dtype = _dtype_from_config(_cfg(config, "model.vae_dtype", default_vae_dtype))
        self.max_prompt_sequence_length = int(_cfg(config, "model.max_prompt_sequence_length", 128))

        object.__setattr__(self, "vae", None)
        self.transformer = None
        self.text_pipeline = None
        self.vae_scale_factor = 8
        self.latent_channels = int(_cfg(config, "model.latent_channels", 16))
        self.context_dim = int(_cfg(config, "model.context_dim", 4096))
        self.image_token_dim = int(_cfg(config, "model.image_token_dim", self.latent_channels * 4))

        self._load_flux_modules()
        self._infer_dimensions()
        self._build_condition_modules()
        self._apply_train_strategy()

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args:
            first = args[0]
            if isinstance(first, (torch.device, str, int)):
                device = first
        if self.vae is not None and self.vae_device.lower() in {"cuda", "gpu", "same"} and device is not None:
            self.vae.to(device=device, dtype=self.vae_dtype)
        elif self.vae is not None and self.vae_device.lower() == "cpu":
            self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return module

    def _load_flux_modules(self):
        try:
            from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel
        except ModuleNotFoundError as exc:
            raise ImportError(
                "RG-FLUX-SR-MS requires diffusers with FLUX support. "
                "Install project dependencies before constructing FluxSRArtist."
            ) from exc

        vae = AutoencoderKL.from_pretrained(
            self.flux_model_path,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
        )
        if self.vae_device.lower() == "cpu":
            vae.to("cpu")
        elif self.vae_device.lower() not in {"cuda", "gpu", "same"}:
            vae.to(self.vae_device)
        object.__setattr__(self, "vae", vae)

        hf_ds_config = None
        hf_zero3_config = _cfg(self.config, "_runtime.hf_zero3_config", None)
        if hf_zero3_config:
            HfDeepSpeedConfig = _import_hf_deepspeed_config()
            hf_ds_config = HfDeepSpeedConfig(hf_zero3_config)
        try:
            self.transformer = FluxTransformer2DModel.from_pretrained(
                self.flux_model_path,
                subfolder="transformer",
                torch_dtype=self.weight_dtype,
            )
        finally:
            if hf_ds_config is not None:
                _clear_hf_deepspeed_config()
                hf_ds_config = None

        self.text_pipeline = FluxPipeline.from_pretrained(
            self.flux_model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.text_encoder_dtype,
        )
        self._validate_text_pipeline()
        if self.text_encoder_device and self.text_encoder_device.lower() != "cpu":
            self.text_pipeline.to(self.text_encoder_device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        for module in (getattr(self.text_pipeline, "text_encoder", None), getattr(self.text_pipeline, "text_encoder_2", None)):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

        if hasattr(self.vae.config, "block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def _validate_text_pipeline(self):
        clip_position_count = _clip_position_embedding_length(self.text_pipeline)
        if clip_position_count == 0:
            raise RuntimeError(
                "FLUX text_encoder position embeddings are empty. This usually means the text pipeline "
                "was loaded while the global Hugging Face DeepSpeed ZeRO-3 config was active. "
                "Keep HfDeepSpeedConfig scoped to FluxTransformer2DModel.from_pretrained only."
            )

    def _infer_dimensions(self):
        vae_config = getattr(self.vae, "config", None)
        if hasattr(vae_config, "latent_channels"):
            self.latent_channels = int(vae_config.latent_channels)
        transformer_config = getattr(self.transformer, "config", None)
        for attr in ("joint_attention_dim", "context_dim"):
            if hasattr(transformer_config, attr):
                self.context_dim = int(getattr(transformer_config, attr))
                break
        if hasattr(transformer_config, "in_channels"):
            self.image_token_dim = int(getattr(transformer_config, "in_channels"))

    def _build_condition_modules(self):
        condition_dropout = float(_cfg(self.config, "condition.condition_dropout", 0.0))
        deg_token_count = int(_cfg(self.config, "condition.deg_token_count", 4))
        self.degradation_encoder = DegradationVectorEncoder(
            in_dim=int(_cfg(self.config, "condition.deg_vector_dim", 8)),
            hidden_dim=int(_cfg(self.config, "condition.deg_hidden_dim", min(self.context_dim, 1024))),
            context_dim=self.context_dim,
            num_tokens=deg_token_count,
            dropout=condition_dropout,
        )
        self.lr_condition_encoder = LRConditionEncoder(
            latent_channels=self.latent_channels,
            context_dim=self.context_dim,
            num_tokens=int(_cfg(self.config, "condition.lr_token_count", 64)),
            mode=self.lr_cond_mode,
            image_token_dim=self.image_token_dim,
            dropout=condition_dropout,
        )
        self.visual_condition_adapter = VisualConditionAdapter(
            in_dim=int(_cfg(self.config, "condition.visual_feature_dim", 768)),
            context_dim=self.context_dim,
            num_tokens=int(_cfg(self.config, "condition.visual_token_count", 64)),
            dropout=condition_dropout,
        )

    def _apply_lora(self):
        if not self.use_lora:
            return
        try:
            from peft import LoraConfig, PeftModel
        except ModuleNotFoundError as exc:
            raise ImportError("LoRA training requires peft. Add peft to the environment.") from exc

        rank = int(_cfg(self.config, "model.lora_rank", 16))
        alpha = int(_cfg(self.config, "model.lora_alpha", rank))
        target_modules = _cfg(
            self.config,
            "model.lora_target_modules",
            [
                "x_embedder",
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ],
        )
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        self.transformer = PeftModel(self.transformer, lora_config, adapter_name="flux_adapter")

    def _unfreeze_last_blocks(self, n_blocks):
        if n_blocks <= 0:
            return
        base = getattr(self.transformer, "base_model", self.transformer)
        base = getattr(base, "model", base)
        blocks = []
        for name in ("transformer_blocks", "single_transformer_blocks"):
            module_list = getattr(base, name, None)
            if module_list is not None:
                blocks.extend(list(module_list)[-n_blocks:])
        for block in blocks:
            for param in block.parameters():
                param.requires_grad_(True)

    def _apply_train_strategy(self):
        self.transformer.requires_grad_(False)
        freeze_transformer = bool(_cfg(self.config, "training.freeze_flux_transformer", True))
        if not freeze_transformer:
            self.transformer.requires_grad_(True)
        self._apply_lora()

        stage = str(_cfg(self.config, "training.stage", "A")).upper()
        if stage == "B":
            self._unfreeze_last_blocks(int(_cfg(self.config, "training.unfreeze_last_n_blocks", 0)))

        for module in (self.degradation_encoder, self.lr_condition_encoder, self.visual_condition_adapter):
            module.train()
            module.requires_grad_(True)

        disable_gradient_checkpointing = bool(
            _cfg(self.config, "_runtime.disable_transformer_gradient_checkpointing", False)
        )
        if disable_gradient_checkpointing:
            if hasattr(self.transformer, "disable_gradient_checkpointing"):
                self.transformer.disable_gradient_checkpointing()
            return

        if bool(_cfg(self.config, "model.gradient_checkpointing", True)):
            if hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    @torch.no_grad()
    def encode_images(self, images, sample=True):
        posterior = self.vae.encode(images.to(device=_module_device(self.vae), dtype=self.vae.dtype)).latent_dist
        latents = posterior.sample() if sample else posterior.mode()
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)
        return ((latents - shift) * scale).to(dtype=self.weight_dtype)

    @torch.no_grad()
    def decode_latents(self, latents):
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)
        latents = latents / scale + shift
        return self.vae.decode(latents.to(device=_module_device(self.vae), dtype=self.vae.dtype), return_dict=False)[0]

    @torch.no_grad()
    def encode_prompts(self, prompts, device=None, dtype=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        text_device = self.text_encoder_device if self.text_encoder_device else "cpu"
        if text_device.lower() == "cpu" and hasattr(self.text_pipeline, "to"):
            self.text_pipeline.to("cpu")
        encode_kwargs = {"prompt": prompts, "prompt_2": None}
        if self.max_prompt_sequence_length > 0:
            encode_kwargs["max_sequence_length"] = self.max_prompt_sequence_length
        try:
            prompt_embeds, pooled_prompt_embeds, text_ids = self.text_pipeline.encode_prompt(**encode_kwargs)
        except TypeError:
            encode_kwargs.pop("max_sequence_length", None)
            prompt_embeds, pooled_prompt_embeds, text_ids = self.text_pipeline.encode_prompt(**encode_kwargs)
        device = device or _module_device(self.transformer)
        dtype = dtype or self.weight_dtype
        return (
            prompt_embeds.to(device=device, dtype=dtype),
            pooled_prompt_embeds.to(device=device, dtype=dtype),
            text_ids.to(device=device, dtype=dtype),
        )

    def extract_visual_tokens(self, lq_up):
        return None

    def _extra_text_ids(self, token_count, device, dtype, token_type=0):
        if token_count <= 0:
            return torch.zeros(0, 3, device=device, dtype=dtype)
        ids = torch.zeros(token_count, 3, device=device, dtype=dtype)
        ids[:, 0] = token_type
        return ids

    def build_context(
        self,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids=None,
        degradation_vector=None,
        z_lr=None,
        dino_tokens=None,
        lr_cond_mode="latent_adapter",
    ):
        context = [prompt_embeds]
        ids = [text_ids] if text_ids is not None else [
            self._extra_text_ids(prompt_embeds.shape[1], prompt_embeds.device, prompt_embeds.dtype)
        ]

        if self.use_degradation_vector and degradation_vector is not None:
            deg_tokens = self.degradation_encoder(degradation_vector.to(prompt_embeds.device))
            if deg_tokens.shape[1] > 0:
                context.append(deg_tokens.to(dtype=prompt_embeds.dtype))
                ids.append(self._extra_text_ids(deg_tokens.shape[1], prompt_embeds.device, prompt_embeds.dtype, token_type=1))

        if self.use_visual_semantic_tokens and dino_tokens is not None:
            visual_tokens = self.visual_condition_adapter(dino_tokens)
            if visual_tokens is not None and visual_tokens.shape[1] > 0:
                context.append(visual_tokens.to(dtype=prompt_embeds.dtype))
                ids.append(self._extra_text_ids(visual_tokens.shape[1], prompt_embeds.device, prompt_embeds.dtype, token_type=2))

        if lr_cond_mode == "latent_adapter" and z_lr is not None:
            lr_tokens = self.lr_condition_encoder(z_lr.to(prompt_embeds.device), mode="latent_adapter")
            if lr_tokens.shape[1] > 0:
                context.append(lr_tokens.to(dtype=prompt_embeds.dtype))
                ids.append(self._extra_text_ids(lr_tokens.shape[1], prompt_embeds.device, prompt_embeds.dtype, token_type=3))

        return torch.cat(context, dim=1), pooled_prompt_embeds, torch.cat(ids, dim=0)

    def forward(
        self,
        z_t,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids=None,
        degradation_vector=None,
        z_lr=None,
        dino_tokens=None,
        lr_cond_mode=None,
    ):
        lr_cond_mode = lr_cond_mode or self.lr_cond_mode
        bsz, channels, height, width = z_t.shape
        hidden_states = _pack_latents(z_t)
        context, pooled, txt_ids = self.build_context(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            degradation_vector=degradation_vector,
            z_lr=z_lr,
            dino_tokens=dino_tokens,
            lr_cond_mode=lr_cond_mode,
        )

        img_ids = _prepare_latent_image_ids(height // 2, width // 2, z_t.device, hidden_states.dtype)
        lr_token_count = 0
        if lr_cond_mode == "latent_concat":
            if z_lr is None:
                raise ValueError("z_lr is required when lr_cond_mode='latent_concat'")
            lr_tokens = self.lr_condition_encoder(z_lr.to(z_t.device), mode="latent_concat").to(dtype=hidden_states.dtype)
            lr_token_count = lr_tokens.shape[1]
            hidden_states = torch.cat([lr_tokens, hidden_states], dim=1)
            lr_img_ids = self._extra_text_ids(lr_token_count, z_t.device, hidden_states.dtype, token_type=4)
            img_ids = torch.cat([lr_img_ids, img_ids], dim=0)

        guidance = torch.full((bsz,), self.guidance_scale, device=z_t.device, dtype=hidden_states.dtype)
        flux_timestep = convert_sigma_to_flux_timestep(timestep.to(z_t.device), self.timestep_mode).to(dtype=hidden_states.dtype)
        model_out = self.transformer(
            hidden_states=hidden_states,
            timestep=flux_timestep,
            guidance=guidance,
            pooled_projections=pooled,
            encoder_hidden_states=context,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        packed_pred = model_out[0] if isinstance(model_out, (tuple, list)) else model_out.sample
        if lr_token_count:
            packed_pred = packed_pred[:, lr_token_count:]
        return _unpack_latents(packed_pred, height, width, channels).to(dtype=z_t.dtype)

    def save_trainable(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.use_lora and hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(output_dir / "flux_adapter")
            lora_state = {
                name: param.detach().cpu()
                for name, param in self.transformer.state_dict().items()
                if "lora" in name.lower()
            }
            torch.save(lora_state, output_dir / "flux_lora_state.pt")
        trainable_state = {
            name: param.detach().cpu()
            for name, param in self.state_dict().items()
            if any(key in name for key in ("degradation_encoder", "lr_condition_encoder", "visual_condition_adapter"))
        }
        torch.save(trainable_state, output_dir / "condition_adapters.pt")

    def load_trainable(self, checkpoint_dir, is_trainable=True):
        checkpoint_dir = Path(checkpoint_dir)
        if (checkpoint_dir / "rg_flux_adapters").exists():
            checkpoint_dir = checkpoint_dir / "rg_flux_adapters"
        adapter_dir = checkpoint_dir / "flux_adapter"
        lora_state_path = checkpoint_dir / "flux_lora_state.pt"
        if lora_state_path.exists() and self.use_lora:
            state = torch.load(lora_state_path, map_location="cpu")
            self.transformer.load_state_dict(state, strict=False)
        elif adapter_dir.exists() and self.use_lora and not hasattr(self.transformer, "peft_config"):
            from peft import PeftModel

            base = getattr(self.transformer, "base_model", self.transformer)
            self.transformer = PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)
        adapter_state = checkpoint_dir / "condition_adapters.pt"
        if adapter_state.exists():
            state = torch.load(adapter_state, map_location="cpu")
            self.load_state_dict(state, strict=False)
