import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn

from models.degradation_vector_encoder import DegradationVectorEncoder
from models.flux_sr_artist import (
    _cfg,
    _clear_hf_deepspeed_config,
    _dtype_from_config,
    _import_hf_deepspeed_config,
    _module_device,
)
from models.lr_condition_encoder import LRConditionEncoder
from models.visual_condition_adapter import VisualConditionAdapter
from rg_flux_fm import convert_sigma_to_flux_timestep


def _module_dtype(module, default=torch.float32):
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return default


def _patchify_latents(latents):
    bsz, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"FLUX.2 patchified latents require even H/W, got {height}x{width}")
    latents = latents.view(bsz, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(bsz, channels * 4, height // 2, width // 2)


def _unpatchify_latents(latents, vae_latent_channels):
    bsz, channels, height, width = latents.shape
    expected_channels = int(vae_latent_channels) * 4
    if channels != expected_channels:
        raise ValueError(f"Expected {expected_channels} FLUX.2 latent channels, got {channels}")
    latents = latents.view(bsz, vae_latent_channels, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(bsz, vae_latent_channels, height * 2, width * 2)


def _flatten_image_tokens(latents):
    return latents.flatten(2).transpose(1, 2)


def _unflatten_image_tokens(tokens, height, width):
    bsz, _, channels = tokens.shape
    return tokens.transpose(1, 2).reshape(bsz, channels, height, width)


def _latent_image_ids(batch_size, height, width, device, dtype):
    ids = torch.zeros(height, width, 4, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    ids = ids.reshape(1, height * width, 4)
    return ids.expand(batch_size, -1, -1)


def _retrieve_latents(encoded, sample=True):
    if hasattr(encoded, "latent_dist"):
        latent_dist = encoded.latent_dist
    else:
        latent_dist = encoded[0]
    if sample and hasattr(latent_dist, "sample"):
        return latent_dist.sample()
    if hasattr(latent_dist, "mode"):
        return latent_dist.mode()
    if hasattr(latent_dist, "mean"):
        return latent_dist.mean
    return latent_dist


def _supported_call_kwargs(callable_obj, kwargs):
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


class Flux2KleinSRArtist(nn.Module):
    backend_name = "flux2_klein"

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
        self.vae_scale_factor = 16
        self.vae_latent_channels = int(_cfg(config, "model.vae_latent_channels", 32))
        self.latent_channels = int(_cfg(config, "model.latent_channels", 128))
        self.context_dim = int(_cfg(config, "model.context_dim", 15360))
        self.image_token_dim = int(_cfg(config, "model.image_token_dim", self.latent_channels))
        self.latent_mean = None
        self.latent_std = None

        self._load_flux2_modules()
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

    def _load_flux2_modules(self):
        try:
            from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "FLUX.2-klein support requires a diffusers version with "
                "AutoencoderKLFlux2, Flux2KleinPipeline, and Flux2Transformer2DModel."
            ) from exc

        vae = AutoencoderKLFlux2.from_pretrained(
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
            self.transformer = Flux2Transformer2DModel.from_pretrained(
                self.flux_model_path,
                subfolder="transformer",
                torch_dtype=self.weight_dtype,
            )
        finally:
            if hf_ds_config is not None:
                _clear_hf_deepspeed_config()
                hf_ds_config = None

        self.text_pipeline = Flux2KleinPipeline.from_pretrained(
            self.flux_model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.text_encoder_dtype,
        )
        if self.text_encoder_device and self.text_encoder_device.lower() != "cpu":
            self.text_pipeline.to(self.text_encoder_device)

        self.vae.requires_grad_(False)
        self.vae.eval()
        for module_name in ("text_encoder", "text_encoder_2"):
            module = getattr(self.text_pipeline, module_name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def _infer_dimensions(self):
        vae_config = getattr(self.vae, "config", None)
        if hasattr(vae_config, "latent_channels"):
            self.vae_latent_channels = int(vae_config.latent_channels)
        transformer_config = getattr(self.transformer, "config", None)
        if hasattr(transformer_config, "in_channels"):
            self.latent_channels = int(transformer_config.in_channels)
            self.image_token_dim = self.latent_channels
        for attr in ("joint_attention_dim", "context_dim"):
            if hasattr(transformer_config, attr):
                self.context_dim = int(getattr(transformer_config, attr))
                break

        vae_bn = getattr(self.vae, "bn", None)
        if vae_bn is not None and hasattr(vae_bn, "running_mean") and hasattr(vae_bn, "running_var"):
            eps = float(getattr(vae_config, "batch_norm_eps", 1e-5))
            self.latent_mean = vae_bn.running_mean.detach().float().view(1, -1, 1, 1)
            self.latent_std = torch.sqrt(vae_bn.running_var.detach().float().view(1, -1, 1, 1) + eps)
        else:
            latent_mean = getattr(vae_config, "latents_mean", None)
            latent_std = getattr(vae_config, "latents_std", None)
            if latent_mean is not None and latent_std is not None:
                self.latent_mean = torch.tensor(latent_mean, dtype=torch.float32).view(1, -1, 1, 1)
                self.latent_std = torch.tensor(latent_std, dtype=torch.float32).view(1, -1, 1, 1)

    def _build_condition_modules(self):
        condition_dropout = float(_cfg(self.config, "condition.condition_dropout", 0.0))
        deg_token_count = int(_cfg(self.config, "condition.deg_token_count", 0))
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
            num_tokens=int(_cfg(self.config, "condition.lr_token_count", 8)),
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

    def _resolve_lora_targets(self):
        configured = _cfg(self.config, "model.lora_target_modules", None)
        if configured:
            return configured

        suffixes = (
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
        )
        targets = []
        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in suffixes):
                targets.append(name)
        if not targets:
            raise RuntimeError(
                "Could not infer FLUX.2-klein LoRA target modules. "
                "Set model.lora_target_modules in the config after inspecting transformer.named_modules()."
            )
        return targets

    def _apply_lora(self):
        if not self.use_lora:
            return
        try:
            from peft import LoraConfig, PeftModel
        except ModuleNotFoundError as exc:
            raise ImportError("FLUX.2-klein LoRA training requires peft.") from exc

        rank = int(_cfg(self.config, "model.lora_rank", 8))
        alpha = int(_cfg(self.config, "model.lora_alpha", rank))
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=self._resolve_lora_targets(),
            init_lora_weights="gaussian",
        )
        self.transformer = PeftModel(self.transformer, lora_config, adapter_name="flux2_klein_adapter")

    def _apply_train_strategy(self):
        self.transformer.requires_grad_(False)
        if not bool(_cfg(self.config, "training.freeze_flux_transformer", True)):
            self.transformer.requires_grad_(True)
        self._apply_lora()

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

        if bool(_cfg(self.config, "model.gradient_checkpointing", False)):
            if hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()

    def _normalize_latents(self, latents):
        if self.latent_mean is None or self.latent_std is None:
            return latents
        mean = self.latent_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.latent_std.to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std

    def _denormalize_latents(self, latents):
        if self.latent_mean is None or self.latent_std is None:
            return latents
        mean = self.latent_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.latent_std.to(device=latents.device, dtype=latents.dtype)
        return latents * std + mean

    @torch.no_grad()
    def encode_images(self, images, sample=True):
        vae_device = _module_device(self.vae)
        vae_dtype = _module_dtype(self.vae, self.vae_dtype)
        encoded = self.vae.encode(images.to(device=vae_device, dtype=vae_dtype))
        latents = _retrieve_latents(encoded, sample=sample)
        latents = _patchify_latents(latents)
        latents = self._normalize_latents(latents)
        return latents.to(dtype=self.weight_dtype)

    @torch.no_grad()
    def decode_latents(self, latents):
        vae_device = _module_device(self.vae)
        vae_dtype = _module_dtype(self.vae, self.vae_dtype)
        latents = self._denormalize_latents(latents)
        latents = _unpatchify_latents(latents, self.vae_latent_channels)
        return self.vae.decode(latents.to(device=vae_device, dtype=vae_dtype), return_dict=False)[0]

    @torch.no_grad()
    def encode_prompts(self, prompts, device=None, dtype=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        if self.text_encoder_device.lower() == "cpu" and hasattr(self.text_pipeline, "to"):
            self.text_pipeline.to("cpu")
        encode_kwargs = {
            "prompt": prompts,
            "prompt_2": None,
            "device": self.text_encoder_device if self.text_encoder_device.lower() != "cpu" else "cpu",
            "num_images_per_prompt": 1,
        }
        if self.max_prompt_sequence_length > 0:
            encode_kwargs["max_sequence_length"] = self.max_prompt_sequence_length
        encode_kwargs = _supported_call_kwargs(self.text_pipeline.encode_prompt, encode_kwargs)
        try:
            result = self.text_pipeline.encode_prompt(**encode_kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            encode_kwargs.pop("max_sequence_length", None)
            encode_kwargs.pop("prompt_2", None)
            result = self.text_pipeline.encode_prompt(**encode_kwargs)
        if isinstance(result, dict):
            prompt_embeds = result.get("prompt_embeds")
            if prompt_embeds is None:
                prompt_embeds = result.get("encoder_hidden_states")
            text_ids = result.get("text_ids")
            if text_ids is None:
                text_ids = result.get("txt_ids")
            pooled_prompt_embeds = result.get("pooled_prompt_embeds")
        elif len(result) == 2:
            prompt_embeds, text_ids = result
            pooled_prompt_embeds = None
        else:
            prompt_embeds = result[0]
            pooled_prompt_embeds = result[1]
            text_ids = result[-1]
        if prompt_embeds is None or text_ids is None:
            raise RuntimeError("Flux2KleinPipeline.encode_prompt did not return prompt embeddings and text ids.")

        device = device or _module_device(self.transformer)
        dtype = dtype or self.weight_dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if text_ids.ndim == 2:
            text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)
        text_ids = text_ids.to(device=device, dtype=dtype)
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = prompt_embeds.new_zeros(prompt_embeds.shape[0], 0)
        else:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def extract_visual_tokens(self, lq_up):
        return None

    def _extra_text_ids(self, batch_size, token_count, device, dtype, token_type=0):
        if token_count <= 0:
            return torch.zeros(batch_size, 0, 4, device=device, dtype=dtype)
        ids = torch.zeros(batch_size, token_count, 4, device=device, dtype=dtype)
        ids[..., 0] = token_type
        ids[..., 3] = torch.arange(token_count, device=device, dtype=dtype)[None, :]
        return ids

    def build_context(
        self,
        prompt_embeds,
        text_ids=None,
        degradation_vector=None,
        z_lr=None,
        dino_tokens=None,
        lr_cond_mode="latent_adapter",
    ):
        batch_size = prompt_embeds.shape[0]
        context = [prompt_embeds]
        ids = [
            text_ids
            if text_ids is not None
            else self._extra_text_ids(batch_size, prompt_embeds.shape[1], prompt_embeds.device, prompt_embeds.dtype)
        ]

        if self.use_degradation_vector and degradation_vector is not None:
            deg_tokens = self.degradation_encoder(degradation_vector.to(prompt_embeds.device))
            if deg_tokens.shape[1] > 0:
                context.append(deg_tokens.to(dtype=prompt_embeds.dtype))
                ids.append(
                    self._extra_text_ids(
                        batch_size,
                        deg_tokens.shape[1],
                        prompt_embeds.device,
                        prompt_embeds.dtype,
                        token_type=1,
                    )
                )

        if self.use_visual_semantic_tokens and dino_tokens is not None:
            visual_tokens = self.visual_condition_adapter(dino_tokens)
            if visual_tokens is not None and visual_tokens.shape[1] > 0:
                context.append(visual_tokens.to(dtype=prompt_embeds.dtype))
                ids.append(
                    self._extra_text_ids(
                        batch_size,
                        visual_tokens.shape[1],
                        prompt_embeds.device,
                        prompt_embeds.dtype,
                        token_type=2,
                    )
                )

        if lr_cond_mode == "latent_adapter" and z_lr is not None:
            lr_tokens = self.lr_condition_encoder(z_lr.to(prompt_embeds.device), mode="latent_adapter")
            if lr_tokens.shape[1] > 0:
                context.append(lr_tokens.to(dtype=prompt_embeds.dtype))
                ids.append(
                    self._extra_text_ids(
                        batch_size,
                        lr_tokens.shape[1],
                        prompt_embeds.device,
                        prompt_embeds.dtype,
                        token_type=3,
                    )
                )

        return torch.cat(context, dim=1), torch.cat(ids, dim=1)

    def forward(
        self,
        z_t,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds=None,
        text_ids=None,
        degradation_vector=None,
        z_lr=None,
        dino_tokens=None,
        lr_cond_mode=None,
    ):
        lr_cond_mode = lr_cond_mode or self.lr_cond_mode
        bsz, channels, height, width = z_t.shape
        hidden_states = _flatten_image_tokens(z_t)
        context, txt_ids = self.build_context(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            degradation_vector=degradation_vector,
            z_lr=z_lr,
            dino_tokens=dino_tokens,
            lr_cond_mode=lr_cond_mode,
        )

        img_ids = _latent_image_ids(bsz, height, width, z_t.device, hidden_states.dtype)
        lr_token_count = 0
        if lr_cond_mode == "latent_concat":
            if z_lr is None:
                raise ValueError("z_lr is required when lr_cond_mode='latent_concat'")
            lr_tokens = self.lr_condition_encoder(z_lr.to(z_t.device), mode="latent_concat").to(dtype=hidden_states.dtype)
            lr_token_count = lr_tokens.shape[1]
            hidden_states = torch.cat([lr_tokens, hidden_states], dim=1)
            lr_img_ids = self._extra_text_ids(bsz, lr_token_count, z_t.device, hidden_states.dtype, token_type=4)
            img_ids = torch.cat([lr_img_ids, img_ids], dim=1)

        flux_timestep = convert_sigma_to_flux_timestep(timestep.to(z_t.device), self.timestep_mode).to(dtype=hidden_states.dtype)
        model_out = self.transformer(
            hidden_states=hidden_states,
            timestep=flux_timestep,
            guidance=None,
            encoder_hidden_states=context,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        packed_pred = model_out[0] if isinstance(model_out, (tuple, list)) else model_out.sample
        if lr_token_count:
            packed_pred = packed_pred[:, lr_token_count:]
        return _unflatten_image_tokens(packed_pred, height, width).to(dtype=z_t.dtype)

    def save_trainable(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"flux_backend": self.backend_name}
        with (output_dir / "rg_flux_checkpoint_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        if self.use_lora and hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(output_dir / "flux2_klein_adapter")
            lora_state = {
                name: param.detach().cpu()
                for name, param in self.transformer.state_dict().items()
                if "lora" in name.lower()
            }
            torch.save(lora_state, output_dir / "flux2_klein_lora_state.pt")
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
        meta_path = checkpoint_dir / "rg_flux_checkpoint_meta.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            backend = str(metadata.get("flux_backend", "")).lower()
            if backend and backend != self.backend_name:
                raise RuntimeError(f"Checkpoint backend '{backend}' is incompatible with '{self.backend_name}'.")
        if (checkpoint_dir / "flux_lora_state.pt").exists() and not (checkpoint_dir / "flux2_klein_lora_state.pt").exists():
            raise RuntimeError("This looks like a FLUX.1 checkpoint. Use a FLUX.2-klein adapter checkpoint instead.")

        adapter_dir = checkpoint_dir / "flux2_klein_adapter"
        lora_state_path = checkpoint_dir / "flux2_klein_lora_state.pt"
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
