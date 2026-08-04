import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def route_logits(logits, mode="soft", top_k=2, temperature=1.0):
    temperature = max(float(temperature), 1e-6)
    scaled = logits / temperature
    if mode == "soft":
        return torch.softmax(scaled, dim=-1)
    if mode != "topk":
        raise ValueError(f"Unsupported routing mode: {mode}")
    top_k = max(1, min(int(top_k), logits.shape[-1]))
    values, indexes = torch.topk(scaled, k=top_k, dim=-1)
    top_alpha = torch.softmax(values, dim=-1)
    alpha = torch.zeros_like(logits)
    alpha.scatter_(-1, indexes, top_alpha)
    return alpha


def teacher_router_mix(progress, start_ratio=0.15, end_ratio=0.35, enabled=True):
    start_ratio = float(start_ratio)
    end_ratio = float(end_ratio)
    if not 0.0 <= start_ratio <= end_ratio <= 1.0:
        raise ValueError(
            "Teacher schedule must satisfy 0 <= start_ratio <= end_ratio <= 1"
        )
    if not enabled:
        return 1.0
    progress = min(max(float(progress), 0.0), 1.0)
    if progress <= start_ratio:
        return 0.0
    if progress >= end_ratio:
        return 1.0
    return (progress - start_ratio) / max(end_ratio - start_ratio, 1e-8)


def blend_teacher_routing(router_alpha, teacher_target, router_mix, valid_mask=None):
    """Blend teacher and learned routes; missing-condition rows always use Router."""
    router_mix = min(max(float(router_mix), 0.0), 1.0)
    teacher_alpha = (1.0 - router_mix) * teacher_target + router_mix * router_alpha
    if valid_mask is None:
        valid = torch.ones(
            router_alpha.shape[0],
            device=router_alpha.device,
            dtype=router_alpha.dtype,
        )
    else:
        valid = (valid_mask.to(router_alpha.device).sum(dim=-1) > 0).to(router_alpha.dtype)
    strength = valid.unsqueeze(-1)
    return strength * teacher_alpha + (1.0 - strength) * router_alpha


class SharedRoutedMoELoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=8, num_routed_experts=4, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("SharedRoutedMoELoRALinear requires an nn.Linear base layer.")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.num_routed_experts = int(num_routed_experts)
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if self.num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be positive.")
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))

        self.shared_lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.shared_lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.routed_lora_A = nn.Parameter(torch.empty(self.num_routed_experts, self.rank, self.in_features))
        self.routed_lora_B = nn.Parameter(torch.empty(self.num_routed_experts, self.out_features, self.rank))
        self._routing_alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.shared_lora_A, std=0.01)
        nn.init.zeros_(self.shared_lora_B)
        nn.init.normal_(self.routed_lora_A, std=0.01)
        nn.init.zeros_(self.routed_lora_B)

    @torch.no_grad()
    def initialize_routed_residuals(self, perturb_scale=0.01):
        """Initialize routed experts as zero-output residuals around the shared A basis."""
        perturb_scale = max(float(perturb_scale), 0.0)
        source = self.shared_lora_A.detach()
        noise_scale = max(source.float().std(unbiased=False).item(), 1e-6) * perturb_scale
        for expert_idx in range(self.num_routed_experts):
            noise = torch.randn_like(source) * noise_scale
            self.routed_lora_A[expert_idx].copy_(source + noise)
        self.routed_lora_B.zero_()

    def set_routing(self, alpha):
        if alpha is None:
            self._routing_alpha = None
            return
        if alpha.ndim != 2 or alpha.shape[-1] != self.num_routed_experts:
            raise ValueError(
                f"Expected routing alpha [B, {self.num_routed_experts}], got {tuple(alpha.shape)}"
            )
        self._routing_alpha = alpha

    def clear_routing(self):
        self._routing_alpha = None

    def set_shared_trainable(self, trainable):
        self.shared_lora_A.requires_grad_(bool(trainable))
        self.shared_lora_B.requires_grad_(bool(trainable))

    def _shared_lora(self, x):
        hidden = F.linear(self.dropout(x), self.shared_lora_A)
        return F.linear(hidden, self.shared_lora_B) * self.scaling

    def _routed_lora(self, x):
        if self._routing_alpha is None:
            alpha = x.new_full((x.shape[0], self.num_routed_experts), 1.0 / self.num_routed_experts)
        else:
            alpha = self._routing_alpha.to(device=x.device, dtype=x.dtype)
        x_drop = self.dropout(x)
        routed = x.new_zeros(*x.shape[:-1], self.out_features)
        for expert_idx in range(self.num_routed_experts):
            hidden = F.linear(x_drop, self.routed_lora_A[expert_idx])
            expert_out = F.linear(hidden, self.routed_lora_B[expert_idx]) * self.scaling
            weight = alpha[:, expert_idx]
            while weight.ndim < expert_out.ndim:
                weight = weight.unsqueeze(-1)
            routed = routed + expert_out * weight
        return routed

    def forward(self, x):
        return self.base_layer(x) + self._shared_lora(x) + self._routed_lora(x)


class ProfileLatentRouter(nn.Module):
    def __init__(
        self,
        prompt_dim,
        latent_channels,
        num_experts=4,
        hidden_dim=1024,
        latent_branch="stat_conv",
        conv_dim=None,
        prototype_scale=1.0,
        input_mode="prompt_lr",
        condition_dim=8,
        timestep_dim=32,
        ema_decay=0.99,
    ):
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.latent_channels = int(latent_channels)
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.input_mode = str(input_mode)
        valid_input_modes = {
            "prompt_lr",
            "prompt_only",
            "condition8",
            "condition8_timestep",
        }
        if self.input_mode not in valid_input_modes:
            raise ValueError(
                f"Unsupported router input_mode: {self.input_mode}; "
                f"expected one of {sorted(valid_input_modes)}"
            )
        self.condition_dim = int(condition_dim)
        self.timestep_dim = int(timestep_dim)
        self.latent_branch = str(latent_branch)
        if self.latent_branch not in {"stat_only", "conv_only", "stat_conv"}:
            raise ValueError(f"Unsupported latent_branch: {latent_branch}")

        conv_dim = int(conv_dim or max(64, min(self.hidden_dim // 2, 512)))
        self.text_proj = None
        self.stat_proj = None
        self.conv_encoder = None
        self.condition_proj = None
        self.timestep_proj = None
        if self.input_mode in {"prompt_lr", "prompt_only"}:
            self.text_proj = nn.Sequential(
                nn.LayerNorm(self.prompt_dim),
                nn.Linear(self.prompt_dim, self.hidden_dim),
                nn.SiLU(),
            )
        if self.input_mode == "prompt_lr" and self.latent_branch in {"stat_only", "stat_conv"}:
            stat_dim = self.latent_channels * 4
            self.stat_proj = nn.Sequential(
                nn.LayerNorm(stat_dim),
                nn.Linear(stat_dim, self.hidden_dim),
                nn.SiLU(),
            )
        if self.input_mode == "prompt_lr" and self.latent_branch in {"conv_only", "stat_conv"}:
            self.conv_encoder = nn.Sequential(
                nn.Conv2d(self.latent_channels, conv_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(conv_dim, self.hidden_dim),
                nn.SiLU(),
            )
        if self.input_mode in {"condition8", "condition8_timestep"}:
            self.condition_proj = nn.Sequential(
                nn.LayerNorm(self.condition_dim * 2 + 1),
                nn.Linear(self.condition_dim * 2 + 1, self.hidden_dim),
                nn.SiLU(),
            )
        if self.input_mode == "condition8_timestep":
            self.timestep_proj = nn.Sequential(
                nn.LayerNorm(self.timestep_dim),
                nn.Linear(self.timestep_dim, self.hidden_dim),
                nn.SiLU(),
            )
        if self.input_mode == "prompt_lr" and self.latent_branch == "stat_conv":
            fusion_in = self.hidden_dim * 3
        elif self.input_mode == "prompt_lr":
            fusion_in = self.hidden_dim * 2
        elif self.input_mode == "condition8_timestep":
            fusion_in = self.hidden_dim * 2
        else:
            fusion_in = self.hidden_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, self.hidden_dim),
            nn.SiLU(),
        )
        self.logit_head = nn.Linear(self.hidden_dim, self.num_experts)
        self.prototypes = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.prototype_scale = float(prototype_scale)
        self.ema_decay = float(ema_decay)
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        self.register_buffer(
            "ema_dispatch_usage",
            torch.full((self.num_experts,), 1.0 / self.num_experts, dtype=torch.float32),
        )
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prototypes, std=0.02)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

    def text_features(self, prompt_embeds):
        pooled = prompt_embeds.float().mean(dim=1)
        return self.text_proj(pooled.to(dtype=self.text_proj[1].weight.dtype))

    def latent_stat_features(self, z_lr):
        z = z_lr.float()
        stats = torch.cat(
            [
                z.mean(dim=(2, 3)),
                z.std(dim=(2, 3), unbiased=False),
                z.amin(dim=(2, 3)),
                z.amax(dim=(2, 3)),
            ],
            dim=-1,
        )
        return self.stat_proj(stats.to(dtype=self.stat_proj[1].weight.dtype))

    def latent_conv_features(self, z_lr):
        return self.conv_encoder(z_lr.to(dtype=self.conv_encoder[0].weight.dtype))

    def condition_features(self, condition, condition_mask=None, condition_confidence=None):
        if condition is None:
            raise ValueError(f"router_condition is required for input_mode={self.input_mode!r}")
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected router_condition [B, {self.condition_dim}], got {tuple(condition.shape)}"
            )
        condition = condition.float()
        if not torch.isfinite(condition).all() or (condition < 0.0).any() or (condition > 1.0).any():
            raise ValueError("router_condition must contain finite values in [0, 1]")
        if condition_mask is None:
            condition_mask = torch.ones_like(condition)
        else:
            condition_mask = condition_mask.float()
            if condition_mask.shape != condition.shape:
                raise ValueError("router_condition_mask must have the same shape as router_condition")
            if (
                not torch.isfinite(condition_mask).all()
                or (condition_mask < 0.0).any()
                or (condition_mask > 1.0).any()
            ):
                raise ValueError("router_condition_mask must contain finite values in [0, 1]")
        if condition_confidence is None:
            condition_confidence = condition_mask.mean(dim=-1)
        condition_confidence = condition_confidence.float().reshape(condition.shape[0], 1)
        if (
            not torch.isfinite(condition_confidence).all()
            or (condition_confidence < 0.0).any()
            or (condition_confidence > 1.0).any()
        ):
            raise ValueError("router_condition_confidence must contain finite values in [0, 1]")
        encoded = torch.cat(
            [condition * condition_mask, condition_mask, condition_confidence],
            dim=-1,
        )
        return self.condition_proj(encoded.to(dtype=self.condition_proj[1].weight.dtype))

    def timestep_features(self, timestep):
        if timestep is None:
            raise ValueError("timestep is required for input_mode='condition8_timestep'")
        timestep = timestep.float().reshape(-1)
        if (
            not torch.isfinite(timestep).all()
            or (timestep < -1e-6).any()
            or (timestep > 1.0 + 1e-6).any()
        ):
            raise ValueError("Router timestep must be the raw flow-matching sigma in [0, 1]")
        timestep = timestep.clamp(0.0, 1.0)
        half = max(self.timestep_dim // 2, 1)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=timestep.dtype)
            / max(half - 1, 1)
        )
        angles = timestep[:, None] * frequencies[None, :] * (2.0 * math.pi)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.timestep_dim:
            embedding = F.pad(embedding, (0, self.timestep_dim - embedding.shape[-1]))
        elif embedding.shape[-1] > self.timestep_dim:
            embedding = embedding[:, : self.timestep_dim]
        return self.timestep_proj(embedding.to(dtype=self.timestep_proj[1].weight.dtype))

    def router_features(
        self,
        prompt_embeds=None,
        z_lr=None,
        router_condition=None,
        router_condition_mask=None,
        router_condition_confidence=None,
        timestep=None,
    ):
        if self.input_mode in {"condition8", "condition8_timestep"}:
            pieces = [
                self.condition_features(
                    router_condition,
                    condition_mask=router_condition_mask,
                    condition_confidence=router_condition_confidence,
                )
            ]
            if self.input_mode == "condition8_timestep":
                pieces.append(self.timestep_features(timestep))
            return self.fusion(torch.cat(pieces, dim=-1))
        if prompt_embeds is None:
            raise ValueError(f"prompt_embeds is required for input_mode={self.input_mode!r}")
        pieces = [self.text_features(prompt_embeds)]
        if self.input_mode == "prompt_lr":
            if z_lr is None:
                raise ValueError("z_lr is required for input_mode='prompt_lr'")
            if self.latent_branch in {"stat_only", "stat_conv"}:
                pieces.append(self.latent_stat_features(z_lr))
            if self.latent_branch in {"conv_only", "stat_conv"}:
                pieces.append(self.latent_conv_features(z_lr))
        return self.fusion(torch.cat(pieces, dim=-1))

    @torch.no_grad()
    def update_ema_usage(self, dispatch_alpha):
        usage = dispatch_alpha.detach().float().mean(dim=0)
        usage = usage / usage.sum().clamp_min(1e-8)
        self.ema_dispatch_usage.mul_(self.ema_decay).add_(usage, alpha=1.0 - self.ema_decay)
        self.ema_dispatch_usage.div_(self.ema_dispatch_usage.sum().clamp_min(1e-8))
        self.ema_update_count.add_(1)

    def forward(
        self,
        prompt_embeds=None,
        z_lr=None,
        router_condition=None,
        router_condition_mask=None,
        router_condition_confidence=None,
        timestep=None,
        routing_mode="soft",
        top_k=2,
        temperature=1.0,
        noise_std=0.0,
        update_ema=False,
        return_details=False,
    ):
        features = self.router_features(
            prompt_embeds=prompt_embeds,
            z_lr=z_lr,
            router_condition=router_condition,
            router_condition_mask=router_condition_mask,
            router_condition_confidence=router_condition_confidence,
            timestep=timestep,
        )
        head_logits = self.logit_head(features)
        proto_logits = F.linear(
            F.normalize(features.float(), dim=-1),
            F.normalize(self.prototypes.float(), dim=-1),
        ).to(dtype=head_logits.dtype)
        logits = head_logits + self.prototype_scale * proto_logits
        routed_logits = logits
        if self.training and float(noise_std) > 0.0:
            routed_logits = logits + torch.randn_like(logits) * float(noise_std)
        clean_dense_alpha = route_logits(logits, mode="soft", temperature=temperature)
        dispatch_dense_alpha = route_logits(routed_logits, mode="soft", temperature=temperature)
        alpha = route_logits(
            routed_logits,
            mode=routing_mode,
            top_k=top_k,
            temperature=temperature,
        )
        if update_ema:
            self.update_ema_usage(alpha)
        if return_details:
            return {
                "logits": logits,
                "routed_logits": routed_logits,
                "dense_alpha": clean_dense_alpha,
                "clean_dense_alpha": clean_dense_alpha,
                "dispatch_dense_alpha": dispatch_dense_alpha,
                "alpha": alpha,
                "features": features,
            }
        return logits, alpha, features


def iter_moe_lora_layers(module):
    for child in module.modules():
        if isinstance(child, SharedRoutedMoELoRALinear):
            yield child


@contextlib.contextmanager
def moe_routing(module, alpha):
    layers = list(iter_moe_lora_layers(module))
    for layer in layers:
        layer.set_routing(alpha)
    try:
        yield
    finally:
        for layer in layers:
            layer.clear_routing()


def set_shared_lora_trainable(module, trainable):
    for layer in iter_moe_lora_layers(module):
        layer.set_shared_trainable(trainable)


def add_moe_persistent_buffers(module, state):
    """Add resumable Router EMA buffers to a parameter-only MoE checkpoint state."""
    state = dict(state)
    for name, buffer in module.named_buffers():
        if name.startswith("moe_router.ema_"):
            state[name] = buffer.detach().cpu().clone()
    return state


def _maybe_gathered_parameters(parameters):
    parameters = list(parameters)
    if not parameters:
        return contextlib.nullcontext()
    try:
        from deepspeed import zero
    except Exception:
        return contextlib.nullcontext()
    gathered_parameters = getattr(zero, "GatheredParameters", None)
    if gathered_parameters is None:
        return contextlib.nullcontext()
    return gathered_parameters(parameters, modifier_rank=0)


def moe_diversity_loss(layers, probe_dim=16, min_effective_norm=1e-4):
    layers = list(layers)
    if not layers:
        return torch.tensor(0.0)
    parameters = []
    for layer in layers:
        parameters.extend([layer.routed_lora_A, layer.routed_lora_B])
    losses = []
    with _maybe_gathered_parameters(parameters):
        for layer in layers:
            if layer.routed_lora_A.ndim < 2 or layer.routed_lora_B.ndim < 2:
                continue
            if layer.routed_lora_A.numel() == 0 or layer.routed_lora_B.numel() == 0:
                continue
            # Compare effective expert functions B_e A_e on an evenly spaced,
            # deterministic input subspace. This is invariant to LoRA A/B
            # reparameterization and avoids materializing the full matrix.
            selected = torch.linspace(
                0,
                max(layer.in_features - 1, 0),
                steps=min(int(probe_dim), layer.in_features),
                device=layer.routed_lora_A.device,
            ).round().long()
            effective = torch.matmul(
                layer.routed_lora_B.float(),
                layer.routed_lora_A.index_select(-1, selected).float(),
            )
            effective = effective.flatten(1)
            norms = effective.norm(dim=-1)
            normalized = F.normalize(effective, dim=-1, eps=max(float(min_effective_norm), 1e-8))
            similarity = normalized @ normalized.t()
            active = norms > float(min_effective_norm)
            mask = torch.triu(
                active[:, None] & active[None, :],
                diagonal=1,
            )
            if mask.any():
                losses.append(similarity[mask].pow(2).mean())
            else:
                # Keep a zero-gradient connection for stable mixed-loss backward.
                losses.append(effective.sum() * 0.0)
    if not losses:
        return parameters[0].new_zeros(())
    return torch.stack(losses).mean()


def routing_entropy(alpha):
    safe = alpha.clamp_min(1e-8)
    return -(safe * safe.log()).sum(dim=-1).mean()


def routing_entropy_loss(alpha, encourage_high_entropy=True):
    entropy = routing_entropy(alpha)
    return -entropy if encourage_high_entropy else entropy


def routing_balance_loss(dense_alpha, ema_dispatch_usage=None):
    """Switch-style load balancing that remains meaningful for batch size one.

    The dispatch frequency is accumulated across steps in an EMA and detached;
    gradients flow through the current dense (pre-top-k) router probabilities.
    """
    probabilities = dense_alpha.float().mean(dim=0)
    if ema_dispatch_usage is None:
        usage = dense_alpha.detach().float().mean(dim=0)
    else:
        usage = ema_dispatch_usage.detach().float().to(probabilities.device)
    usage = usage / usage.sum().clamp_min(1e-8)
    return dense_alpha.shape[-1] * torch.sum(usage * probabilities)
