import contextlib

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
    ):
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.latent_channels = int(latent_channels)
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.latent_branch = str(latent_branch)
        if self.latent_branch not in {"stat_only", "conv_only", "stat_conv"}:
            raise ValueError(f"Unsupported latent_branch: {latent_branch}")

        conv_dim = int(conv_dim or max(64, min(self.hidden_dim // 2, 512)))
        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.prompt_dim),
            nn.Linear(self.prompt_dim, self.hidden_dim),
            nn.SiLU(),
        )
        stat_dim = self.latent_channels * 4
        self.stat_proj = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, conv_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(conv_dim, self.hidden_dim),
            nn.SiLU(),
        )
        if self.latent_branch == "stat_conv":
            fusion_in = self.hidden_dim * 3
        else:
            fusion_in = self.hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, self.hidden_dim),
            nn.SiLU(),
        )
        self.logit_head = nn.Linear(self.hidden_dim, self.num_experts)
        self.prototypes = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.prototype_scale = float(prototype_scale)
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

    def router_features(self, prompt_embeds, z_lr):
        text_feat = self.text_features(prompt_embeds)
        pieces = [text_feat]
        if self.latent_branch in {"stat_only", "stat_conv"}:
            pieces.append(self.latent_stat_features(z_lr))
        if self.latent_branch in {"conv_only", "stat_conv"}:
            pieces.append(self.latent_conv_features(z_lr))
        return self.fusion(torch.cat(pieces, dim=-1))

    def forward(self, prompt_embeds, z_lr, routing_mode="soft", top_k=2, temperature=1.0):
        features = self.router_features(prompt_embeds, z_lr)
        head_logits = self.logit_head(features)
        proto_logits = F.linear(
            F.normalize(features.float(), dim=-1),
            F.normalize(self.prototypes.float(), dim=-1),
        ).to(dtype=head_logits.dtype)
        logits = head_logits + self.prototype_scale * proto_logits
        alpha = route_logits(logits, mode=routing_mode, top_k=top_k, temperature=temperature)
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


def moe_diversity_loss(layers):
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
            a = F.normalize(layer.routed_lora_A.flatten(1).float(), dim=-1)
            b = F.normalize(layer.routed_lora_B.flatten(1).float(), dim=-1)
            sim_a = a @ a.t()
            sim_b = b @ b.t()
            mask = torch.triu(torch.ones_like(sim_a, dtype=torch.bool), diagonal=1)
            if mask.any():
                losses.append(sim_a[mask].pow(2).mean() + sim_b[mask].pow(2).mean())
    if not losses:
        return parameters[0].new_zeros(())
    return torch.stack(losses).mean()


def routing_entropy(alpha):
    safe = alpha.clamp_min(1e-8)
    return -(safe * safe.log()).sum(dim=-1).mean()


def routing_entropy_loss(alpha, encourage_high_entropy=True):
    entropy = routing_entropy(alpha)
    return -entropy if encourage_high_entropy else entropy


def routing_balance_loss(alpha):
    usage = alpha.float().mean(dim=0)
    return alpha.shape[-1] * usage.pow(2).sum()
