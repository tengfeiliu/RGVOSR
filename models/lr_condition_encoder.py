import torch
import torch.nn as nn
import torch.nn.functional as F


class LRConditionEncoder(nn.Module):
    def __init__(
        self,
        latent_channels=16,
        context_dim=4096,
        num_tokens=64,
        mode="latent_adapter",
        image_token_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.context_dim = int(context_dim)
        self.num_tokens = int(num_tokens)
        self.mode = mode
        self.image_token_dim = int(image_token_dim or latent_channels * 4)
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if mode not in {"latent_adapter", "latent_concat"}:
            raise ValueError(f"Unsupported LR condition mode: {mode}")

        self.adapter_proj = nn.Linear(self.latent_channels, self.context_dim)
        self.concat_proj = nn.Linear(self.latent_channels, self.image_token_dim)
        self.adapter_type_embed = nn.Parameter(torch.zeros(1, 1, self.context_dim))
        self.concat_type_embed = nn.Parameter(torch.zeros(1, 1, self.image_token_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.adapter_type_embed, std=0.02)
        nn.init.normal_(self.concat_type_embed, std=0.02)

    def _pool_tokens(self, z_lr):
        bsz, channels, height, width = z_lr.shape
        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}")
        tokens = z_lr.flatten(2).transpose(1, 2)
        if self.num_tokens == 0:
            return tokens[:, :0]
        tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.num_tokens).transpose(1, 2)
        return tokens

    def forward(self, z_lr, mode=None):
        mode = mode or self.mode
        tokens = self._pool_tokens(z_lr)
        if mode == "latent_adapter":
            tokens = self.adapter_proj(tokens.to(dtype=self.adapter_proj.weight.dtype))
            return self.dropout(tokens + self.adapter_type_embed)
        if mode == "latent_concat":
            tokens = self.concat_proj(tokens.to(dtype=self.concat_proj.weight.dtype))
            return self.dropout(tokens + self.concat_type_embed)
        raise ValueError(f"Unsupported LR condition mode: {mode}")
