import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualConditionAdapter(nn.Module):
    def __init__(self, in_dim=768, context_dim=4096, num_tokens=64, dropout=0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.context_dim = int(context_dim)
        self.num_tokens = int(num_tokens)
        self.proj = nn.Linear(self.in_dim, self.context_dim)
        self.type_embed = nn.Parameter(torch.zeros(1, 1, self.context_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.type_embed, std=0.02)

    def forward(self, visual_features):
        if visual_features is None:
            return None
        if isinstance(visual_features, (list, tuple)):
            visual_features = torch.cat([item for item in visual_features if item is not None], dim=1)
        if visual_features.ndim == 4:
            visual_features = visual_features.flatten(2).transpose(1, 2)
        if visual_features.ndim != 3:
            raise ValueError("visual_features must be [B, L, C] or [B, C, H, W]")
        if visual_features.shape[-1] != self.in_dim:
            raise ValueError(f"Expected visual feature dim {self.in_dim}, got {visual_features.shape[-1]}")
        if self.num_tokens == 0:
            return visual_features.new_zeros(visual_features.shape[0], 0, self.context_dim)
        tokens = F.adaptive_avg_pool1d(visual_features.transpose(1, 2), self.num_tokens).transpose(1, 2)
        tokens = self.proj(tokens.to(dtype=self.proj.weight.dtype))
        return self.dropout(tokens + self.type_embed)
