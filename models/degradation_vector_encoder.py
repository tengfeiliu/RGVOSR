import torch
import torch.nn as nn


class DegradationVectorEncoder(nn.Module):
    def __init__(
        self,
        in_dim=8,
        hidden_dim=512,
        context_dim=4096,
        num_tokens=4,
        dropout=0.0,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.context_dim = int(context_dim)
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(self.num_tokens, 1) * self.context_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max(self.num_tokens, 1), self.context_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 1, self.context_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.type_embed, std=0.02)

    def forward(self, degradation_vector):
        batch_size = degradation_vector.shape[0]
        if self.num_tokens == 0:
            return degradation_vector.new_zeros(batch_size, 0, self.context_dim)

        x = degradation_vector.to(dtype=self.net[0].weight.dtype)
        tokens = self.net(x).view(batch_size, self.num_tokens, self.context_dim)
        tokens = tokens + self.pos_embed[:, : self.num_tokens] + self.type_embed
        return self.dropout(tokens)
