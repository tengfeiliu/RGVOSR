"""Additive, zero-gated original-LR and round conditioning for refinement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRRefinementConditionAdapter(nn.Module):
    """Modulate existing prompt context without appending extra attention tokens.

    A zero gate gives an exact no-op to the incoming prompt embeddings.  This is
    important for comparing a new refinement checkpoint against its old model;
    appending zero tokens would still alter softmax normalization.
    """

    def __init__(
        self,
        latent_channels: int,
        context_dim: int,
        anchor_tokens: int = 8,
        dropout: float = 0.0,
        max_round: int = 16,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.context_dim = int(context_dim)
        self.anchor_tokens = max(int(anchor_tokens), 1)
        self.anchor_proj = nn.Linear(self.latent_channels, self.context_dim)
        self.max_round = max(int(max_round), 2)
        self.round_embed = nn.Embedding(self.max_round + 1, self.context_dim)
        self.anchor_gate = nn.Parameter(torch.zeros(()))
        self.round_gate = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.round_embed.weight, std=0.02)

    def _anchor_residual(self, z_anchor_lr: torch.Tensor, prompt_length: int) -> torch.Tensor:
        if z_anchor_lr.ndim != 4:
            raise ValueError(f"Expected BCHW anchor latent, got {tuple(z_anchor_lr.shape)}")
        if z_anchor_lr.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} anchor channels, got {z_anchor_lr.shape[1]}"
            )
        tokens = z_anchor_lr.flatten(2).transpose(1, 2)
        tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.anchor_tokens).transpose(1, 2)
        tokens = self.anchor_proj(tokens.to(dtype=self.anchor_proj.weight.dtype))
        return F.adaptive_avg_pool1d(tokens.transpose(1, 2), prompt_length).transpose(1, 2)

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        z_anchor_lr: torch.Tensor | None = None,
        refinement_round: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        if prompt_embeds.ndim != 3:
            raise ValueError(f"Expected BLC prompt embeddings, got {tuple(prompt_embeds.shape)}")
        output = prompt_embeds
        if z_anchor_lr is not None:
            anchor = self._anchor_residual(z_anchor_lr, prompt_embeds.shape[1]).to(
                device=prompt_embeds.device, dtype=prompt_embeds.dtype
            )
            output = output + self.anchor_gate.to(dtype=output.dtype) * self.dropout(anchor)
        if refinement_round is not None:
            if not torch.is_tensor(refinement_round):
                refinement_round = torch.full(
                    (prompt_embeds.shape[0],), int(refinement_round), device=prompt_embeds.device
                )
            refinement_round = refinement_round.to(device=prompt_embeds.device, dtype=torch.long).reshape(-1)
            if refinement_round.shape[0] == 1 and prompt_embeds.shape[0] > 1:
                refinement_round = refinement_round.expand(prompt_embeds.shape[0])
            if refinement_round.shape[0] != prompt_embeds.shape[0]:
                raise ValueError("refinement_round batch size must match prompt embeddings")
            if torch.any(refinement_round < 0) or torch.any(refinement_round >= self.round_embed.num_embeddings):
                raise ValueError(f"refinement_round must be in [0, {self.max_round}]")
            round_residual = self.round_embed(refinement_round).unsqueeze(1).expand_as(output)
            output = output + self.round_gate.to(dtype=output.dtype) * self.dropout(round_residual)
        return output
