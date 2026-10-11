"""Additive, zero-gated original-LR and round conditioning for refinement."""

from __future__ import annotations

import math
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
        anchor_mode: str = "legacy",
        attention_dim: int = 128,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.context_dim = int(context_dim)
        self.anchor_tokens = max(int(anchor_tokens), 1)
        self.anchor_mode = str(anchor_mode)
        self.attention_dim = int(attention_dim)
        if self.anchor_mode not in {"legacy", "spatial"}:
            raise ValueError(f"Unsupported anchor_mode: {self.anchor_mode}")
        if self.attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if self.anchor_mode == "spatial":
            self.grid_size = math.isqrt(self.anchor_tokens)
            if self.grid_size ** 2 != self.anchor_tokens:
                raise ValueError("Spatial anchor_tokens must form a square 2D grid")
            self.anchor_query = nn.Linear(self.context_dim, self.attention_dim, bias=False)
            self.anchor_key = nn.Linear(self.latent_channels, self.attention_dim, bias=False)
            self.anchor_position = nn.Linear(2, self.attention_dim, bias=False)
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

    def _spatial_anchor_residual(self, z_anchor_lr, prompt_embeds):
        if z_anchor_lr.ndim != 4 or z_anchor_lr.shape[1] != self.latent_channels:
            raise ValueError("Spatial anchor must be BCHW with the configured latent channels")
        tokens = F.adaptive_avg_pool2d(z_anchor_lr, (self.grid_size, self.grid_size))
        tokens = tokens.flatten(2).transpose(1, 2).to(dtype=self.anchor_proj.weight.dtype)
        axis = torch.linspace(-1, 1, self.grid_size, device=tokens.device, dtype=tokens.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((yy, xx), dim=-1).reshape(1, self.anchor_tokens, 2)
        keys = self.anchor_key(tokens) + self.anchor_position(positions)
        queries = self.anchor_query(prompt_embeds.to(dtype=self.anchor_query.weight.dtype))
        values = self.anchor_proj(tokens)
        # Small cross-attention; keep probabilities/reduction in FP32 even
        # when the backbone's autocast uses BF16.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            scores = queries.float() @ keys.float().transpose(1, 2) / math.sqrt(self.attention_dim)
            return scores.softmax(dim=-1) @ values.float()

    def checkpoint_metadata(self):
        return {"anchor_mode": self.anchor_mode, "anchor_tokens": self.anchor_tokens,
                "attention_dim": self.attention_dim, "max_round": self.max_round}

    def validate_checkpoint_metadata(self, metadata, has_refinement_weights):
        if metadata is None:
            if has_refinement_weights and self.anchor_mode != "legacy":
                raise RuntimeError("A legacy refinement checkpoint requires anchor_mode: legacy; "
                                   "initialize spatial refinement from F0 instead")
            return
        expected = self.checkpoint_metadata()
        if metadata != expected:
            raise RuntimeError(f"Refinement anchor configuration mismatch: checkpoint={metadata}, model={expected}")

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
            anchor = (self._spatial_anchor_residual(z_anchor_lr, prompt_embeds)
                      if self.anchor_mode == "spatial"
                      else self._anchor_residual(z_anchor_lr, prompt_embeds.shape[1])).to(
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
