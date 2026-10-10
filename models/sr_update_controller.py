"""Small RGB-only acceptance controller for real sampled SR proposals."""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class SRUpdateController(nn.Module):
    """Predict a global acceptance scalar and a smooth, shared RGB mask.

    All three images are aligned FP32 BCHW RGB in [-1, 1]. HR is deliberately
    absent from this interface. The spatial head operates at one eighth of the
    input resolution; bilinear interpolation supplies the full-size mask.
    Sigmoid heads start open with a small alpha so neither gate blocks learning.
    """

    def __init__(self, channels=32, max_round=16, init_alpha=0.1):
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, Integral) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if isinstance(max_round, bool) or not isinstance(max_round, Integral) or max_round < 1:
            raise ValueError("max_round must be a positive integer")
        if not math.isfinite(float(init_alpha)) or not 0 < float(init_alpha) < 1:
            raise ValueError("init_alpha must be finite and strictly between zero and one")
        self.channels = int(channels)
        self.max_round = int(max_round)
        self.init_alpha = float(init_alpha)
        self.checkpoint_metadata = {}
        self.encoder = nn.Sequential(
            nn.Conv2d(12, self.channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(self.channels, self.channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(self.channels, self.channels, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.round_embedding = nn.Embedding(self.max_round + 1, self.channels)
        self.alpha_head = nn.Linear(self.channels, 1)
        self.mask_head = nn.Conv2d(self.channels, 1, 3, padding=1)
        nn.init.normal_(self.round_embedding.weight, std=0.02)
        # Small nonzero head weights let image/round features learn immediately.
        nn.init.normal_(self.alpha_head.weight, std=0.01)
        nn.init.constant_(self.alpha_head.bias, math.log(self.init_alpha / (1 - self.init_alpha)))
        nn.init.normal_(self.mask_head.weight, std=0.01)
        nn.init.zeros_(self.mask_head.bias)

    @property
    def config(self):
        return {"channels": self.channels, "max_round": self.max_round, "init_alpha": self.init_alpha}

    def _round_indices(self, target_round, batch, device):
        if isinstance(target_round, Integral) and not isinstance(target_round, bool):
            indices = torch.full((batch,), int(target_round), device=device, dtype=torch.long)
        elif torch.is_tensor(target_round):
            if target_round.dtype == torch.bool or target_round.is_complex():
                raise TypeError("target_round must contain integer round indices")
            indices = target_round.to(device=device).reshape(-1)
            if indices.numel() == 1:
                indices = indices.expand(batch)
            if indices.numel() != batch:
                raise ValueError("target_round batch size must match the RGB images")
            if not torch.isfinite(indices).all() or not torch.equal(indices, indices.floor()):
                raise ValueError("target_round must contain finite integer round indices")
            indices = indices.to(dtype=torch.long)
        else:
            raise TypeError("target_round must be an integer or tensor")
        if torch.any(indices < 0) or torch.any(indices > self.max_round):
            raise ValueError(f"target_round must be in [0, {self.max_round}]")
        return indices

    def forward(self, current, original_lr_up, proposal, target_round):
        for name, image in (("current", current), ("original_lr_up", original_lr_up), ("proposal", proposal)):
            if not torch.is_tensor(image) or image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"{name} must be BCHW RGB")
            if image.dtype != torch.float32:
                raise TypeError(f"{name} must preserve FP32 RGB")
            if min(image.shape) <= 0:
                raise ValueError(f"{name} must have nonempty geometry")
            if image.shape != current.shape or image.device != current.device:
                raise ValueError("controller RGB inputs must have identical geometry and device")
            if not torch.isfinite(image).all() or torch.any(image < -1) or torch.any(image > 1):
                raise ValueError(f"{name} must be finite RGB in [-1, 1]")
        indices = self._round_indices(target_round, current.shape[0], current.device)
        with torch.autocast(device_type=current.device.type, enabled=False):
            inputs = torch.cat((current, original_lr_up, proposal, proposal - current), dim=1)
            features = self.encoder(inputs)
            features = features + self.round_embedding(indices)[:, :, None, None]
            alpha = self.alpha_head(features.mean(dim=(2, 3))).sigmoid()[:, :, None, None]
            low_resolution_mask = self.mask_head(features).sigmoid()
            mask = F.interpolate(low_resolution_mask, size=current.shape[-2:], mode="bilinear", align_corners=False)
            return {"alpha": alpha, "mask": mask, "weight": alpha * mask}

    def save(self, path, metadata=None):
        """Save portable FP32 weights, architecture, and caller provenance."""
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("checkpoint metadata must be a mapping")
            self.checkpoint_metadata = deepcopy(dict(metadata))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1,
            "config": self.config,
            "metadata": deepcopy(self.checkpoint_metadata),
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Restore the architecture and metadata without loading a generator."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported SRUpdateController checkpoint format")
        model = cls(**checkpoint["config"]).to(device=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.checkpoint_metadata = deepcopy(checkpoint.get("metadata", {}))
        return model
