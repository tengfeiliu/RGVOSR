"""Output-level DiffusionNFT utilities for rectified-flow SR."""

from __future__ import annotations

import torch


def renoise_rectified_flow(z0: torch.Tensor, tau: torch.Tensor | None = None, eps: torch.Tensor | None = None):
    if z0.ndim < 2:
        raise ValueError("z0 must include a batch dimension")
    if eps is None:
        eps = torch.randn_like(z0)
    if tau is None:
        tau = torch.rand(z0.shape[0], device=z0.device, dtype=z0.dtype)
    if tau.ndim != 1 or tau.shape[0] != z0.shape[0]:
        raise ValueError("tau must be shape [batch]")
    tau_view = tau.reshape(-1, *([1] * (z0.ndim - 1))).to(device=z0.device, dtype=z0.dtype)
    return (1.0 - tau_view) * z0 + tau_view * eps, eps - z0


def nft_velocity_loss(
    velocity_current: torch.Tensor,
    velocity_old: torch.Tensor,
    velocity_target: torch.Tensor,
    optimality_probability: torch.Tensor,
    beta: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if velocity_current.shape != velocity_old.shape or velocity_current.shape != velocity_target.shape:
        raise ValueError("NFT velocities must have identical shapes")
    if float(beta) <= 0:
        raise ValueError("beta must be positive")
    probability = optimality_probability.to(device=velocity_current.device, dtype=velocity_current.dtype).reshape(-1)
    if probability.shape[0] != velocity_current.shape[0]:
        raise ValueError("optimality_probability batch size must match velocities")
    probability = probability.clamp(0.0, 1.0)
    view = probability.reshape(-1, *([1] * (velocity_current.ndim - 1)))
    beta = float(beta)
    positive = (1.0 - beta) * velocity_old.detach() + beta * velocity_current
    negative = (1.0 + beta) * velocity_old.detach() - beta * velocity_current
    per_item = (
        view * (positive - velocity_target).square()
        + (1.0 - view) * (negative - velocity_target).square()
    ).flatten(1).mean(dim=1)
    if sample_weight is not None:
        weights = sample_weight.to(device=per_item.device, dtype=per_item.dtype).reshape(-1).clamp_min(0)
        if weights.shape != per_item.shape:
            raise ValueError("sample_weight batch size must match velocities")
        return (per_item * weights).sum() / weights.sum().clamp_min(torch.finfo(per_item.dtype).eps)
    return per_item.mean()


def reference_velocity_loss(velocity_current: torch.Tensor, velocity_reference: torch.Tensor) -> torch.Tensor:
    if velocity_current.shape != velocity_reference.shape:
        raise ValueError("Reference velocities must have identical shapes")
    return (velocity_current - velocity_reference.detach()).square().mean()


def group_optimality_probabilities(
    rewards: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    scale_floor: float = 0.05,
) -> torch.Tensor:
    """Map group-relative rewards to DiffusionNFT optimality probabilities."""

    rewards = rewards.float().reshape(-1)
    if rewards.numel() == 0:
        return rewards
    if float(scale_floor) <= 0:
        raise ValueError("scale_floor must be positive")
    if group_ids is None:
        group_ids = torch.zeros_like(rewards, dtype=torch.long)
    group_ids = group_ids.reshape(-1).to(device=rewards.device)
    if group_ids.shape != rewards.shape:
        raise ValueError("group_ids must match rewards")
    output = torch.empty_like(rewards)
    for group in torch.unique(group_ids):
        mask = group_ids == group
        values = rewards[mask]
        centered = values - values.mean()
        mad = (values - values.median()).abs().median() * 1.4826
        scale = mad.clamp_min(float(scale_floor))
        output[mask] = 0.5 + 0.5 * (centered / scale).clamp(-1.0, 1.0)
    return output
