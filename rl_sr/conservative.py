"""FP32 RGB fusion and paired conservative controller objectives."""

from __future__ import annotations

import math

import torch


DEFAULT_LOSS_WEIGHTS = {"pixel": 1.0, "lpips": 0.05, "regress": 1.0, "identity": 1.0, "smooth": 0.001}


def _validate_rgb(images):
    reference = next(iter(images.values()))
    for name, image in images.items():
        if not torch.is_tensor(image) or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"{name} must be BCHW RGB")
        if not image.is_floating_point():
            raise TypeError(f"{name} must be floating point RGB")
        if min(image.shape) <= 0:
            raise ValueError(f"{name} must have nonempty geometry")
        if image.shape != reference.shape or image.device != reference.device:
            raise ValueError("RGB tensors must have identical geometry and device")
        if not torch.isfinite(image).all() or torch.any(image < -1) or torch.any(image > 1):
            raise ValueError(f"{name} must be finite RGB in [-1, 1]")


def _validate_gate(gate, image, name, allow_scalar=False):
    if not torch.is_tensor(gate) or gate.ndim != 4 or gate.shape[:2] != (image.shape[0], 1):
        raise ValueError(f"{name} must have shape [B, 1, H, W] shared across RGB")
    spatial_sizes = (image.shape[-2:], (1, 1)) if allow_scalar else (image.shape[-2:],)
    if gate.shape[-2:] not in spatial_sizes or gate.device != image.device:
        raise ValueError(f"{name} must match RGB spatial geometry and device")
    if not gate.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(gate).all() or torch.any(gate < 0) or torch.any(gate > 1):
        raise ValueError(f"{name} must be finite and bounded in [0, 1]")


def fuse_rgb(current, proposal, weight, noop_threshold=None):
    """Apply a shared convex gate in FP32, with an exact parent RGB bypass.

    A threshold stops an entire image only when its maximum effective weight is
    below the threshold. Scalar [B,1,1,1] gates also support fixed-alpha ablations.
    FP32 callers receive their original tensor when every image is a no-op.
    """
    _validate_rgb({"current": current, "proposal": proposal})
    _validate_gate(weight, current, "weight", allow_scalar=True)
    if noop_threshold is not None:
        noop_threshold = float(noop_threshold)
        if not math.isfinite(noop_threshold) or not 0 <= noop_threshold <= 1:
            raise ValueError("noop_threshold must be finite and in [0, 1]")
    with torch.autocast(device_type=current.device.type, enabled=False):
        current_fp32, proposal_fp32, weight_fp32 = current.float(), proposal.float(), weight.float()
        cutoff = 0.0 if noop_threshold is None else noop_threshold
        noop = weight_fp32.flatten(1).amax(dim=1) <= cutoff
        if noop.all():
            return current_fp32
        fused = (1 - weight_fp32) * current_fp32 + weight_fp32 * proposal_fp32
        fused = torch.where(weight_fp32 == 0, current_fp32, fused)
        return torch.where(noop[:, None, None, None], current_fp32, fused)


def _per_image_error(image, hr, coefficients, lpips_model):
    difference = image - hr
    error = coefficients["pixel"] * torch.sqrt(difference.square() + 1e-6).mean(dim=(1, 2, 3))
    if lpips_model is not None and coefficients["lpips"]:
        perceptual = lpips_model(image, hr)
        if not torch.is_tensor(perceptual) or perceptual.ndim < 1 or perceptual.shape[0] != image.shape[0]:
            raise ValueError("lpips_model must return a per-image distance")
        perceptual = perceptual.float().reshape(image.shape[0], -1).mean(dim=1)
        if not torch.isfinite(perceptual).all():
            raise ValueError("lpips_model returned a nonfinite distance")
        error = error + coefficients["lpips"] * perceptual
    return error


def conservative_update_losses(output, current, hr, mask, identity_mask=None, lpips_model=None, weights=None):
    """Combine paired error, per-image regression, identity L1, and mask TV.

    The regression comparison always detaches the parent error. Identity labels
    designate exact current=HR samples and are checked to prevent accidental
    copying constraints on imperfect states. LPIPS inputs remain differentiable;
    the caller is responsible for freezing any perceptual model parameters.
    Returned component losses are unweighted, apart from the pixel/LPIPS mixture
    inside the paired errors. ``loss`` applies regress/identity/smooth weights.
    """
    _validate_rgb({"output": output, "current": current, "hr": hr})
    _validate_gate(mask, current, "mask")
    coefficients = dict(DEFAULT_LOSS_WEIGHTS)
    if weights is not None:
        if set(weights) - set(coefficients):
            raise ValueError("unknown conservative loss weight")
        coefficients.update({name: float(value) for name, value in weights.items()})
    if any(not math.isfinite(value) or value < 0 for value in coefficients.values()):
        raise ValueError("loss weights must be finite and nonnegative")
    if identity_mask is None:
        identity = torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
    else:
        identity = torch.as_tensor(identity_mask, device=current.device).reshape(-1)
        if identity.numel() != current.shape[0]:
            raise ValueError("identity_mask batch size must match RGB images")
        if not torch.isfinite(identity).all() or torch.any((identity != 0) & (identity != 1)):
            raise ValueError("identity_mask must contain boolean selection values")
        identity = identity.bool()
        if identity.any() and not torch.equal(current[identity], hr[identity]):
            raise ValueError("identity samples must have current exactly equal to HR")
    with torch.autocast(device_type=current.device.type, enabled=False):
        output_fp32, current_fp32, hr_fp32, mask_fp32 = output.float(), current.float(), hr.float(), mask.float()
        error_out = _per_image_error(output_fp32, hr_fp32, coefficients, lpips_model)
        with torch.no_grad():
            error_current = _per_image_error(current_fp32, hr_fp32, coefficients, lpips_model)
        loss_hr = error_out.mean()
        loss_regress = (error_out - error_current).relu().mean()
        loss_identity = (output_fp32[identity] - current_fp32[identity].detach()).abs().mean() if identity.any() else output_fp32.sum() * 0
        loss_smooth = mask_fp32.sum() * 0
        if mask.shape[-2] > 1:
            loss_smooth = loss_smooth + (mask_fp32[:, :, 1:, :] - mask_fp32[:, :, :-1, :]).abs().mean()
        if mask.shape[-1] > 1:
            loss_smooth = loss_smooth + (mask_fp32[:, :, :, 1:] - mask_fp32[:, :, :, :-1]).abs().mean()
        loss = loss_hr + coefficients["regress"] * loss_regress + coefficients["identity"] * loss_identity + coefficients["smooth"] * loss_smooth
        return {"loss": loss, "loss_hr": loss_hr, "loss_regress": loss_regress,
                "loss_identity": loss_identity, "loss_smooth": loss_smooth,
                "error_out": error_out, "error_current": error_current}
