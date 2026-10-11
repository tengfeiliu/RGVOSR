"""Shared, optional paired image supervision for C and E replay."""
import torch
import torch.nn.functional as F


def backward_policy_then_replay(policy_loss, replay_forward, backward):
    """Sum gradients without retaining two full transformer activation graphs.

    ``backward`` is Accelerator.backward so both terms receive the same
    accumulation/AMP scaling. The caller takes one optimizer step afterwards.
    ``replay_forward`` returns its weighted loss and any logging components.
    """
    backward(policy_loss)
    replay = replay_forward()
    backward(replay['loss'])
    return policy_loss.detach() + replay['loss'].detach(), replay


def refinement_image_losses(artist, config, global_step, z0_pred, hq):
    from train_rg_flux_sr import (crop_image_loss_inputs, charbonnier_loss,
                                 _get_lpips_loss_model, warmup_weight)
    options = (config.get("rl_sr") or {}).get("image_loss") or {}
    charb_weight = float(options.get("charb_weight", 0.0))
    lpips_weight = warmup_weight(global_step, options.get("lpips_warmup_start", 0),
                                options.get("lpips_warmup_end", 0), options.get("lpips_weight", 0.0))
    zero = z0_pred.float().new_zeros(())
    loss_charb, loss_lpips = zero, zero
    every = int(options.get("every", 1))
    if every <= 0 or charb_weight < 0 or float(options.get("lpips_weight", 0)) < 0:
        raise ValueError("Image loss weights must be nonnegative and every must be positive")
    due = global_step % every == 0 and (charb_weight > 0 or lpips_weight > 0)
    if due:
        with torch.autocast(device_type=z0_pred.device.type, enabled=False):
            # No downsample-to-degraded-LR loss: paired HR is the reference.
            crop_config = {"loss": {"image_loss_crop_size": int(options.get("crop_size", 256))}}
            latent, target, _, _ = crop_image_loss_inputs(z0_pred.float(), hq.float(), hq.float(), crop_config)
            prediction = artist.decode_latents_for_loss(latent).float()
            if prediction.shape != target.shape:
                raise ValueError(f"Decoded image/HR crop mismatch: {prediction.shape} vs {target.shape}")
            if charb_weight:
                loss_charb = charbonnier_loss(prediction, target)
            if lpips_weight:
                size = int(options.get("lpips_resize", 256))
                if size > 0:
                    prediction = F.interpolate(prediction, (size, size), mode="bilinear", align_corners=False)
                    target = F.interpolate(target, (size, size), mode="bilinear", align_corners=False)
                loss_lpips = _get_lpips_loss_model(prediction.device)(prediction.clamp(-1, 1), target.clamp(-1, 1)).mean()
    return {"loss_image": charb_weight * loss_charb + lpips_weight * loss_lpips,
            "loss_charb": loss_charb, "loss_lpips": loss_lpips, "lpips_weight": lpips_weight}
