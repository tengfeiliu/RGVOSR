import math

import torch


def sample_sigma(
    batch_size,
    device,
    sampling="uniform",
    eps=1e-5,
    logit_mean=0.0,
    logit_std=1.0,
):
    if sampling == "uniform":
        sigma = torch.rand(batch_size, device=device)
    elif sampling in {"logit_normal", "lognorm"}:
        if not math.isfinite(float(logit_mean)):
            raise ValueError("logit_mean must be finite.")
        if not math.isfinite(float(logit_std)) or float(logit_std) <= 0.0:
            raise ValueError("logit_std must be finite and greater than zero.")
        logits = torch.randn(batch_size, device=device) * float(logit_std) + float(logit_mean)
        sigma = torch.sigmoid(logits)
    else:
        raise ValueError(f"Unsupported sigma sampling mode: {sampling}")
    return sigma.clamp(eps, 1.0 - eps)


def build_flow_matching_inputs(z_hr, eps=None, sigma=None):
    if eps is None:
        eps = torch.randn_like(z_hr)
    if sigma is None:
        sigma = sample_sigma(z_hr.shape[0], z_hr.device)
    sigma_view = sigma.reshape(-1, *([1] * (z_hr.ndim - 1))).to(device=z_hr.device, dtype=z_hr.dtype)
    z_t = (1.0 - sigma_view) * z_hr + sigma_view * eps
    v_target = eps - z_hr
    return z_t, v_target


def convert_sigma_to_flux_timestep(sigma, mode="sigma"):
    if mode == "sigma":
        return sigma
    if mode in {"sigma_1000", "diffusers_1000"}:
        return sigma * 1000.0
    raise ValueError(f"Unsupported FLUX timestep conversion mode: {mode}")


def compute_flux2_empirical_mu(image_seq_len, num_steps):
    """Return the resolution/step-aware FLUX.2 Klein dynamic-shift parameter."""
    image_seq_len = int(image_seq_len)
    num_steps = int(num_steps)
    if image_seq_len <= 0:
        raise ValueError("image_seq_len must be greater than zero.")
    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero.")

    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    mu_200 = a2 * image_seq_len + b2
    mu_10 = a1 * image_seq_len + b1
    slope = (mu_200 - mu_10) / 190.0
    intercept = mu_200 - 200.0 * slope
    return float(slope * num_steps + intercept)


def build_sigma_schedule(
    shape,
    num_steps=25,
    schedule="linear",
    sigma_start=1.0,
    device=None,
    dtype=None,
):
    """Build Euler sigma nodes for FLUX.2 Klein BCHW image-token latents."""
    if len(shape) != 4:
        raise ValueError(f"Expected BCHW latent shape, got {tuple(shape)}.")
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be greater than zero.")
    sigma_start = float(sigma_start)
    if not math.isfinite(sigma_start) or not 0.0 < sigma_start <= 1.0:
        raise ValueError("sigma_start must be finite and in (0, 1].")

    schedule = str(schedule).strip().lower()
    sigma_seq = torch.linspace(1.0, 0.0, int(num_steps) + 1, device=device, dtype=dtype)
    if schedule in {"linear", "uniform"}:
        pass
    elif schedule in {"empirical_shift", "flux2_empirical", "flux2_empirical_shift"}:
        height, width = int(shape[-2]), int(shape[-1])
        # FLUX.2 Klein stores already-packed VAE features as C=128 BCHW and
        # flattens HxW directly into transformer image tokens.
        image_seq_len = height * width
        mu = compute_flux2_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
        exp_mu = math.exp(mu)
        sigma_seq = (exp_mu * sigma_seq) / (1.0 + (exp_mu - 1.0) * sigma_seq)
    else:
        raise ValueError(f"Unsupported inference sigma schedule: {schedule}")

    # Scaling the complete normalized schedule preserves NFE while starting the
    # ODE at the same sigma used by LR warm-start initialization.
    return sigma_seq * sigma_start


@torch.no_grad()
def sample_multistep_fm(
    artist,
    shape,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids=None,
    degradation_vector=None,
    z_lr=None,
    dino_tokens=None,
    lr_cond_mode="latent_adapter",
    router_condition=None,
    router_condition_mask=None,
    router_condition_confidence=None,
    num_steps=25,
    schedule="linear",
    init_mode="pure_noise",
    sigma_start=1.0,
    device=None,
    dtype=None,
):
    device = device or prompt_embeds.device
    dtype = dtype or prompt_embeds.dtype
    init_mode = str(init_mode).strip().lower()
    sigma_start = float(sigma_start)
    noise = torch.randn(shape, device=device, dtype=dtype)
    if init_mode in {"pure_noise", "noise"}:
        if not math.isclose(sigma_start, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError("pure_noise initialization requires sigma_start=1.0.")
        z = noise
    elif init_mode in {"lr_warm_start", "lr_warmstart", "lr_noised"}:
        if z_lr is None:
            raise ValueError("lr_warm_start initialization requires z_lr.")
        if tuple(z_lr.shape) != tuple(shape):
            raise ValueError(
                f"z_lr shape {tuple(z_lr.shape)} does not match requested latent shape {tuple(shape)}."
            )
        z_lr = z_lr.to(device=device, dtype=dtype)
        z = (1.0 - sigma_start) * z_lr + sigma_start * noise
    else:
        raise ValueError(f"Unsupported inference initialization mode: {init_mode}")

    sigma_seq = build_sigma_schedule(
        shape=shape,
        num_steps=num_steps,
        schedule=schedule,
        sigma_start=sigma_start,
        device=device,
        dtype=dtype,
    )

    for i in range(num_steps):
        sigma_cur = sigma_seq[i]
        sigma_next = sigma_seq[i + 1]
        sigma_batch = sigma_cur.expand(shape[0])
        v_pred = artist(
            z_t=z,
            timestep=sigma_batch,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            degradation_vector=degradation_vector,
            z_lr=z_lr,
            dino_tokens=dino_tokens,
            lr_cond_mode=lr_cond_mode,
            router_condition=router_condition,
            router_condition_mask=router_condition_mask,
            router_condition_confidence=router_condition_confidence,
        )
        z = z - (sigma_cur - sigma_next) * v_pred

    return z
