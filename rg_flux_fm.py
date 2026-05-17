import torch


def sample_sigma(batch_size, device, sampling="uniform", eps=1e-5):
    if sampling == "uniform":
        sigma = torch.rand(batch_size, device=device)
    elif sampling in {"logit_normal", "lognorm"}:
        sigma = torch.sigmoid(torch.randn(batch_size, device=device))
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
    num_steps=25,
    device=None,
    dtype=None,
):
    device = device or prompt_embeds.device
    dtype = dtype or prompt_embeds.dtype
    z = torch.randn(shape, device=device, dtype=dtype)
    sigma_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

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
        )
        z = z - (sigma_cur - sigma_next) * v_pred

    return z
