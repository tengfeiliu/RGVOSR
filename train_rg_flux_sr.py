import argparse
import copy
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from dataloaders.rg_flux_jsonl_dataset import RGFluxSRJsonlDataset, rg_flux_collate_fn
from models.flux_sr_artist import FluxSRArtist
from rg_flux_fm import build_flow_matching_inputs, sample_sigma


logger = get_logger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cfg(config, path, default=None):
    current = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def normalize_report_to(report_to):
    if report_to is None:
        return None
    if isinstance(report_to, str):
        value = report_to.strip()
        if value.lower() in {"", "none", "null", "false", "off", "no"}:
            return None
        return value
    return report_to


def create_logger(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))],
    )
    return logging.getLogger(__name__)


def weight_dtype_from_accelerator(accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def find_latest_checkpoint(output_dir, resume_ckpt=None):
    if resume_ckpt:
        path = Path(resume_ckpt)
        if path.exists():
            return path
        print(f"Warning: resume checkpoint does not exist: {resume_ckpt}")
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint-*"))
    return candidates[-1] if candidates else None


def save_rg_checkpoint(accelerator, artist, optimizer, lr_scheduler, checkpoint_dir, global_step):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(artist)
    unwrapped.save_trainable(checkpoint_dir / "rg_flux_adapters")
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )


def load_rg_checkpoint(accelerator, artist, optimizer, lr_scheduler, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    adapter_dir = checkpoint_dir / "rg_flux_adapters"
    unwrapped = accelerator.unwrap_model(artist)
    unwrapped.load_trainable(adapter_dir if adapter_dir.exists() else checkpoint_dir, is_trainable=True)

    state_path = checkpoint_dir / "training_state.pt"
    global_step = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state.get("optimizer", {}))
        lr_scheduler.load_state_dict(state.get("lr_scheduler", {}))
        global_step = int(state.get("global_step", 0))
    return global_step


def make_experiment_name(config):
    suffix = cfg(config, "training.suffix", "")
    lr_mode = cfg(config, "condition.lr_cond_mode", "latent_adapter")
    stage = cfg(config, "training.stage", "A")
    crop = cfg(config, "data.crop_size", 512)
    return f"rg_flux_sr_ms_stage{stage}_{lr_mode}_size{crop}{suffix}"


def main(config_path, dry_run=False):
    config = load_config(config_path)
    config.setdefault("training", {})
    config.setdefault("model", {})
    config.setdefault("data", {})
    config.setdefault("condition", {})

    report_to = normalize_report_to(cfg(config, "training.report_to", None))
    exp_name = cfg(config, "training.exp_name", None) or make_experiment_name(config)
    output_root = Path(cfg(config, "training.output_dir", "exp_rg_flux_sr"))
    output_dir = output_root / exp_name
    logging_dir = output_dir / cfg(config, "training.logging_dir", "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg(config, "training.grad_accum_steps", 1)),
        mixed_precision=str(cfg(config, "model.dtype", "bf16")),
        log_with=report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (output_dir / "args.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        local_logger = create_logger(logging_dir)
        local_logger.info("Experiment directory created at %s", output_dir)

    seed = cfg(config, "training.seed", 42)
    if seed is not None:
        set_seed(int(seed))

    dataset = RGFluxSRJsonlDataset(
        jsonl_path=cfg(config, "data.jsonl_path"),
        crop_size=int(cfg(config, "data.crop_size", 512)),
        scale=int(cfg(config, "data.scale", 4)),
        mode="train",
        use_prompt=bool(cfg(config, "condition.use_prompt", True)),
        use_suggestions=bool(cfg(config, "condition.use_suggestions", True)),
        use_degradation_vector=bool(cfg(config, "condition.use_degradation_vector", True)),
        vae_align=int(cfg(config, "data.vae_align", 16)),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg(config, "data.batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg(config, "data.num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=bool(cfg(config, "data.num_workers", 4) > 0),
        collate_fn=rg_flux_collate_fn,
    )

    artist = FluxSRArtist(config)
    trainable_named_params = [(name, param) for name, param in artist.named_parameters() if param.requires_grad]
    trainable_params = [param for _, param in trainable_named_params]
    if not trainable_named_params:
        raise RuntimeError("No trainable parameters found for RG-FLUX-SR-MS Stage A/B.")
    lora_params = [
        param
        for name, param in trainable_named_params
        if "lora" in name.lower() or name.startswith("transformer.")
    ]
    lora_param_ids = {id(param) for param in lora_params}
    adapter_params = [param for _, param in trainable_named_params if id(param) not in lora_param_ids]
    param_groups = []
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": float(cfg(config, "training.lr_adapter", 1e-4))})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": float(cfg(config, "training.lr_lora", 5e-5))})

    optimizer_class = torch.optim.AdamW
    if bool(cfg(config, "training.use_8bit_adam", False)):
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit

    optimizer = optimizer_class(
        param_groups,
        betas=(float(cfg(config, "training.adam_beta1", 0.9)), float(cfg(config, "training.adam_beta2", 0.95))),
        weight_decay=float(cfg(config, "training.weight_decay", 0.01)),
        eps=float(cfg(config, "training.adam_epsilon", 1e-8)),
    )

    max_steps = 1 if dry_run else int(cfg(config, "training.max_steps", 100000))
    lr_scheduler = get_scheduler(
        cfg(config, "training.lr_scheduler", "constant_with_warmup"),
        optimizer=optimizer,
        num_warmup_steps=int(cfg(config, "training.lr_warmup_steps", 0)) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
        num_cycles=int(cfg(config, "training.lr_num_cycles", 1)),
    )

    artist, optimizer, dataloader, lr_scheduler = accelerator.prepare(artist, optimizer, dataloader, lr_scheduler)
    weight_dtype = weight_dtype_from_accelerator(accelerator)

    global_step = 0
    resume_path = find_latest_checkpoint(output_dir, cfg(config, "training.resume_ckpt", None))
    if resume_path:
        if accelerator.is_main_process:
            logger.info("Loading RG-FLUX-SR-MS state from %s", resume_path)
        global_step = load_rg_checkpoint(accelerator, artist, optimizer, lr_scheduler, resume_path)

    if accelerator.is_main_process and report_to is not None:
        accelerator.init_trackers(
            project_name=cfg(config, "training.tracker_project_name", "rg_flux_sr"),
            config=copy.deepcopy(config),
        )

    progress_bar = tqdm(
        range(global_step, max_steps),
        initial=global_step,
        total=max_steps,
        desc="RG-FLUX-SR-MS",
        disable=not accelerator.is_local_main_process,
    )
    fm_weight = float(cfg(config, "loss.fm_weight", 1.0))
    checkpoint_dir = output_dir / "checkpoints"
    save_every = int(cfg(config, "training.save_every", 5000))
    log_every = int(cfg(config, "training.log_every", 100))
    sigma_sampling = cfg(config, "flow_matching.sigma_sampling", "uniform")
    lr_cond_mode = cfg(config, "condition.lr_cond_mode", "latent_adapter")

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break
            hq = batch["hq"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            lq_up = batch["lq_up"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            degradation_vector = batch["degradation_vector"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
            prompts = batch["prompt"]

            unwrapped_artist = accelerator.unwrap_model(artist)
            with torch.no_grad():
                z_hr = unwrapped_artist.encode_images(hq)
                z_lr = unwrapped_artist.encode_images(lq_up)
                prompt_embeds, pooled_prompt_embeds, text_ids = unwrapped_artist.encode_prompts(
                    prompts,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                dino_tokens = unwrapped_artist.extract_visual_tokens(lq_up)
                sigma = sample_sigma(z_hr.shape[0], z_hr.device, sampling=sigma_sampling).to(dtype=weight_dtype)
                eps = torch.randn_like(z_hr)
                z_t, v_target = build_flow_matching_inputs(z_hr, eps=eps, sigma=sigma)

            with accelerator.accumulate(artist):
                with accelerator.autocast():
                    v_pred = artist(
                        z_t=z_t,
                        timestep=sigma,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        degradation_vector=degradation_vector,
                        z_lr=z_lr,
                        dino_tokens=dino_tokens,
                        lr_cond_mode=lr_cond_mode,
                    )
                    if v_pred.shape != v_target.shape:
                        raise RuntimeError(f"v_pred shape {tuple(v_pred.shape)} != target {tuple(v_target.shape)}")
                    loss_fm = torch.nn.functional.mse_loss(v_pred.float(), v_target.float())
                    loss = fm_weight * loss_fm

                accelerator.backward(loss)
                if accelerator.sync_gradients and float(cfg(config, "training.max_grad_norm", 1.0)) > 0:
                    accelerator.clip_grad_norm_(trainable_params, float(cfg(config, "training.max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if accelerator.is_main_process and global_step % log_every == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "loss_fm": loss_fm.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                if save_every > 0 and global_step % save_every == 0:
                    save_rg_checkpoint(
                        accelerator,
                        artist,
                        optimizer,
                        lr_scheduler,
                        checkpoint_dir / f"checkpoint-{global_step:08d}",
                        global_step,
                    )

    save_rg_checkpoint(
        accelerator,
        artist,
        optimizer,
        lr_scheduler,
        checkpoint_dir / f"checkpoint-{global_step:08d}",
        global_step,
    )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_rg_flux_sr_ms.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Run exactly one optimization step.")
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
