"""Stage C: train shared G_phi on cached refinement states.

This is intentionally a separate entrypoint from stage-0 training.  Its input
is `(original LR x, current SR y_k, round k+1, HR)`, and it hard-freezes the
FLUX.2 backbone before constructing the optimizer.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataloaders.sr_refinement_state_dataset import (
    SRRefinementStateDataset,
    sr_refinement_state_collate_fn,
    build_refinement_dataloader,
    set_refinement_loader_epoch,
)
from models.rg_flux_artist_factory import build_rg_flux_artist
from models.text_embedding_cache import get_text_embedding_cache, resolve_prompt_embeddings
from rg_flux_fm import build_flow_matching_inputs, sample_sigma
from rl_sr.artifact_store import write_json_atomic
from rl_sr.conditioning import state_router_condition_tensors
from rl_sr.schema import content_tree_sha256
from rl_sr.snapshots import assert_rl_frozen_backbone
from rl_sr.stable_training import (
    SFT_LOSS_FIELDS,
    build_sft_parameter_groups,
    finite_sft_metrics,
    resolve_sft_controls,
    save_sft_checkpoint,
)
from rl_sr.training import refinement_image_losses
from train_rg_flux_sr import cfg, load_config


def _weight_dtype(accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _require_refinement_config(config):
    if str(cfg(config, "model.flux_backend", "")).lower() not in {"flux2_klein", "flux2-klein", "flux_2_klein"}:
        raise ValueError("RL-SR refinement currently requires model.flux_backend: flux2_klein")
    if not bool(cfg(config, "training.freeze_flux_transformer", False)):
        raise ValueError("RL-SR permanently freezes FLUX.2: training.freeze_flux_transformer must be true")
    if not bool(cfg(config, "condition.refinement.enabled", False)):
        raise ValueError("Set condition.refinement.enabled: true for stages C/E")
    if not cfg(config, "rl_sr.init_adapter", None):
        raise ValueError("Stage C requires rl_sr.init_adapter or --init_adapter; refusing random F0 initialization")


def train(config, state_jsonl=None, output_dir=None, max_steps=None):
    _require_refinement_config(config)
    rl_config = config.get("rl_sr") or {}
    # Reject malformed stability controls before loading data or a large model.
    resolve_sft_controls(config)
    state_jsonl = state_jsonl or rl_config.get("state_jsonl")
    if not state_jsonl:
        raise ValueError("Provide --state_jsonl or rl_sr.state_jsonl")
    output_dir = Path(output_dir or rl_config.get("sft_output_dir") or "exp_rg_flux_refiner_sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_device_batch = int(rl_config.get("batch_size", cfg(config, "data.batch_size", 1)))
    accumulation = int(rl_config.get("grad_accum_steps", cfg(config, "training.grad_accum_steps", 1)))
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(cfg(config, "model.dtype", "bf16")),
    )
    if accelerator.is_main_process:
        write_json_atomic(output_dir / "config.json", config)
    set_seed(int(rl_config.get("seed", cfg(config, "training.seed", 42))))
    dataset = SRRefinementStateDataset(
        state_jsonl,
        require_hr=True,
        vae_align=int(cfg(config, "data.vae_align", 16)),
    )
    dataloader = build_refinement_dataloader(dataset, per_device_batch,
        int(rl_config.get("num_workers", cfg(config, "data.num_workers", 2))),
        sr_refinement_state_collate_fn, config)
    if not len(dataloader):
        raise ValueError("State dataset is smaller than one batch; lower rl_sr.batch_size")
    artist = build_rg_flux_artist(config)
    init_adapter = rl_config.get("init_adapter")
    if init_adapter:
        artist.load_trainable(init_adapter, is_trainable=True)
    trainable, param_groups, sft_controls = build_sft_parameter_groups(artist, config)
    assert_rl_frozen_backbone(artist, train_router=False)
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(float(rl_config.get("adam_beta1", 0.9)), float(rl_config.get("adam_beta2", 0.95))),
        eps=float(rl_config.get("adam_epsilon", 1.0e-8)),
    )
    requested_steps = int(max_steps if max_steps is not None else rl_config.get(
        "sft_max_steps", cfg(config, "training.max_steps", 1000)))
    if requested_steps <= 0:
        raise ValueError("SFT max_steps must be a positive number of optimizer updates")
    scheduler = get_scheduler(
        str(rl_config.get("lr_scheduler", "constant_with_warmup")),
        optimizer,
        num_warmup_steps=int(rl_config.get("lr_warmup_steps", 0)),
        num_training_steps=requested_steps,
    )
    artist, optimizer, dataloader, scheduler = accelerator.prepare(artist, optimizer, dataloader, scheduler)
    weight_dtype = _weight_dtype(accelerator)
    text_cache = get_text_embedding_cache(config, dtype=weight_dtype)
    lr_cond_mode = str(cfg(config, "condition.lr_cond_mode", "latent_adapter"))
    sigma_sampling = str(cfg(config, "flow_matching.sigma_sampling", "uniform"))
    sigma_mean = float(cfg(config, "flow_matching.sigma_logit_mean", 0.0))
    sigma_std = float(cfg(config, "flow_matching.sigma_logit_std", 1.0))
    fm_weight = float(rl_config.get("sft_fm_weight", 1.0))
    latent_weight = float(rl_config.get("sft_latent_weight", 0.10))
    global_step = 0
    micro_steps = 0
    optimizer_steps_skipped = 0
    accumulated_micro_steps = 0
    accumulated_losses = {name: 0.0 for name in SFT_LOSS_FIELDS}
    checkpoint_records = []
    last_metrics = None
    csv_fields = ["optimizer_update", "micro_steps", "accumulated_micro_steps", *SFT_LOSS_FIELDS,
                  "grad_norm", "lora_lr", "refinement_lr"]
    loss_csv = output_dir / "loss_history.csv"
    if accelerator.is_main_process:
        with loss_csv.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=csv_fields).writeheader()
    epoch = 0
    artist.train()
    while global_step < requested_steps:
        set_refinement_loader_epoch(dataloader, epoch)
        epoch += 1
        for batch in dataloader:
            if global_step >= requested_steps:
                break
            unwrapped = accelerator.unwrap_model(artist)
            with torch.no_grad():
                hq = batch["hr"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                current = batch["current_sr"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                anchor = batch["original_lr_up"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                z_hr = unwrapped.encode_images(hq).to(accelerator.device, dtype=weight_dtype)
                z_current = unwrapped.encode_images(
                    current, sample=lr_cond_mode != "flux2_image_concat"
                ).to(accelerator.device, dtype=weight_dtype)
                z_anchor = unwrapped.encode_images(
                    anchor, sample=lr_cond_mode != "flux2_image_concat"
                ).to(accelerator.device, dtype=weight_dtype)
                prompt_embeds, pooled_prompt_embeds, text_ids = resolve_prompt_embeddings(
                    artist=unwrapped,
                    prompts=batch["prompt"],
                    image_keys=batch["sample_key"],
                    config=config,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    cache=text_cache,
                )
                sigma = sample_sigma(
                    z_hr.shape[0], z_hr.device, sampling=sigma_sampling,
                    logit_mean=sigma_mean, logit_std=sigma_std,
                ).to(weight_dtype)
                z_t, v_target = build_flow_matching_inputs(z_hr, sigma=sigma)
                router_condition, router_condition_mask, router_condition_confidence = state_router_condition_tensors(
                    batch["router_condition"], config, accelerator.device, weight_dtype
                )
            with accelerator.accumulate(artist):
                with accelerator.autocast():
                    v_pred = artist(
                        z_t=z_t,
                        timestep=sigma,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        z_lr=z_current,
                        z_anchor_lr=z_anchor,
                        refinement_round=batch["round_index"].to(accelerator.device),
                        lr_cond_mode=lr_cond_mode,
                        router_condition=router_condition,
                        router_condition_mask=router_condition_mask,
                        router_condition_confidence=router_condition_confidence,
                    )
                    loss_fm = F.mse_loss(v_pred.float(), v_target.float())
                    sigma_view = sigma.reshape(-1, 1, 1, 1).to(dtype=z_t.dtype)
                    z0_pred = z_t.float() - sigma_view.float() * v_pred.float()
                    loss_latent = F.smooth_l1_loss(z0_pred.float(), z_hr.float())
                    image_losses = refinement_image_losses(unwrapped, config, global_step, z0_pred, hq)
                    loss = fm_weight * loss_fm + latent_weight * loss_latent + image_losses["loss_image"]
                metrics = finite_sft_metrics({"loss": loss, "loss_fm": loss_fm,
                    "loss_latent": loss_latent, "loss_charb": image_losses["loss_charb"],
                    "loss_lpips": image_losses["loss_lpips"]}, accelerator)
                micro_steps += 1
                accumulated_micro_steps += 1
                for name, value in metrics.items():
                    accumulated_losses[name] += value
                accelerator.backward(loss)
                synchronized = accelerator.sync_gradients
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(artist.parameters(), float(rl_config.get("max_grad_norm", 1.0)))
                    finite_sft_metrics({"grad_norm": grad_norm}, accelerator)
                # Record the rates used for this update, before scheduler.step().
                update_lrs = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not synchronized:
                continue
            if accelerator.optimizer_step_was_skipped:
                optimizer_steps_skipped += 1
                accumulated_micro_steps = 0
                accumulated_losses = {name: 0.0 for name in SFT_LOSS_FIELDS}
                continue
            global_step += 1
            # The loss row is the mean of all microbatches in this update,
            # averaged across ranks, not just the final microbatch's loss.
            means = torch.tensor([accumulated_losses[name] / accumulated_micro_steps for name in SFT_LOSS_FIELDS],
                                 device=accelerator.device, dtype=torch.float64)
            means = accelerator.reduce(means, reduction="mean").tolist()
            last_metrics = finite_sft_metrics({**dict(zip(SFT_LOSS_FIELDS, means)), "grad_norm": grad_norm}, accelerator)
            if accelerator.is_main_process:
                row = {"optimizer_update": global_step, "micro_steps": micro_steps,
                       "accumulated_micro_steps": accumulated_micro_steps, **last_metrics,
                       "lora_lr": update_lrs.get("lora", 0.0), "refinement_lr": update_lrs.get("refinement", 0.0)}
                with loss_csv.open("a", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=csv_fields).writerow(row)
            accumulated_micro_steps = 0
            accumulated_losses = {name: 0.0 for name in SFT_LOSS_FIELDS}
            if sft_controls.checkpoint_every and global_step % sft_controls.checkpoint_every == 0:
                checkpoint = save_sft_checkpoint(unwrapped, output_dir / f"checkpoint-{global_step:08d}",
                    {"optimizer_updates": global_step, "micro_steps": micro_steps,
                     "optimizer_steps_skipped": optimizer_steps_skipped}, last_metrics, accelerator)
                if checkpoint is not None:
                    checkpoint_records.append({"optimizer_updates": global_step,
                        "path": f"checkpoint-{global_step:08d}", "adapter_sha256": checkpoint["adapter_sha256"],
                        "weights_sha256": checkpoint["weights_sha256"]})
            if accelerator.is_main_process and global_step % max(1, int(rl_config.get("log_every", 20))) == 0:
                print(
                    f"[refiner-sft] update={global_step} micro_steps={micro_steps} loss={last_metrics['loss']:.6f} "
                    f"fm={last_metrics['loss_fm']:.6f} latent={last_metrics['loss_latent']:.6f} "
                    f"charb={last_metrics['loss_charb']:.6f} lpips={last_metrics['loss_lpips']:.6f}",
                    flush=True,
                )
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(artist)
    final_checkpoint = save_sft_checkpoint(unwrapped, output_dir,
        {"optimizer_updates": global_step, "micro_steps": micro_steps,
         "optimizer_steps_skipped": optimizer_steps_skipped}, last_metrics, accelerator)
    if accelerator.is_main_process:
        adapter_dir = output_dir / "rg_flux_adapters"
        summary = {
            "stage": "C_shared_refinement_sft",
            "global_step": global_step,
            "optimizer_updates": global_step,
            "requested_optimizer_updates": requested_steps,
            "micro_steps": micro_steps,
            "optimizer_steps_skipped": optimizer_steps_skipped,
            "grad_accum_steps": accumulation,
            "state_jsonl": str(state_jsonl),
            "adapter_sha256": content_tree_sha256(adapter_dir),
            "weights_sha256": final_checkpoint["weights_sha256"],
            "sft_controls": asdict(sft_controls),
            "optimizer_parameter_groups": [{"name": group["name"],
                "initial_lr": group["initial_lr"], "final_lr": float(group["lr"]),
                "parameter_count": sum(parameter.numel() for parameter in group["params"])}
                for group in optimizer.param_groups],
            "loss_history_csv": str(loss_csv),
            "last_update_metrics": last_metrics,
            "checkpoints": checkpoint_records,
            "backbone_frozen": True,
            "routing": unwrapped.set_moe_inference_schedule(),
            "image_loss": rl_config.get("image_loss", {}),
            "round_balanced_batches": bool(rl_config.get("round_balanced_batches", False)),
            "trainable_parameter_names": [name for name, _ in trainable],
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return summary
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state_jsonl", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--init_adapter", default=None, help="Runtime F0 adapter path; overrides rl_sr.init_adapter.")
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_config = load_config(args.config)
    if args.init_adapter:
        resolved_config.setdefault("rl_sr", {})["init_adapter"] = args.init_adapter
    result = train(resolved_config, args.state_jsonl, args.output_dir, args.max_steps)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
