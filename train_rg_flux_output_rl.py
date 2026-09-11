"""Stage E: direct output-level DiffusionNFT for shared multi-round SR.

This trainer never updates FLUX.2, VAE, text encoders, F0, or the MoE router.
It learns only G_phi's LoRA/condition-adapter tensors from whole final outputs
sampled by G_old.  It uses the saved final latent action, re-noises it, and
trains the native rectified-flow velocity field with DiffusionNFT.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset

from dataloaders.sr_refinement_state_dataset import (
    SRRefinementStateDataset,
    sr_refinement_state_collate_fn,
)
from losses.output_nft import nft_velocity_loss, reference_velocity_loss, renoise_rectified_flow
from models.rg_flux_artist_factory import build_rg_flux_artist
from models.text_embedding_cache import get_text_embedding_cache, resolve_prompt_embeddings
from rl_sr.artifact_store import read_jsonl, write_json_atomic
from rl_sr.conditioning import state_router_condition_tensors
from rl_sr.schema import RolloutRecord, StateRecord, adapter_content_sha256, content_sha256, content_tree_sha256
from rl_sr.snapshots import AdapterSnapshot, assert_rl_frozen_backbone, configure_rl_trainables, is_rl_trainable_name
from train_rg_flux_sr import cfg, load_config


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _weight_dtype(accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _require_rl_config(config):
    if str(cfg(config, "model.flux_backend", "")).lower() not in {"flux2_klein", "flux2-klein", "flux_2_klein"}:
        raise ValueError("Output NFT currently requires model.flux_backend: flux2_klein")
    if not bool(cfg(config, "training.freeze_flux_transformer", False)):
        raise ValueError("Output NFT requires training.freeze_flux_transformer: true")
    if not bool(cfg(config, "condition.refinement.enabled", False)):
        raise ValueError("Output NFT requires condition.refinement.enabled: true")


class ScoredRolloutDataset(Dataset):
    """Joins scored action latents with their state without using paths as IDs."""

    def __init__(self, state_jsonl, scored_rollout_jsonl, min_confidence=0.15):
        self.states = {record.state_id: record for record in read_jsonl(state_jsonl, StateRecord)}
        expected_policy_hash = None
        records = []
        for row in read_jsonl(scored_rollout_jsonl):
            record = RolloutRecord.from_dict(row)
            if record.state_id not in self.states:
                raise KeyError(f"Scored rollout references missing state {record.state_id}")
            if record.total_reward is None or record.optimality_probability is None or record.confidence is None:
                raise ValueError(f"Rollout {record.rollout_id} is unscored; run tools/score_sr_rollouts.py first")
            if not record.runtime.sample_latent or not Path(record.runtime.sample_latent).is_file():
                raise FileNotFoundError(f"Missing action latent for rollout {record.rollout_id}")
            if content_sha256(record.runtime.sample_latent) != record.sample_latent_sha256:
                raise RuntimeError(
                    f"Action latent hash mismatch for {record.rollout_id}; the buffer was modified after scoring."
                )
            if expected_policy_hash is None:
                expected_policy_hash = record.policy_adapter_sha256
            elif expected_policy_hash != record.policy_adapter_sha256:
                raise ValueError("A single NFT update buffer must contain outputs from exactly one G_old policy hash")
            if not record.hard_violation and float(record.confidence) < float(min_confidence):
                continue
            records.append(record)
        if not records:
            raise RuntimeError("No scored rollout survives the confidence threshold")
        self.records = records
        self.policy_adapter_sha256 = expected_policy_hash

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rollout = self.records[index]
        state = self.states[rollout.state_id]
        z0 = _torch_load(rollout.runtime.sample_latent)
        if not torch.is_tensor(z0):
            raise TypeError(f"Saved action latent is not a tensor: {rollout.runtime.sample_latent}")
        if z0.ndim == 4 and z0.shape[0] == 1:
            z0 = z0[0]
        if z0.ndim != 3:
            raise ValueError(f"Expected CHW final latent for {rollout.rollout_id}, got {tuple(z0.shape)}")
        return {
            "z0": z0.float(),
            "state": state,
            "optimality_probability": float(rollout.optimality_probability),
            "confidence": float(rollout.confidence),
            "hard_violation": bool(rollout.hard_violation),
            "rollout_id": rollout.rollout_id,
        }


def rollout_collate_fn(batch):
    return {
        "z0": torch.stack([item["z0"] for item in batch]),
        "states": [item["state"] for item in batch],
        "optimality_probability": torch.tensor([item["optimality_probability"] for item in batch]),
        "confidence": torch.tensor([item["confidence"] for item in batch]),
        "hard_violation": torch.tensor([item["hard_violation"] for item in batch], dtype=torch.bool),
        "rollout_id": [item["rollout_id"] for item in batch],
    }


def _read_state_images(states, device, dtype):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    originals, currents = [], []
    for state in states:
        if not state.runtime.original_lr or not state.runtime.current_sr:
            raise ValueError(f"State {state.state_id} lacks original_lr/current_sr runtime paths")
        with Image.open(state.runtime.current_sr) as image:
            image.load()
            current = image.convert("RGB")
        with Image.open(state.runtime.original_lr) as image:
            image.load()
            original = image.convert("RGB").resize(current.size, Image.Resampling.BICUBIC)
        currents.append(to_tensor(current).mul(2.0).sub(1.0))
        originals.append(to_tensor(original).mul(2.0).sub(1.0))
    return torch.stack(originals).to(device=device, dtype=dtype), torch.stack(currents).to(device=device, dtype=dtype)


def _device_snapshot(snapshot, artist):
    parameters = dict(artist.named_parameters())
    return {
        name: value.to(device=parameters[name].device, dtype=parameters[name].dtype)
        for name, value in snapshot.state.items()
    }


def _capture_live_adapter_state(artist):
    return {
        name: parameter.detach().clone()
        for name, parameter in artist.named_parameters()
        if is_rl_trainable_name(name) and not name.startswith("moe_router.")
    }


def _apply_adapter_state(artist, state):
    parameters = dict(artist.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))


def _load_snapshot(artist, checkpoint, label):
    artist.load_trainable(checkpoint, is_trainable=False)
    configure_rl_trainables(artist, train_router=False)
    assert_rl_frozen_backbone(artist, train_router=False)
    return AdapterSnapshot.capture(artist, label=label, train_router=False)


def _velocity(
    artist, z_t, tau, prompt_embeds, pooled, text_ids, z_current, z_anchor, rounds, lr_cond_mode,
    router_condition=None, router_condition_mask=None, router_condition_confidence=None,
):
    return artist(
        z_t=z_t,
        timestep=tau,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
        text_ids=text_ids,
        z_lr=z_current,
        z_anchor_lr=z_anchor,
        refinement_round=rounds,
        lr_cond_mode=lr_cond_mode,
        router_condition=router_condition,
        router_condition_mask=router_condition_mask,
        router_condition_confidence=router_condition_confidence,
    )


def train(config, args):
    _require_rl_config(config)
    rl_config = config.get("rl_sr") or {}
    output_dir = Path(args.output_dir or rl_config.get("rl_output_dir") or "exp_rg_flux_output_rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    update_dataset = ScoredRolloutDataset(
        args.state_jsonl,
        args.scored_rollout_jsonl,
        min_confidence=float(rl_config.get("min_reward_confidence", 0.15)),
    )
    old_hash = adapter_content_sha256(args.old_adapter)
    if update_dataset.policy_adapter_sha256 != old_hash:
        raise ValueError(
            "Scored rollout buffer was not sampled by --old_adapter. Do not run NFT against stale or mixed-policy actions."
        )
    replay_dataset = SRRefinementStateDataset(args.sft_state_jsonl or args.state_jsonl, require_hr=True,
                                                vae_align=int(cfg(config, "data.vae_align", 16)))
    batch_size = int(rl_config.get("rl_batch_size", 1))
    replay_batch_size = int(rl_config.get("sft_replay_batch_size", batch_size))
    update_loader = torch.utils.data.DataLoader(
        update_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=int(rl_config.get("num_workers", 2)), pin_memory=True, collate_fn=rollout_collate_fn,
    )
    replay_loader = torch.utils.data.DataLoader(
        replay_dataset, batch_size=replay_batch_size, shuffle=True, drop_last=True,
        num_workers=int(rl_config.get("num_workers", 2)), pin_memory=True, collate_fn=sr_refinement_state_collate_fn,
    )
    if not len(update_loader):
        raise ValueError("RL rollout buffer is smaller than rl_sr.rl_batch_size")
    if not len(replay_loader):
        raise ValueError("SFT replay set is smaller than rl_sr.sft_replay_batch_size")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(rl_config.get("rl_grad_accum_steps", 1)),
        mixed_precision=str(cfg(config, "model.dtype", "bf16")),
    )
    set_seed(int(rl_config.get("seed", cfg(config, "training.seed", 42))))
    artist = build_rg_flux_artist(config)
    # Capture immutable G_ref and G_old before loading the trainable G_phi.
    reference_snapshot = _load_snapshot(artist, args.reference_adapter, "G_ref")
    old_snapshot = _load_snapshot(artist, args.old_adapter, "G_old")
    artist.load_trainable(args.policy_adapter, is_trainable=True)
    trainable = configure_rl_trainables(artist, train_router=False)
    assert_rl_frozen_backbone(artist, train_router=False)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(rl_config.get("rl_lr", 2.0e-5)),
        betas=(float(rl_config.get("adam_beta1", 0.9)), float(rl_config.get("adam_beta2", 0.95))),
        weight_decay=float(rl_config.get("weight_decay", 0.01)),
        eps=float(rl_config.get("adam_epsilon", 1.0e-8)),
    )
    max_steps = int(args.max_steps or rl_config.get("rl_max_steps", len(update_loader)))
    scheduler = get_scheduler(
        str(rl_config.get("rl_lr_scheduler", "constant")), optimizer,
        num_warmup_steps=int(rl_config.get("rl_lr_warmup_steps", 0)), num_training_steps=max_steps,
    )
    artist, optimizer, update_loader, replay_loader, scheduler = accelerator.prepare(
        artist, optimizer, update_loader, replay_loader, scheduler
    )
    unwrapped = accelerator.unwrap_model(artist)
    old_device_state = _device_snapshot(old_snapshot, unwrapped)
    reference_device_state = _device_snapshot(reference_snapshot, unwrapped)
    dtype = _weight_dtype(accelerator)
    text_cache = get_text_embedding_cache(config, dtype=dtype)
    lr_cond_mode = str(cfg(config, "condition.lr_cond_mode", "latent_adapter"))
    beta = float(rl_config.get("nft_beta", 1.0))
    nft_weight = float(rl_config.get("nft_weight", 1.0))
    reference_weight = float(rl_config.get("reference_weight", 0.10))
    sft_weight = float(rl_config.get("sft_replay_weight", 0.25))
    sft_latent_weight = float(rl_config.get("sft_replay_latent_weight", 0.05))
    replay_iterator = iter(replay_loader)
    global_step = 0
    artist.train()
    while global_step < max_steps:
        for rollout_batch in update_loader:
            if global_step >= max_steps:
                break
            try:
                replay_batch = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(replay_loader)
                replay_batch = next(replay_iterator)
            with torch.no_grad():
                z0 = rollout_batch["z0"].to(accelerator.device, dtype=dtype)
                anchor, current = _read_state_images(rollout_batch["states"], accelerator.device, dtype)
                z_current = unwrapped.encode_images(current, sample=lr_cond_mode != "flux2_image_concat").to(dtype=dtype)
                z_anchor = unwrapped.encode_images(anchor, sample=lr_cond_mode != "flux2_image_concat").to(dtype=dtype)
                prompts = [state.prompt for state in rollout_batch["states"]]
                image_keys = [state.sample_key for state in rollout_batch["states"]]
                prompt_embeds, pooled, text_ids = resolve_prompt_embeddings(
                    unwrapped, prompts, image_keys, config, accelerator.device, dtype, cache=text_cache
                )
                rounds = torch.tensor([state.round_index for state in rollout_batch["states"]], device=accelerator.device)
                router_condition, router_condition_mask, router_condition_confidence = state_router_condition_tensors(
                    rollout_batch["states"], config, accelerator.device, dtype
                )
                tau = torch.rand(z0.shape[0], device=accelerator.device, dtype=dtype)
                z_t, velocity_target = renoise_rectified_flow(z0, tau=tau)
                # Build replay targets under no-grad; only G_phi follows gradients.
                replay_hq = replay_batch["hr"].to(accelerator.device, dtype=dtype)
                replay_current = replay_batch["current_sr"].to(accelerator.device, dtype=dtype)
                replay_anchor = replay_batch["original_lr_up"].to(accelerator.device, dtype=dtype)
                z_hr = unwrapped.encode_images(replay_hq).to(dtype=dtype)
                rz_current = unwrapped.encode_images(replay_current, sample=lr_cond_mode != "flux2_image_concat").to(dtype=dtype)
                rz_anchor = unwrapped.encode_images(replay_anchor, sample=lr_cond_mode != "flux2_image_concat").to(dtype=dtype)
                r_prompt, r_pooled, r_text_ids = resolve_prompt_embeddings(
                    unwrapped, replay_batch["prompt"], replay_batch["sample_key"], config,
                    accelerator.device, dtype, cache=text_cache
                )
                r_tau = torch.rand(z_hr.shape[0], device=accelerator.device, dtype=dtype)
                r_eps = torch.randn_like(z_hr)
                r_tau_view = r_tau.reshape(-1, 1, 1, 1)
                r_z_t = (1.0 - r_tau_view) * z_hr + r_tau_view * r_eps
                r_target = r_eps - z_hr
                r_router_condition, r_router_condition_mask, r_router_condition_confidence = state_router_condition_tensors(
                    replay_batch["router_condition"], config, accelerator.device, dtype
                )
            # One model holds all policies.  Device copies contain adapters only,
            # so G_old/G_ref need no duplicated FLUX.2 backbone.
            live_policy = _capture_live_adapter_state(unwrapped)
            _apply_adapter_state(unwrapped, old_device_state)
            with torch.no_grad():
                v_old = _velocity(unwrapped, z_t, tau, prompt_embeds, pooled, text_ids,
                                  z_current, z_anchor, rounds, lr_cond_mode,
                                  router_condition, router_condition_mask, router_condition_confidence)
            _apply_adapter_state(unwrapped, reference_device_state)
            with torch.no_grad():
                v_reference = _velocity(unwrapped, z_t, tau, prompt_embeds, pooled, text_ids,
                                        z_current, z_anchor, rounds, lr_cond_mode,
                                        router_condition, router_condition_mask, router_condition_confidence)
            _apply_adapter_state(unwrapped, live_policy)
            with accelerator.accumulate(artist):
                with accelerator.autocast():
                    v_phi = _velocity(artist, z_t, tau, prompt_embeds, pooled, text_ids,
                                      z_current, z_anchor, rounds, lr_cond_mode,
                                      router_condition, router_condition_mask, router_condition_confidence)
                    probability = rollout_batch["optimality_probability"].to(accelerator.device, dtype=dtype)
                    confidence = rollout_batch["confidence"].to(accelerator.device, dtype=dtype)
                    hard = rollout_batch["hard_violation"].to(accelerator.device)
                    sample_weight = torch.where(hard, torch.ones_like(confidence), confidence)
                    loss_nft = nft_velocity_loss(v_phi, v_old, velocity_target, probability,
                                                 beta=beta, sample_weight=sample_weight)
                    loss_reference = reference_velocity_loss(v_phi, v_reference)
                    r_v_phi = _velocity(
                        artist, r_z_t, r_tau, r_prompt, r_pooled, r_text_ids, rz_current, rz_anchor,
                        replay_batch["round_index"].to(accelerator.device), lr_cond_mode,
                        r_router_condition, r_router_condition_mask, r_router_condition_confidence,
                    )
                    loss_sft_fm = F.mse_loss(r_v_phi.float(), r_target.float())
                    r_z0_pred = r_z_t - r_tau_view * r_v_phi
                    loss_sft_latent = F.smooth_l1_loss(r_z0_pred.float(), z_hr.float())
                    loss = (
                        nft_weight * loss_nft
                        + reference_weight * loss_reference
                        + sft_weight * loss_sft_fm
                        + sft_latent_weight * loss_sft_latent
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(artist.parameters(), float(rl_config.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if accelerator.is_main_process and global_step % int(rl_config.get("log_every", 10)) == 0:
                print(
                    f"[output-nft] step={global_step} total={loss.detach().float().item():.6f} "
                    f"nft={loss_nft.detach().float().item():.6f} ref={loss_reference.detach().float().item():.6f} "
                    f"sft={loss_sft_fm.detach().float().item():.6f}", flush=True
                )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        adapter_dir = output_dir / "rg_flux_adapters"
        unwrapped.save_trainable(adapter_dir, save_files=True)
        summary = {
            "stage": "E_output_level_diffusion_nft",
            "global_step": global_step,
            "policy_adapter_sha256": content_tree_sha256(adapter_dir),
            "old_adapter_sha256": old_hash,
            "reference_adapter_sha256": adapter_content_sha256(args.reference_adapter),
            "rollout_buffer_policy_sha256": update_dataset.policy_adapter_sha256,
            "backbone_frozen": True,
            "router_frozen": True,
            "action": "saved_final_rectified_flow_latent_z0",
            "trainable_parameter_names": [name for name, _ in trainable],
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return summary
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state_jsonl", required=True)
    parser.add_argument("--scored_rollout_jsonl", required=True)
    parser.add_argument("--reference_adapter", required=True, help="Frozen G_ref = post-C SFT checkpoint.")
    parser.add_argument("--old_adapter", required=True, help="Frozen G_old that generated this exact buffer.")
    parser.add_argument("--policy_adapter", required=True, help="Trainable G_phi; usually equal to G_old at first update.")
    parser.add_argument("--sft_state_jsonl", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    result = train(load_config(cli_args.config), cli_args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
