"""Collect whole-output candidates from frozen G_old for Stage E.

The saved latent is the exact final ODE output used by NFT.  ``executed_sr`` is
currently identical to ``raw_sr``; if a future product postprocesses images,
its output must remain a reward-only artifact and must never replace the saved
action latent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.sr_refinement_state_dataset import SRRefinementStateDataset
from models.rg_flux_artist_factory import build_rg_flux_artist
from models.text_embedding_cache import get_text_embedding_cache, resolve_prompt_embeddings
from rg_flux_fm import sample_multistep_fm
from rl_sr.artifact_store import write_jsonl_atomic
from rl_sr.conditioning import state_router_condition_tensors
from rl_sr.schema import RuntimePaths, RolloutRecord, adapter_content_sha256, content_sha256, identity_sha256
from rl_sr.snapshots import assert_rl_frozen_backbone, configure_rl_trainables
from train_rg_flux_sr import cfg, load_config


def _dtype(config):
    name = str(cfg(config, "model.dtype", "bf16")).lower()
    return {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16}.get(name, torch.float32)


def _seed(base_seed, state_id, candidate_index):
    digest = hashlib.sha256(f"{state_id}:{candidate_index}".encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _sampler_payload(args, config):
    return {
        "num_steps": int(args.num_inference_steps or cfg(config, "flow_matching.num_inference_steps", 25)),
        "schedule": str(args.inference_schedule or cfg(config, "flow_matching.inference_schedule", "linear")),
        "init_mode": str(args.inference_init_mode or cfg(config, "flow_matching.inference_init_mode", "pure_noise")),
        "sigma_start": float(
            args.inference_sigma_start
            if args.inference_sigma_start is not None
            else cfg(config, "flow_matching.inference_sigma_start", 1.0)
        ),
        "action_representation": "final_rectified_flow_latent_z0",
    }


def main(args):
    config = load_config(args.config)
    if not bool(cfg(config, "training.freeze_flux_transformer", False)):
        raise ValueError("Stage E requires training.freeze_flux_transformer: true")
    if not bool(cfg(config, "condition.refinement.enabled", False)):
        raise ValueError("Stage E requires condition.refinement.enabled: true")
    device = torch.device(args.device)
    dtype = _dtype(config)
    artist = build_rg_flux_artist(config).to(device)
    artist.load_trainable(args.old_adapter, is_trainable=False)
    configure_rl_trainables(artist, train_router=False)
    assert_rl_frozen_backbone(artist, train_router=False)
    artist.eval()
    if hasattr(artist, "align_inference_dtype"):
        artist.align_inference_dtype(dtype)
    if hasattr(artist, "set_moe_inference_schedule"):
        artist.set_moe_inference_schedule()
    dataset = SRRefinementStateDataset(args.state_jsonl, require_hr=False, vae_align=int(cfg(config, "data.vae_align", 16)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    to_pil = ToPILImage()
    text_cache = get_text_embedding_cache(config, dtype=dtype)
    lr_cond_mode = str(cfg(config, "condition.lr_cond_mode", "latent_adapter"))
    policy_hash = adapter_content_sha256(args.old_adapter)
    sampler = _sampler_payload(args, config)
    sampler_hash = identity_sha256(sampler)
    records = []
    with torch.no_grad():
        for item in dataset:
            current = item["current_sr"].unsqueeze(0).to(device=device, dtype=dtype)
            anchor = item["original_lr_up"].unsqueeze(0).to(device=device, dtype=dtype)
            z_current = artist.encode_images(current, sample=lr_cond_mode != "flux2_image_concat").to(device=device, dtype=dtype)
            z_anchor = artist.encode_images(anchor, sample=lr_cond_mode != "flux2_image_concat").to(device=device, dtype=dtype)
            prompt_embeds, pooled_prompt_embeds, text_ids = resolve_prompt_embeddings(
                artist=artist,
                prompts=[item["prompt"]],
                image_keys=[item["sample_key"]],
                config=config,
                device=device,
                dtype=dtype,
                cache=text_cache,
            )
            dino_tokens = artist.extract_visual_tokens(current)
            router_condition, router_condition_mask, router_condition_confidence = state_router_condition_tensors(
                [item["router_condition"]], config, device, dtype
            )
            for candidate_index in range(args.num_candidates):
                seed = _seed(args.seed, item["state_id"], candidate_index)
                with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    z0 = sample_multistep_fm(
                        artist=artist,
                        shape=tuple(z_current.shape),
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        z_lr=z_current,
                        z_anchor_lr=z_anchor,
                        dino_tokens=dino_tokens,
                        lr_cond_mode=lr_cond_mode,
                        refinement_round=int(item["round_index"]),
                        router_condition=router_condition,
                        router_condition_mask=router_condition_mask,
                        router_condition_confidence=router_condition_confidence,
                        num_steps=sampler["num_steps"],
                        schedule=sampler["schedule"],
                        init_mode=sampler["init_mode"],
                        sigma_start=sampler["sigma_start"],
                        device=device,
                        dtype=dtype,
                    )
                sample_dir = output_dir / "samples" / item["state_id"] / f"candidate-{candidate_index:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                latent_path = sample_dir / "z0.pt"
                raw_path = sample_dir / "raw_sr.png"
                torch.save(z0.detach().cpu(), latent_path)
                sr = artist.decode_latents(z0).clamp(-1, 1).add(1.0).mul(0.5).clamp(0, 1)
                to_pil(sr[0].float().cpu()).save(raw_path)
                # No hidden postprocessing in v1: the reward sees exactly this decoded action.
                executed_path = raw_path
                records.append(
                    RolloutRecord(
                        state_id=item["state_id"],
                        policy_adapter_sha256=policy_hash,
                        candidate_index=candidate_index,
                        sample_latent_sha256=content_sha256(latent_path),
                        raw_sr_sha256=content_sha256(raw_path),
                        executed_sr_sha256=content_sha256(executed_path),
                        sampler_sha256=sampler_hash,
                        runtime=RuntimePaths(
                            sample_latent=str(latent_path.resolve()),
                            raw_sr=str(raw_path.resolve()),
                            executed_sr=str(executed_path.resolve()),
                        ),
                    )
                )
    output_jsonl = output_dir / "rollouts.jsonl"
    write_jsonl_atomic(output_jsonl, records)
    summary = {
        "rollout_count": len(records),
        "state_count": len(dataset),
        "num_candidates": args.num_candidates,
        "policy_adapter_sha256": policy_hash,
        "sampler_sha256": sampler_hash,
        "sampler": sampler,
        "action": "saved_final_rectified_flow_latent_z0",
    }
    (output_dir / "rollout_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state_jsonl", required=True)
    parser.add_argument("--old_adapter", required=True, help="Frozen G_old that produced this rollout buffer.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--inference_schedule", default=None)
    parser.add_argument("--inference_init_mode", default=None)
    parser.add_argument("--inference_sigma_start", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
