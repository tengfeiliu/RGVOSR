import argparse
import random
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.rg_flux_jsonl_dataset import RGFluxSRJsonlDataset, rg_flux_collate_fn  # noqa: E402
from models.flux_sr_artist import _load_state_dict_with_shape_check  # noqa: E402
from models.rg_flux_artist_factory import build_rg_flux_artist  # noqa: E402
from models.router_condition import condition_to_expert_target  # noqa: E402
from models.text_embedding_cache import get_text_embedding_cache, resolve_prompt_embeddings  # noqa: E402


PROTOTYPE_SIGMA_GRID = (1.0, 0.75, 0.5, 0.25, 0.0)


def expected_prototype_feature_count(source_sample_count, router_input_mode):
    multiplier = len(PROTOTYPE_SIGMA_GRID) if router_input_mode == "condition8_timestep" else 1
    return int(source_sample_count) * multiplier


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def torch_dtype(name):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def kmeans(features, num_clusters, num_iters=20):
    if features.shape[0] < num_clusters:
        repeats = (num_clusters + features.shape[0] - 1) // features.shape[0]
        features = features.repeat(repeats, 1)
    centers = features[:num_clusters].clone()
    for _ in range(int(num_iters)):
        distances = torch.cdist(features.float(), centers.float())
        labels = distances.argmin(dim=1)
        next_centers = []
        for idx in range(num_clusters):
            mask = labels == idx
            if mask.any():
                next_centers.append(features[mask].mean(dim=0))
            else:
                next_centers.append(centers[idx])
        centers = torch.stack(next_centers, dim=0)
    return centers


def load_condition_adapters(artist, single_lora_checkpoint):
    checkpoint_dir = Path(single_lora_checkpoint)
    if (checkpoint_dir / "rg_flux_adapters").exists():
        checkpoint_dir = checkpoint_dir / "rg_flux_adapters"
    adapter_state = checkpoint_dir / "condition_adapters.pt"
    if not adapter_state.exists():
        return False
    state = torch.load(adapter_state, map_location="cpu")
    _load_state_dict_with_shape_check(artist, state, adapter_state, "condition adapter")
    return True


@torch.no_grad()
def initialize_prototypes(artist, config, device, dtype, num_samples, text_embedding_cache):
    jsonl_path = config.get("data", {}).get("jsonl_path")
    if not jsonl_path or int(num_samples) <= 0:
        return {"source_sample_count": 0, "feature_count": 0}
    dataset = RGFluxSRJsonlDataset(
        jsonl_path=jsonl_path,
        crop_size=int(config.get("data", {}).get("crop_size", 256)),
        scale=int(config.get("data", {}).get("scale", 4)),
        mode="eval",
        use_prompt=bool(config.get("condition", {}).get("use_prompt", True)),
        use_suggestions=bool(config.get("condition", {}).get("use_suggestions", True)),
        prompt_variant=config.get("condition", {}).get("prompt_variant"),
        include_caption=bool(
            config.get("condition", {}).get("include_caption", False)
        ),
        use_degradation_vector=bool(config.get("condition", {}).get("use_degradation_vector", True)),
        use_router_condition=str(
            config.get("model", {}).get("lora_moe", {}).get("router_input_mode", "prompt_lr")
        )
        in {"condition8", "condition8_timestep"},
        router_condition_version=str(
            config.get("model", {}).get("lora_moe", {}).get(
                "router_condition_version",
                "text8_v1",
            )
        ),
        vae_align=int(config.get("data", {}).get("vae_align", 16)),
        pre_cropped=bool(config.get("data", {}).get("pre_cropped", True)),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=rg_flux_collate_fn)
    lr_cond_mode = config.get("condition", {}).get("lr_cond_mode", "latent_adapter")
    moe_config = config.get("model", {}).get("lora_moe", {}) or {}
    router_input_mode = str(moe_config.get("router_input_mode", "prompt_lr"))
    semantic_prototype_init = bool(moe_config.get("semantic_prototype_init", False))
    features = []
    teacher_targets = []
    teacher_confidences = []
    source_sample_count = 0
    for batch in loader:
        if source_sample_count >= int(num_samples):
            break
        source_sample_count += 1
        prompt_embeds = None
        z_lr = None
        if router_input_mode in {"prompt_lr", "prompt_only"}:
            prompt_embeds, _, _ = resolve_prompt_embeddings(
                artist=artist,
                prompts=batch["prompt"],
                image_keys=batch["lq_path"],
                config=config,
                device=device,
                dtype=dtype,
                cache=text_embedding_cache,
            )
        if router_input_mode == "prompt_lr":
            lq_up = batch["lq_up"].to(device=device, dtype=dtype)
            z_lr = artist.encode_images(
                lq_up,
                sample=lr_cond_mode != "flux2_image_concat",
            ).to(device=device, dtype=dtype)
        condition = batch.get("router_condition")
        condition_mask = batch.get("router_condition_mask")
        condition_confidence = batch.get("router_condition_confidence")
        if condition is not None:
            condition = condition.to(device=device, dtype=dtype)
            condition_mask = condition_mask.to(device=device, dtype=dtype)
            condition_confidence = condition_confidence.to(device=device, dtype=dtype)
        sigmas = PROTOTYPE_SIGMA_GRID if router_input_mode == "condition8_timestep" else (0.5,)
        for sigma_value in sigmas:
            feature = artist.compute_router_features(
                prompt_embeds=prompt_embeds,
                z_lr=z_lr,
                router_condition=condition,
                router_condition_mask=condition_mask,
                router_condition_confidence=condition_confidence,
                timestep=torch.full((1,), sigma_value, device=device, dtype=dtype),
            )
            features.append(feature.detach().float().cpu())
            if condition is not None and semantic_prototype_init:
                teacher_targets.append(
                    condition_to_expert_target(
                        condition * condition_mask,
                        temperature=float(moe_config.get("teacher_target_temperature", 0.7)),
                        score_matrix=moe_config.get("teacher_expert_score_matrix"),
                    ).detach().float().cpu()
                )
                teacher_confidences.append(condition_confidence.detach().float().cpu().reshape(-1))
    if not features:
        return {"source_sample_count": 0, "feature_count": 0}
    feature_tensor = torch.cat(features, dim=0)
    expected_count = expected_prototype_feature_count(source_sample_count, router_input_mode)
    if feature_tensor.shape[0] != expected_count:
        raise RuntimeError(
            f"Prototype feature count mismatch: got {feature_tensor.shape[0]}, expected {expected_count}"
        )
    if teacher_targets:
        target_tensor = torch.cat(teacher_targets, dim=0)
        confidence_tensor = torch.cat(teacher_confidences, dim=0).clamp(0.0, 1.0)
        weights = target_tensor * confidence_tensor.unsqueeze(-1)
        denominators = weights.sum(dim=0)
        weighted_sum = weights.transpose(0, 1) @ feature_tensor
        centers = weighted_sum / denominators.clamp_min(1e-6).unsqueeze(-1)
        under_supported = denominators < 1e-4
        if under_supported.any():
            valid_features = feature_tensor[confidence_tensor > 0]
            fallback_source = valid_features if valid_features.shape[0] else feature_tensor
            fallback = kmeans(fallback_source, artist.moe_router.num_experts)
            centers[under_supported] = fallback[under_supported]
            print(
                "[init_flux2_lora_moe] warning: semantic prototype fallback for experts "
                f"{under_supported.nonzero(as_tuple=False).flatten().tolist()}",
                flush=True,
            )
    else:
        centers = kmeans(feature_tensor, artist.moe_router.num_experts)
    artist.moe_router.prototypes.copy_(centers.to(device=artist.moe_router.prototypes.device, dtype=artist.moe_router.prototypes.dtype))
    return {
        "source_sample_count": source_sample_count,
        "feature_count": int(feature_tensor.shape[0]),
    }


def main(args):
    config = load_yaml(args.config)
    seed = int(
        args.seed
        if args.seed is not None
        else config.get("_runtime", {}).get(
            "moe_init_seed",
            config.get("training", {}).get("seed", 42),
        )
    )
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config.setdefault("_runtime", {})["moe_init_seed"] = seed
    expert_init_seed = int(args.expert_init_seed if args.expert_init_seed is not None else seed)
    config["_runtime"]["moe_expert_init_seed"] = expert_init_seed
    config.setdefault("model", {})
    config["model"]["flux_backend"] = "flux2_klein"
    config["model"]["lora_backend"] = "moe"

    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    artist = build_rg_flux_artist(config).to(device=device)
    # Router modes instantiate different module graphs and therefore consume a
    # different amount of RNG. Reset immediately before routed-A perturbation so
    # S1-S5 share identical expert residual noise under the same single LoRA.
    random.seed(expert_init_seed)
    torch.manual_seed(expert_init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(expert_init_seed)
    artist.initialize_moe_from_single_lora(args.single_lora_checkpoint, perturb_scale=args.perturb_scale)
    load_condition_adapters(artist, args.single_lora_checkpoint)
    text_encoding_config = config.get("text_encoding", {}) or {}
    text_embedding_cache = get_text_embedding_cache(
        config,
        dtype=text_encoding_config.get("dtype") or args.dtype,
    )

    initialized = initialize_prototypes(
        artist,
        config,
        device,
        dtype,
        args.prototype_num_samples,
        text_embedding_cache,
    )
    print(
        "[init_flux2_lora_moe] initialized prototypes from "
        f"{initialized['source_sample_count']} source samples / "
        f"{initialized['feature_count']} router features",
        flush=True,
    )

    output = Path(args.output)
    artist.save_trainable(output, save_files=True)
    print(f"[init_flux2_lora_moe] wrote MoE adapter checkpoint to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a FLUX.2-klein LoRA-MoE checkpoint from a single-LoRA baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--single_lora_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prototype_num_samples", type=int, default=128)
    parser.add_argument("--perturb_scale", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--expert_init_seed", type=int, default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    main(parser.parse_args())
