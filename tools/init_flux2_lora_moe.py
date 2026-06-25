import argparse
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
from models.text_embedding_cache import get_text_embedding_cache, resolve_prompt_embeddings  # noqa: E402


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
        return 0
    dataset = RGFluxSRJsonlDataset(
        jsonl_path=jsonl_path,
        crop_size=int(config.get("data", {}).get("crop_size", 256)),
        scale=int(config.get("data", {}).get("scale", 4)),
        mode="eval",
        use_prompt=bool(config.get("condition", {}).get("use_prompt", True)),
        use_suggestions=bool(config.get("condition", {}).get("use_suggestions", True)),
        use_degradation_vector=bool(config.get("condition", {}).get("use_degradation_vector", True)),
        vae_align=int(config.get("data", {}).get("vae_align", 16)),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=rg_flux_collate_fn)
    lr_cond_mode = config.get("condition", {}).get("lr_cond_mode", "latent_adapter")
    features = []
    for batch in loader:
        if len(features) >= int(num_samples):
            break
        lq_up = batch["lq_up"].to(device=device, dtype=dtype)
        z_lr = artist.encode_images(
            lq_up,
            sample=lr_cond_mode != "flux2_image_concat",
        ).to(device=device, dtype=dtype)
        prompt_embeds, _, _ = resolve_prompt_embeddings(
            artist=artist,
            prompts=batch["prompt"],
            image_keys=batch["lq_path"],
            config=config,
            device=device,
            dtype=dtype,
            cache=text_embedding_cache,
        )
        features.append(artist.compute_router_features(prompt_embeds, z_lr).detach().float().cpu())
    if not features:
        return 0
    feature_tensor = torch.cat(features, dim=0)
    centers = kmeans(feature_tensor, artist.moe_router.num_experts)
    artist.moe_router.prototypes.copy_(centers.to(device=artist.moe_router.prototypes.device, dtype=artist.moe_router.prototypes.dtype))
    return feature_tensor.shape[0]


def main(args):
    config = load_yaml(args.config)
    config.setdefault("model", {})
    config["model"]["flux_backend"] = "flux2_klein"
    config["model"]["lora_backend"] = "moe"

    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    artist = build_rg_flux_artist(config).to(device=device)
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
    print(f"[init_flux2_lora_moe] initialized prototypes from {initialized} samples", flush=True)

    output = Path(args.output)
    artist.save_trainable(output, save_files=True)
    print(f"[init_flux2_lora_moe] wrote MoE adapter checkpoint to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a FLUX.2-klein LoRA-MoE checkpoint from a single-LoRA baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--single_lora_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prototype_num_samples", type=int, default=128)
    parser.add_argument("--perturb_scale", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    main(parser.parse_args())
