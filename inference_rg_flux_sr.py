import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from dataloaders.degradation_meta import DEGRADATION_KEYS
from models.rg_flux_artist_factory import build_rg_flux_artist
from models.prompt_builder import build_sr_prompt
from rg_flux_fm import sample_multistep_fm


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def cfg(config, path, default=None):
    current = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(checkpoint, explicit_config=None):
    if explicit_config:
        return load_yaml(explicit_config)
    cur = Path(checkpoint).resolve()
    for parent in [cur, *cur.parents]:
        args_json = parent / "args.json"
        if args_json.exists():
            with args_json.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return load_yaml("configs/train_rg_flux_sr_ms.yaml")


def list_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() == ".txt":
            return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.suffix.lower() in IMG_EXTENSIONS)
    return []


def load_jsonl_results(jsonl_path):
    if not jsonl_path:
        return {}
    index = {}
    path = Path(jsonl_path)
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            for key in ("lq_path", "hq_path"):
                value = record.get(key)
                if value:
                    index[str(Path(value))] = result
                    index[Path(value).name] = result
    return index


def result_for_image(result_index, image_path):
    return result_index.get(str(image_path)) or result_index.get(image_path.name) or {}


def degradation_tensor(result, device, dtype, use_degradation_vector=True):
    vector = result.get("degradation_vector") if isinstance(result, dict) else {}
    vector = vector if isinstance(vector, dict) and use_degradation_vector else {}
    values = [float(vector.get(key, 0.0) or 0.0) for key in DEGRADATION_KEYS]
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0)


def prepare_lq_up(image_path, upscale, align=16, min_size=None):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), Image.Resampling.BICUBIC)
    if min_size and min(image.size) < min_size:
        ratio = min_size / max(min(image.size), 1)
        image = image.resize((round(image.width * ratio), round(image.height * ratio)), Image.Resampling.BICUBIC)
    width = max(align, image.width - image.width % align)
    height = max(align, image.height - image.height % align)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    tensor = to_tensor(image).unsqueeze(0).mul(2.0).sub(1.0)
    return image, original_size, tensor


def main(args):
    config = load_config(args.checkpoint, args.config)
    config.setdefault("condition", {})
    config["condition"]["lr_cond_mode"] = args.lr_cond_mode or cfg(config, "condition.lr_cond_mode", "latent_adapter")
    config["condition"]["use_prompt"] = args.use_prompt
    config["condition"]["use_degradation_vector"] = args.use_degradation_vector

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    artist = build_rg_flux_artist(config).to(device=device)
    artist.load_trainable(args.checkpoint, is_trainable=False)
    artist.eval()

    result_index = load_jsonl_results(args.jsonl_path)
    image_paths = list_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No input images found: {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()

    for image_path in tqdm(image_paths, desc="RG-FLUX-SR inference"):
        result = result_for_image(result_index, image_path)
        prompt = build_sr_prompt(
            result,
            use_prompt=args.use_prompt,
            use_suggestions=args.use_suggestions,
        )
        lq_up_pil, original_size, lq_up = prepare_lq_up(
            image_path,
            upscale=args.upscale,
            align=int(cfg(config, "data.vae_align", 16)),
            min_size=args.min_size,
        )
        lq_up = lq_up.to(device=device, dtype=dtype)

        with torch.no_grad():
            z_lr = artist.encode_images(lq_up).to(device=device, dtype=dtype)
            prompt_embeds, pooled_prompt_embeds, text_ids = artist.encode_prompts([prompt], device=device, dtype=dtype)
            degradation_vector = degradation_tensor(result, device, dtype, args.use_degradation_vector)
            dino_tokens = artist.extract_visual_tokens(lq_up)
            sr_latent = sample_multistep_fm(
                artist=artist,
                shape=tuple(z_lr.shape),
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                degradation_vector=degradation_vector,
                z_lr=z_lr,
                dino_tokens=dino_tokens,
                lr_cond_mode=config["condition"]["lr_cond_mode"],
                num_steps=args.num_inference_steps,
                device=device,
                dtype=dtype,
            )
            sr = artist.decode_latents(sr_latent).clamp(-1, 1).add(1.0).mul(0.5).clamp(0, 1)

        out_image = to_pil(sr[0].float().cpu())
        if args.restore_input_size:
            out_image = out_image.resize((original_size[0] * args.upscale, original_size[1] * args.upscale), Image.Resampling.LANCZOS)
        out_image.save(output_dir / f"{image_path.stem}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input LQ image, folder, or txt list.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="RG-FLUX-SR-MS adapter checkpoint directory.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--jsonl_path", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--lr_cond_mode", choices=["latent_adapter", "latent_concat"], default=None)
    parser.add_argument("--use_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_suggestions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_degradation_vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--min_size", type=int, default=None)
    parser.add_argument("--restore_input_size", action="store_true")
    main(parser.parse_args())
