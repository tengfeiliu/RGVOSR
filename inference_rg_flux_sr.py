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
from models.prompt_builder import PROMPT_VARIANTS, build_sr_prompt
from models.text_embedding_cache import get_text_embedding_cache, resolve_prompt_embeddings
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


TORCH_DTYPE_BY_NAME = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_inference_dtype(config, args_dtype):
    requested_dtype = str(args_dtype or "bf16").strip().lower()
    model_dtype = cfg(config, "model.dtype", None)
    effective_dtype = str(model_dtype or requested_dtype).strip().lower()
    if effective_dtype not in TORCH_DTYPE_BY_NAME:
        raise ValueError(f"Unsupported inference dtype: {effective_dtype}")
    if model_dtype is not None and requested_dtype != effective_dtype:
        print(
            f"Warning: --dtype {requested_dtype} differs from config model.dtype {effective_dtype}; "
            f"using model dtype {effective_dtype} for FLUX inference.",
            flush=True,
        )
    config.setdefault("model", {})["dtype"] = effective_dtype
    return TORCH_DTYPE_BY_NAME[effective_dtype], effective_dtype


def list_images(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() == ".txt":
            return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path]
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.suffix.lower() in IMG_EXTENSIONS)
    return []


def parse_dataset_dirs(dataset_dirs):
    datasets = []
    for item in dataset_dirs or []:
        if "=" not in item:
            raise ValueError(f"--dataset_dirs entries must use name=folder_path format, got: {item}")
        dataset_name, input_path = item.split("=", 1)
        dataset_name = dataset_name.strip()
        input_path = input_path.strip()
        if not dataset_name or not input_path:
            raise ValueError(f"--dataset_dirs entries must use name=folder_path format, got: {item}")
        datasets.append((dataset_name, Path(input_path)))
    return datasets


def resolve_inference_datasets(args, output_dir=None):
    output_dir = Path(output_dir if output_dir is not None else args.output_dir)
    if args.dataset_dirs:
        datasets = []
        for dataset_name, input_path in parse_dataset_dirs(args.dataset_dirs):
            datasets.append((dataset_name, input_path, output_dir / dataset_name))
        return datasets
    if args.input:
        return [("default", Path(args.input), output_dir)]
    raise ValueError("Either --input or --dataset_dirs is required.")


def format_checkpoint_step(checkpoint_step):
    value = str(checkpoint_step or "").strip()
    if not value:
        raise ValueError("--checkpoint_step is required when --run_dir is used.")
    if value.lower() == "latest":
        return "latest"
    if value.startswith("checkpoint-"):
        value = value[len("checkpoint-") :]
    try:
        step = int(value)
    except ValueError as exc:
        raise ValueError(f"--checkpoint_step must be an integer step, checkpoint-XXXXXXXX, or latest: {checkpoint_step}") from exc
    if step < 0:
        raise ValueError(f"--checkpoint_step must be non-negative: {checkpoint_step}")
    return f"checkpoint-{step:08d}"


def find_latest_run_checkpoint(run_dir):
    checkpoint_root = Path(run_dir) / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Run checkpoint directory does not exist: {checkpoint_root}")
    candidates = sorted(path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under: {checkpoint_root}")
    return candidates[-1]


def infer_checkpoint_step(checkpoint_path):
    path = Path(checkpoint_path)
    for item in [path, *path.parents]:
        if item.name.startswith("checkpoint-"):
            return item.name
    return None


def default_run_inference_dir(run_dir, checkpoint_name):
    return Path(run_dir) / "inference" / checkpoint_name


def validate_checkpoint_adapter(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint adapter directory does not exist: {checkpoint}")
    return checkpoint


def resolve_inference_run(args):
    if args.run_dir:
        if args.checkpoint:
            raise ValueError("--run_dir cannot be combined with --checkpoint.")
        if args.output_root and args.output_dir:
            raise ValueError("--output_root cannot be combined with --output_dir when --run_dir is used.")
        checkpoint_step = format_checkpoint_step(args.checkpoint_step)
        run_dir = Path(args.run_dir)
        checkpoint_dir = (
            find_latest_run_checkpoint(run_dir)
            if checkpoint_step == "latest"
            else run_dir / "checkpoints" / checkpoint_step
        )
        checkpoint = validate_checkpoint_adapter(checkpoint_dir / "rg_flux_adapters")
        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif args.output_root:
            output_dir = Path(args.output_root) / run_dir.name / checkpoint_dir.name
        else:
            output_dir = default_run_inference_dir(run_dir, checkpoint_dir.name)
        return {
            "run_dir": run_dir,
            "checkpoint": checkpoint,
            "checkpoint_step": checkpoint_dir.name,
            "output_dir": output_dir,
        }

    if args.checkpoint_step or args.output_root:
        raise ValueError("--checkpoint_step and --output_root require --run_dir.")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --run_dir is used.")
    if not args.output_dir:
        raise ValueError("--output_dir is required unless --run_dir is used.")
    checkpoint = validate_checkpoint_adapter(args.checkpoint)
    return {
        "run_dir": None,
        "checkpoint": checkpoint,
        "checkpoint_step": infer_checkpoint_step(checkpoint),
        "output_dir": Path(args.output_dir),
    }


def write_inference_manifest(manifest_path, run_dir, checkpoint_step, checkpoint_path, output_dir, datasets):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir) if run_dir is not None else None,
        "checkpoint_step": checkpoint_step,
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "datasets": [
            {
                "name": dataset_name,
                "input_path": str(input_path),
                "output_dir": str(dataset_output_dir),
            }
            for dataset_name, input_path, dataset_output_dir in datasets
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def load_jsonl_conditions(jsonl_path):
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
            profile = None
            unipercept_raw = record.get("unipercept_raw")
            if isinstance(unipercept_raw, dict) and isinstance(unipercept_raw.get("profile"), dict):
                profile = unipercept_raw["profile"]
            result = record.get("result")
            if not isinstance(result, dict):
                result = {}
            condition = {
                "profile": profile,
                "result": result,
                "record": record,
            }
            dataset_name = record.get("dataset_name") or record.get("dataset")
            for key in ("lq_path", "hq_path", "image_path", "path"):
                value = record.get(key)
                if value:
                    for alias in path_lookup_aliases(value, dataset_name=dataset_name):
                        index[alias] = condition
    return index


def _normalize_lookup_path(value):
    return str(value).replace("\\", "/")


def path_lookup_aliases(value, dataset_name=None):
    path = Path(value)
    normalized = _normalize_lookup_path(value)
    parts = [part for part in normalized.split("/") if part]
    aliases = []

    def add(alias):
        if alias and alias not in aliases:
            aliases.append(alias)

    add(normalized)
    if dataset_name:
        add(f"{dataset_name}/{path.name}")
        add(f"{dataset_name}/lq/{path.name}")
    for start in range(len(parts)):
        add("/".join(parts[start:]))
    add(path.name)
    return aliases


def extend_lookup_aliases(target, aliases):
    for alias in aliases:
        if alias not in target:
            target.append(alias)


def image_lookup_aliases(image_path, dataset_name=None, input_root=None):
    aliases = []
    extend_lookup_aliases(aliases, path_lookup_aliases(image_path, dataset_name=dataset_name))
    image_path = Path(image_path)
    if input_root is not None:
        try:
            rel_path = image_path.relative_to(Path(input_root))
        except ValueError:
            rel_path = None
        if rel_path is not None:
            extend_lookup_aliases(aliases, path_lookup_aliases(rel_path, dataset_name=dataset_name))
    return aliases


def condition_for_image(condition_index, image_path, dataset_name=None, input_root=None):
    for alias in image_lookup_aliases(image_path, dataset_name=dataset_name, input_root=input_root):
        condition = condition_index.get(alias)
        if condition is not None:
            return condition
    return None


def append_inference_failure(log_path, image_path, reason, condition=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(image_path),
        "reason": reason,
    }
    record = condition.get("record") if isinstance(condition, dict) else None
    if isinstance(record, dict):
        for key in ("lq_path", "hq_path"):
            value = record.get(key)
            if value:
                payload[key] = value
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def run_inference_dataset(
    dataset_name,
    input_path,
    output_dir,
    artist,
    config,
    args,
    condition_index,
    text_embedding_cache,
    device,
    dtype,
    lr_cond_mode,
):
    image_paths = list_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No input images found for dataset '{dataset_name}': {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log_path = output_dir / "inference_failures.jsonl"
    to_pil = transforms.ToPILImage()

    for image_path in tqdm(image_paths, desc=f"RG-FLUX-SR inference [{dataset_name}]"):
        condition = None
        if args.jsonl_path:
            condition = condition_for_image(
                condition_index,
                image_path,
                dataset_name=dataset_name,
                input_root=input_path,
            )
            if condition is None:
                append_inference_failure(
                    failure_log_path,
                    image_path=image_path,
                    reason="missing_jsonl_match",
                    condition=None,
                )
                print(f"[inference] skipped {image_path}: missing JSONL match", flush=True)
                continue
            profile = condition.get("profile")
            if not isinstance(profile, dict):
                append_inference_failure(
                    failure_log_path,
                    image_path=image_path,
                    reason="missing_unipercept_raw.profile",
                    condition=condition,
                )
                print(f"[inference] skipped {image_path}: missing unipercept_raw.profile", flush=True)
                continue
            result = condition.get("result") if isinstance(condition.get("result"), dict) else {}
        else:
            profile = {}
            result = {}
        prompt = build_sr_prompt(
            profile,
            use_prompt=args.use_prompt,
            use_suggestions=args.use_suggestions,
            prompt_variant=args.prompt_variant,
        )
        lq_up_pil, original_size, lq_up = prepare_lq_up(
            image_path,
            upscale=args.upscale,
            align=int(cfg(config, "data.vae_align", 16)),
            min_size=args.min_size,
        )
        lq_up = lq_up.to(device=device, dtype=dtype)

        with torch.no_grad():
            z_lr = artist.encode_images(
                lq_up,
                sample=lr_cond_mode != "flux2_image_concat",
            ).to(device=device, dtype=dtype)
            cache_image_key = str(image_path)
            if isinstance(condition, dict):
                record = condition.get("record")
                if isinstance(record, dict) and record.get("lq_path"):
                    cache_image_key = record["lq_path"]
            prompt_embeds, pooled_prompt_embeds, text_ids = resolve_prompt_embeddings(
                artist=artist,
                prompts=[prompt],
                image_keys=[cache_image_key],
                config=config,
                device=device,
                dtype=dtype,
                cache=text_embedding_cache,
            )
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
                lr_cond_mode=lr_cond_mode,
                num_steps=args.num_inference_steps,
                device=device,
                dtype=dtype,
            )
            sr = artist.decode_latents(sr_latent).clamp(-1, 1).add(1.0).mul(0.5).clamp(0, 1)

        out_image = to_pil(sr[0].float().cpu())
        if args.restore_input_size:
            out_image = out_image.resize((original_size[0] * args.upscale, original_size[1] * args.upscale), Image.Resampling.LANCZOS)
        out_image.save(output_dir / f"{image_path.stem}.png")


def main(args):
    resolved_run = resolve_inference_run(args)
    config = load_config(resolved_run["checkpoint"], args.config)
    config.setdefault("condition", {})
    config.setdefault("text_encoding", {})
    config["condition"]["lr_cond_mode"] = args.lr_cond_mode or cfg(config, "condition.lr_cond_mode", "latent_adapter")
    config["condition"]["use_prompt"] = args.use_prompt
    config["condition"]["use_degradation_vector"] = args.use_degradation_vector
    config["condition"]["use_suggestions"] = args.use_suggestions
    if args.prompt_variant is None:
        args.prompt_variant = cfg(config, "condition.prompt_variant", None)
    else:
        config["condition"]["prompt_variant"] = args.prompt_variant
    lr_cond_mode = config["condition"]["lr_cond_mode"]
    if args.text_encoding_mode is not None:
        config["text_encoding"]["mode"] = args.text_encoding_mode
    if args.text_embedding_cache is not None:
        config["text_encoding"]["cache_dir"] = args.text_embedding_cache

    dtype, dtype_name = resolve_inference_dtype(config, args.dtype)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    artist = build_rg_flux_artist(config).to(device=device)
    artist.load_trainable(resolved_run["checkpoint"], is_trainable=False)
    if hasattr(artist, "align_inference_dtype"):
        artist.align_inference_dtype(dtype=dtype)
    artist.eval()
    text_embedding_cache = get_text_embedding_cache(
        config,
        dtype=cfg(config, "text_encoding.dtype", dtype_name),
    )
    if hasattr(artist, "set_moe_training_schedule"):
        artist.set_moe_training_schedule(global_step=1, max_steps=1)

    condition_index = load_jsonl_conditions(args.jsonl_path)
    datasets = resolve_inference_datasets(args, output_dir=resolved_run["output_dir"])
    for dataset_name, input_path, output_dir in datasets:
        run_inference_dataset(
            dataset_name=dataset_name,
            input_path=input_path,
            output_dir=output_dir,
            artist=artist,
            config=config,
            args=args,
            condition_index=condition_index,
            text_embedding_cache=text_embedding_cache,
            device=device,
            dtype=dtype,
            lr_cond_mode=lr_cond_mode,
        )
    write_inference_manifest(
        manifest_path=resolved_run["output_dir"] / "inference_manifest.json",
        run_dir=resolved_run["run_dir"],
        checkpoint_step=resolved_run["checkpoint_step"],
        checkpoint_path=resolved_run["checkpoint"],
        output_dir=resolved_run["output_dir"],
        datasets=datasets,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", default=None, help="Input LQ image, folder, or txt list.")
    input_group.add_argument(
        "--dataset_dirs",
        nargs="+",
        default=None,
        help="Multiple datasets as name=folder_path entries. Outputs are written to output_dir/name.",
    )
    parser.add_argument("--output_dir", default=None, help="Legacy direct output directory. Required with --checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Legacy RG-FLUX-SR-MS adapter checkpoint directory.")
    parser.add_argument("--run_dir", default=None, help="Experiment run directory containing checkpoints/ and args.json.")
    parser.add_argument(
        "--checkpoint_step",
        default=None,
        help="Checkpoint step used with --run_dir, e.g. 32000, checkpoint-00032000, or latest.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "Optional output root used with --run_dir. Results are written to "
            "output_root/run_name/checkpoint-XXXXXXXX. If omitted, outputs default to "
            "run_dir/inference/checkpoint-XXXXXXXX."
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--jsonl_path", default=None)
    parser.add_argument("--text_encoding_mode", choices=["online", "cached", "auto"], default=None)
    parser.add_argument("--text_embedding_cache", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument(
        "--lr_cond_mode",
        choices=["latent_adapter", "latent_concat", "flux2_image_concat"],
        default=None,
    )
    parser.add_argument("--use_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_suggestions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt_variant", choices=PROMPT_VARIANTS, default=None)
    parser.add_argument("--use_degradation_vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--min_size", type=int, default=None)
    parser.add_argument("--restore_input_size", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
