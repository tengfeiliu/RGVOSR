import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.prompt_builder import build_sr_prompt  # noqa: E402
from models.rg_flux_artist_factory import build_rg_flux_artist  # noqa: E402
from models.text_embedding_cache import (  # noqa: E402
    TextEmbeddingCache,
    normalize_image_key,
    validate_online_prompt_lengths,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def torch_dtype(name):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def iter_jsonl_records(jsonl_path, limit=None):
    count = 0
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[text-cache] skip invalid JSON at line {line_no}: {exc}", flush=True)
                continue
            yield record
            count += 1
            if limit is not None and count >= limit:
                break


def prompt_from_record(
    record,
    use_prompt=True,
    use_suggestions=True,
    prompt_variant=None,
    include_caption=False,
):
    unipercept_raw = record.get("unipercept_raw")
    unipercept_raw = unipercept_raw if isinstance(unipercept_raw, dict) else {}
    profile = unipercept_raw.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("record is missing unipercept_raw.profile")
    return build_sr_prompt(
        profile,
        use_prompt=use_prompt,
        use_suggestions=use_suggestions,
        prompt_variant=prompt_variant,
        include_caption=include_caption,
    )


def build_online_artist_config(config, device, dtype_name, output_dir):
    runtime_config = copy.deepcopy(config)
    runtime_config.setdefault("model", {})
    runtime_config.setdefault("training", {})
    runtime_config.setdefault("text_encoding", {})
    runtime_config["text_encoding"]["mode"] = "online"
    runtime_config["text_encoding"]["cache_dir"] = str(output_dir)
    runtime_config["text_encoding"]["dtype"] = dtype_name
    runtime_config["model"]["text_encoder_device"] = device
    runtime_config["model"]["use_lora"] = False
    runtime_config["training"]["freeze_flux_transformer"] = True
    return runtime_config


def main(args):
    config = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    if args.overwrite:
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

    runtime_config = build_online_artist_config(config, args.device, args.dtype, output_dir)
    cache = TextEmbeddingCache(
        cache_dir=output_dir,
        config=runtime_config,
        dtype=args.dtype,
        strict=True,
        validate_prompt_hash=True,
        load_existing=(args.resume or args.skip_existing) and not args.overwrite,
    )
    artist = build_rg_flux_artist(runtime_config)
    artist.eval()
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    use_prompt = bool(runtime_config.get("condition", {}).get("use_prompt", True))
    use_suggestions = bool(runtime_config.get("condition", {}).get("use_suggestions", True))
    prompt_variant = runtime_config.get("condition", {}).get("prompt_variant")
    include_caption = bool(
        runtime_config.get("condition", {}).get("include_caption", False)
    )

    generated = 0
    skipped = 0
    reused = 0
    failed = 0
    records = iter_jsonl_records(args.jsonl_path, limit=args.limit)
    for record in tqdm(records, desc="Caching RG-FLUX text embeddings"):
        lq_path = record.get("lq_path")
        hq_path = record.get("hq_path")
        if not lq_path:
            failed += 1
            print("[text-cache] skip record without lq_path", flush=True)
            continue
        image_key = normalize_image_key(lq_path)
        try:
            prompt = prompt_from_record(
                record,
                use_prompt=use_prompt,
                use_suggestions=use_suggestions,
                prompt_variant=prompt_variant,
                include_caption=include_caption,
            )
            existing = cache.find_record(prompt, image_key=image_key, allow_prompt_reuse=True)
            if args.skip_existing and existing is not None:
                if normalize_image_key(existing.get("image_key")) == image_key:
                    skipped += 1
                else:
                    cache.register_existing_embedding(existing, prompt, image_key, lq_path, hq_path)
                    reused += 1
                continue

            with torch.no_grad():
                validate_online_prompt_lengths(
                    artist,
                    [prompt],
                    runtime_config,
                )
                prompt_embeds, pooled_prompt_embeds, text_ids = artist.encode_prompts(
                    [prompt],
                    device=device,
                    dtype=dtype,
                )
            cache.save_embedding(
                prompt=prompt,
                image_key=image_key,
                lq_path=lq_path,
                hq_path=hq_path,
                state={
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "text_ids": text_ids,
                },
            )
            generated += 1
        except Exception as exc:
            failed += 1
            print(f"[text-cache] failed {lq_path}: {exc}", flush=True)

    print(
        f"[text-cache] done: generated={generated}, skipped={skipped}, reused={reused}, failed={failed}, "
        f"manifest={cache.manifest_path}",
        flush=True,
    )
    if failed:
        raise RuntimeError(f"Text embedding cache generation failed for {failed} records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--jsonl_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
