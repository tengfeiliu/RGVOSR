import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.degradation_meta import to_jsonable  # noqa: E402
from models.prompt_builder import validate_prompt_profile  # noqa: E402
from tools.generate_unipercept_raw_cache import (  # noqa: E402
    UniPerceptRawAnalyzer,
    append_jsonl,
    build_result_from_unipercept_profile,
    list_hq_images,
    load_image_tensor,
    resolve_device_name,
    save_lq_tensor,
)


def stable_digest(*parts):
    payload = "\0".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_sample_id(source_path, crop_index, crop_size, crop_seed):
    source_path = str(Path(source_path).expanduser().resolve())
    return stable_digest(
        "rgvosr-precrop-v2",
        source_path,
        int(crop_index),
        int(crop_size),
        int(crop_seed),
    )[:32]


def stable_seed(sample_id, base_seed):
    return int(stable_digest(sample_id, int(base_seed))[:16], 16) % (2**63 - 1)


def crop_iou(first, second, crop_size):
    x1, y1 = first
    x2, y2 = second
    overlap_w = max(0, min(x1, x2) + crop_size - max(x1, x2))
    overlap_h = max(0, min(y1, y2) + crop_size - max(y1, y2))
    intersection = overlap_w * overlap_h
    union = 2 * crop_size * crop_size - intersection
    return float(intersection / union) if union else 0.0


def deterministic_crop_positions(
    width,
    height,
    source_key,
    crop_size=512,
    crops_per_image=2,
    crop_seed=42,
    max_crop_iou=0.25,
    crop_search_attempts=32,
):
    max_x = max(int(width) - int(crop_size), 0)
    max_y = max(int(height) - int(crop_size), 0)
    rng = random.Random(
        int(stable_digest(source_key, crop_seed, crop_size)[:16], 16)
    )

    first = (
        rng.randint(0, max_x) if max_x else 0,
        rng.randint(0, max_y) if max_y else 0,
    )
    positions = [
        {
            "x": first[0],
            "y": first[1],
            "iou_with_first": 0.0,
            "overlap_constraint_met": True,
        }
    ]

    for _crop_index in range(1, int(crops_per_image)):
        candidates = []
        attempts = max(int(crop_search_attempts), 1)
        for _ in range(attempts):
            candidate = (
                rng.randint(0, max_x) if max_x else 0,
                rng.randint(0, max_y) if max_y else 0,
            )
            iou = crop_iou(first, candidate, int(crop_size))
            candidates.append((iou, candidate))
            if iou <= float(max_crop_iou):
                break
        iou, candidate = min(candidates, key=lambda item: item[0])
        positions.append(
            {
                "x": candidate[0],
                "y": candidate[1],
                "iou_with_first": iou,
                "overlap_constraint_met": iou <= float(max_crop_iou),
            }
        )
    return positions


def resize_short_side(image, crop_size):
    original_size = image.size
    if min(original_size) >= int(crop_size):
        return image, original_size, 1.0
    scale = int(crop_size) / max(min(original_size), 1)
    resized_size = (
        max(int(round(image.width * scale)), int(crop_size)),
        max(int(round(image.height * scale)), int(crop_size)),
    )
    return (
        image.resize(resized_size, Image.Resampling.BICUBIC),
        resized_size,
        float(scale),
    )


@contextlib.contextmanager
def isolated_rng(seed):
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def atomic_save_rgb(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, output_path)


def atomic_save_lq(tensor, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.png")
    save_lq_tensor(tensor, temporary)
    os.replace(temporary, output_path)


def load_seen_sample_ids(*paths):
    seen = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = record.get("sample_id")
                if sample_id:
                    seen.add(str(sample_id))
    return seen


def validate_crop_profile(unipercept_raw):
    profile = (
        unipercept_raw.get("profile")
        if isinstance(unipercept_raw, dict)
        else None
    )
    validate_prompt_profile(
        profile,
        prompt_variant="iqa",
        include_caption=True,
    )
    return profile


def process_crop(
    source_path,
    crop_index,
    crop_position,
    source_image,
    original_size,
    resized_size,
    resize_scale,
    args,
    degradation,
    analyzer,
    device,
):
    sample_id = stable_sample_id(
        source_path,
        crop_index,
        args.crop_size,
        args.crop_seed,
    )
    degradation_seed = stable_seed(sample_id, args.degradation_seed)
    x = int(crop_position["x"])
    y = int(crop_position["y"])
    hq_crop = source_image.crop(
        (x, y, x + args.crop_size, y + args.crop_size)
    ).convert("RGB")

    hq_path = (Path(args.hq_output_dir) / f"{sample_id}.png").resolve()
    lq_path = (Path(args.lq_output_dir) / f"{sample_id}.png").resolve()
    atomic_save_rgb(hq_crop, hq_path)

    hq_tensor = load_image_tensor(hq_path, device)
    with isolated_rng(degradation_seed):
        _, lq_tensor, degradation_meta = degradation.degrade_process(
            hq_tensor,
            resize_bak=True,
            return_meta=True,
        )
    if tuple(lq_tensor.shape[-2:]) != (args.crop_size, args.crop_size):
        raise ValueError(
            f"Degradation returned LQ shape {tuple(lq_tensor.shape)} for {sample_id}; "
            f"expected RGB {args.crop_size}x{args.crop_size}"
        )
    atomic_save_lq(lq_tensor, lq_path)

    unipercept_raw = analyzer.analyze(lq_path)
    profile = validate_crop_profile(unipercept_raw)
    profile.setdefault("iaa", {})
    profile.setdefault("ista", {})
    profile.setdefault("suggestion", "")
    result = build_result_from_unipercept_profile(
        degradation_meta,
        unipercept_raw,
    )

    return {
        "schema_version": 2,
        "sample_id": sample_id,
        "source_hq_path": str(Path(source_path).expanduser().resolve()),
        "hq_path": str(hq_path),
        "lq_path": str(lq_path),
        "crop": {
            "crop_index": int(crop_index),
            "crop_size": int(args.crop_size),
            "original_size": [int(original_size[0]), int(original_size[1])],
            "resized_size": [int(resized_size[0]), int(resized_size[1])],
            "resize_scale": float(resize_scale),
            "x": x,
            "y": y,
            "crop_seed": int(args.crop_seed),
            "iou_with_first": float(crop_position["iou_with_first"]),
            "overlap_constraint_met": bool(
                crop_position["overlap_constraint_met"]
            ),
        },
        "degradation_seed": int(degradation_seed),
        "raw_degradation_params": to_jsonable(degradation_meta),
        "unipercept_raw": to_jsonable(unipercept_raw),
        "annotation_status": {
            "caption": "complete",
            "iaa": "pending",
            "iqa": "complete",
            "ista": "pending",
            "suggestion": "pending",
        },
        "result": result,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic pre-cropped HQ/LQ pairs and crop-local "
            "UniPercept caption+IQA annotations."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--hq-output-dir", required=True)
    parser.add_argument("--lq-output-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--invalid-output", required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--crops-per-image", type=int, default=2)
    parser.add_argument("--crop-seed", type=int, default=42)
    parser.add_argument("--max-crop-iou", type=float, default=0.25)
    parser.add_argument("--crop-search-attempts", type=int, default=32)
    parser.add_argument("--degradation-seed", type=int, default=42)
    parser.add_argument(
        "--profile-sections",
        nargs="+",
        choices=["caption", "iaa", "iqa", "ista"],
        default=["caption", "iqa"],
    )
    parser.add_argument(
        "--reward-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--opt-name", default="params_realsr.yml")
    parser.add_argument("--unipercept-repo", default="/data/code/UniPercept/")
    parser.add_argument("--unipercept-model-path", default="/data/models/UniPercept/")
    parser.add_argument(
        "--unipercept-backend",
        choices=["profile"],
        default="profile",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.crop_size <= 0 or args.crops_per_image <= 0:
        raise ValueError("--crop-size and --crops-per-image must be positive")
    if not 0.0 <= args.max_crop_iou <= 1.0:
        raise ValueError("--max-crop-iou must be between 0 and 1")
    if args.crop_search_attempts <= 0:
        raise ValueError("--crop-search-attempts must be positive")
    if set(args.profile_sections) != {"caption", "iqa"}:
        raise ValueError(
            "The pre-cropped training cache currently requires exactly "
            "--profile-sections caption iqa"
        )

    output = Path(args.output)
    invalid_output = Path(args.invalid_output)
    if output.resolve() == invalid_output.resolve():
        raise ValueError("--output and --invalid-output must be different files")
    if (
        Path(args.hq_output_dir).resolve()
        == Path(args.lq_output_dir).resolve()
    ):
        raise ValueError(
            "--hq-output-dir and --lq-output-dir must be different directories"
        )
    if args.overwrite:
        for path in (output, invalid_output):
            if path.exists():
                path.unlink()
    elif not args.resume and (output.exists() or invalid_output.exists()):
        raise FileExistsError(
            "Output JSONL already exists. Use --resume to continue or "
            "--overwrite to start a new cache."
        )
    seen = (
        load_seen_sample_ids(output, invalid_output)
        if args.resume and not args.overwrite
        else set()
    )

    images = list_hq_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]
    args.device = resolve_device_name(args.device)

    import torch
    from dataloaders.realesrgan_gpu import RealESRGAN_degradation

    device = torch.device(args.device)
    degradation = RealESRGAN_degradation(args.opt_name, device=device)
    analyzer = UniPerceptRawAnalyzer(
        device=args.device,
        model_path=args.unipercept_model_path,
        unipercept_repo=args.unipercept_repo,
        backend=args.unipercept_backend,
        profile_sections=args.profile_sections,
        include_reward_scores=args.reward_scores,
    )

    generated = 0
    skipped = 0
    failed = 0
    for source_path in images:
        source_sample_ids = [
            stable_sample_id(
                source_path,
                crop_index,
                args.crop_size,
                args.crop_seed,
            )
            for crop_index in range(args.crops_per_image)
        ]
        if all(sample_id in seen for sample_id in source_sample_ids):
            skipped += len(source_sample_ids)
            continue
        try:
            with Image.open(source_path) as image:
                image.load()
                source_image = image.convert("RGB")
            original_size = source_image.size
            source_image, resized_size, resize_scale = resize_short_side(
                source_image,
                args.crop_size,
            )
            positions = deterministic_crop_positions(
                source_image.width,
                source_image.height,
                source_key=str(Path(source_path).resolve()),
                crop_size=args.crop_size,
                crops_per_image=args.crops_per_image,
                crop_seed=args.crop_seed,
                max_crop_iou=args.max_crop_iou,
                crop_search_attempts=args.crop_search_attempts,
            )
        except Exception as exc:
            for crop_index in range(args.crops_per_image):
                sample_id = source_sample_ids[crop_index]
                if sample_id in seen:
                    skipped += 1
                    continue
                append_jsonl(
                    invalid_output,
                    {
                        "schema_version": 2,
                        "sample_id": sample_id,
                        "source_hq_path": str(source_path),
                        "crop_index": crop_index,
                        "reason": str(exc),
                    },
                )
                seen.add(sample_id)
                failed += 1
            continue

        for crop_index, crop_position in enumerate(positions):
            sample_id = stable_sample_id(
                source_path,
                crop_index,
                args.crop_size,
                args.crop_seed,
            )
            if sample_id in seen:
                skipped += 1
                continue
            try:
                record = process_crop(
                    source_path=source_path,
                    crop_index=crop_index,
                    crop_position=crop_position,
                    source_image=source_image,
                    original_size=original_size,
                    resized_size=resized_size,
                    resize_scale=resize_scale,
                    args=args,
                    degradation=degradation,
                    analyzer=analyzer,
                    device=device,
                )
                append_jsonl(output, record)
                generated += 1
            except Exception as exc:
                append_jsonl(
                    invalid_output,
                    {
                        "schema_version": 2,
                        "sample_id": sample_id,
                        "source_hq_path": str(Path(source_path).resolve()),
                        "crop_index": crop_index,
                        "reason": str(exc),
                    },
                )
                failed += 1
            seen.add(sample_id)

    print(
        f"[precrop] done: generated={generated}, skipped={skipped}, "
        f"failed={failed}, output={output}",
        flush=True,
    )
    if failed:
        raise RuntimeError(f"Pre-cropped cache generation failed for {failed} crops.")


if __name__ == "__main__":
    main()
