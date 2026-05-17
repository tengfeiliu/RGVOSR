import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.degradation_meta import (  # noqa: E402
    default_semantic_result,
    make_cache_record,
    merge_analysis_result,
    read_jsonl_paths,
)
from tools.qwen_semantic_risk_analyzer import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    analyze_image,
    list_images,
)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def stable_lq_name(image_path):
    image_path = Path(image_path)
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{digest}_{image_path.stem}.png"


def load_image_tensor(image_path, device):
    from torchvision import transforms

    with Image.open(image_path) as image:
        image = image.convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return tensor


def save_lq_tensor(tensor, output_path):
    from torchvision import transforms

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().clamp(0, 1).squeeze(0).cpu()
    transforms.ToPILImage()(tensor).save(output_path)


def process_image(image_path, args, degradation, device):
    hq = load_image_tensor(image_path, device)
    _, lq, meta = degradation.degrade_process(hq, resize_bak=args.resize_bak, return_meta=True)

    lq_path = Path(args.lq_output_dir) / stable_lq_name(image_path)
    save_lq_tensor(lq, lq_path)

    physical_vector = meta.get("degradation_vector", {})
    if args.skip_qwen:
        semantic_result = default_semantic_result(physical_vector)
        raw_qwen_response = ""
    else:
        semantic_result, raw_qwen_response = analyze_image(
            lq_path,
            model=args.model,
            base_url=args.base_url,
        )

    result = merge_analysis_result(physical_vector, semantic_result)
    record = make_cache_record(
        hq_path=str(image_path),
        lq_path=str(lq_path),
        raw_degradation_params=meta,
        result=result,
    )
    if raw_qwen_response:
        record["raw_qwen_response"] = raw_qwen_response
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic LQ images plus RealESRGAN/Qwen-VL degradation-analysis JSONL."
    )
    parser.add_argument("--input", required=False, default='/data/datasets/LSDIR/shard-00.txt', help="HQ image path, directory, or txt list.")
    parser.add_argument("--lq-output-dir", required=False, default='/data/datasets/LSDIR_lq', help="Directory for generated LQ PNG images.")
    parser.add_argument("--output", required=False, default='datasets/LSDIR_cache/valid.jsonl', help="Valid merged cache JSONL path.")
    parser.add_argument("--invalid-output", required=False, default='datasets/LSDIR_cache/invalid.jsonl', help="Invalid/error JSONL path.")
    parser.add_argument("--opt-name", default="params_realsr.yml", help="RealESRGAN degradation YAML under dataloaders/.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true sk-a322b14b4b014d288bb02e98ed5ddc19")
    parser.add_argument("--resize-bak", action="store_true", default=True)
    parser.add_argument("--no-resize-bak", dest="resize_bak", action="store_false")
    parser.add_argument("--skip-qwen", action="store_true", help="Use physical-only fallback semantic fields.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    images = list_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]

    seen = set()
    if args.resume:
        seen.update(read_jsonl_paths(args.output, key="hq_path"))
        seen.update(read_jsonl_paths(args.invalid_output, key="hq_path"))

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    from dataloaders.realesrgan_gpu import RealESRGAN_degradation

    degradation = RealESRGAN_degradation(args.opt_name, device=device)

    for image_path in images:
        image_key = str(image_path)
        if image_key in seen:
            continue
        try:
            record = process_image(image_path, args, degradation, device)
            append_jsonl(args.output, record)
        except Exception as exc:
            append_jsonl(
                args.invalid_output,
                {
                    "hq_path": image_key,
                    "reason": str(exc),
                },
            )


if __name__ == "__main__":
    main()
