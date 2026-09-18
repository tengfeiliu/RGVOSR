import csv
import contextlib
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


DEFAULT_OMGSR_METRICS = ["clipiqa", "clipiqa+", "nima", "niqe", "liqe", "musiq", "maniqa-pipal"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LOWER_BETTER_FALLBACKS = {"niqe", "brisque", "piqe", "ilniqe"}
# PyIQA NIQE evaluates 96x96 blocks at two scales. A short side below 192
# becomes an empty block grid at the second scale. Only the metric input is
# enlarged; saved SR images and inputs to every other metric remain untouched.
METRIC_MIN_SHORT_SIDE = {"niqe": 192}


def parse_name_path(values, flag_name):
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag_name} entry must use name=path format: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name:
            raise ValueError(f"{flag_name} contains an empty dataset name: {value}")
        if not raw_path:
            raise ValueError(f"{flag_name} contains an empty path for dataset '{name}'")
        if name in parsed:
            raise ValueError(f"Duplicate dataset name in {flag_name}: {name}")
        parsed[name] = Path(raw_path).expanduser()
    return parsed


def parse_expected_counts(values):
    if not values:
        return {}
    parsed = parse_name_path(values, "--expected_counts")
    counts = {}
    for name, raw_count in parsed.items():
        count = int(str(raw_count))
        if count < 0:
            raise ValueError(f"Expected count for '{name}' must be non-negative: {count}")
        counts[name] = count
    return counts


def collect_images(dataset_dirs):
    images_by_dataset = {}
    for dataset, directory in dataset_dirs.items():
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Dataset directory does not exist for '{dataset}': {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Dataset path must be a directory for '{dataset}': {directory}")
        images = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No images found for '{dataset}' under {directory}")
        images_by_dataset[dataset] = images
    return images_by_dataset


def sample_images(images_by_dataset, max_samples_per_dataset=None, sample_seed=42):
    """Select a portable deterministic metric subset for each dataset.

    Selection depends on dataset name, image filename and the explicit seed;
    absolute server paths never affect which images are chosen.  ``None`` means
    all images.  Expected dataset counts must be checked before calling this.
    """
    if max_samples_per_dataset is None:
        return {name: list(images) for name, images in images_by_dataset.items()}
    limit = int(max_samples_per_dataset)
    if limit <= 0:
        raise ValueError("max_samples_per_dataset must be positive or None")
    selected = {}
    for dataset, images in images_by_dataset.items():
        ranked = sorted(
            images,
            key=lambda path: hashlib.sha256(
                f"{int(sample_seed)}:{dataset}:{Path(path).name}".encode("utf-8")
            ).hexdigest(),
        )
        chosen = set(ranked[:limit])
        selected[dataset] = [path for path in images if path in chosen]
    return selected


def validate_expected_counts(images_by_dataset, expected_counts):
    for dataset, expected_count in expected_counts.items():
        if dataset not in images_by_dataset:
            raise ValueError(f"--expected_counts includes '{dataset}', but --dataset_dirs does not")
        actual_count = len(images_by_dataset[dataset])
        if actual_count != expected_count:
            raise ValueError(f"Image count mismatch for '{dataset}': expected {expected_count}, found {actual_count}")


def get_image_size(path):
    with Image.open(path) as image:
        return image.size


def score_to_float(score):
    if hasattr(score, "detach"):
        score = score.detach().cpu()
        if score.numel() != 1:
            score = score.reshape(-1).mean()
        return float(score.item())
    if isinstance(score, dict):
        for key in ("score", "quality", "value"):
            if key in score:
                return score_to_float(score[key])
    if isinstance(score, (list, tuple)):
        numeric_values = [score_to_float(item) for item in score]
        return float(sum(numeric_values) / len(numeric_values))
    return float(score)


def metric_direction(metric_name, metric):
    lower_better = getattr(metric, "lower_better", None)
    if lower_better is None:
        lower_better = metric_name.lower() in LOWER_BETTER_FALLBACKS
    return "lower_better" if lower_better else "higher_better"


def mean(values):
    return sum(values) / len(values)


def population_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def build_rows(images_by_dataset):
    rows = []
    for dataset, images in images_by_dataset.items():
        for image_path in images:
            width, height = get_image_size(image_path)
            rows.append(
                {
                    "dataset": dataset,
                    "filename": image_path.name,
                    "path": str(image_path),
                    "width": width,
                    "height": height,
                }
            )
    return rows


def metric_resize_size(width, height, metric_name):
    min_short_side = METRIC_MIN_SHORT_SIDE.get(str(metric_name).lower())
    width, height = int(width), int(height)
    if min_short_side is None or min(width, height) >= min_short_side:
        return None
    scale = float(min_short_side) / float(min(width, height))
    return (
        max(min_short_side, int(math.ceil(width * scale))),
        max(min_short_side, int(math.ceil(height * scale))),
    )


def metric_input(row, metric_name, torch_module):
    resized_size = metric_resize_size(row["width"], row["height"], metric_name)
    if resized_size is None or torch_module is None:
        return row["path"]

    import numpy as np

    with Image.open(row["path"]) as source:
        image = source.convert("RGB")
        image = image.resize(resized_size, Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch_module.from_numpy(array).permute(2, 0, 1).unsqueeze(0).div_(255.0)


def evaluate_metrics(rows, metrics, device):
    import pyiqa
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if str(device).startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError(f"Requested device '{device}', but CUDA is not available")

    directions = {}
    no_grad = torch.no_grad if torch is not None else contextlib.nullcontext
    with no_grad():
        for metric_name in metrics:
            metric = pyiqa.create_metric(metric_name, device=device)
            if hasattr(metric, "eval"):
                metric.eval()
            directions[metric_name] = metric_direction(metric_name, metric)
            for row in rows:
                target = metric_input(row, metric_name, torch)
                row[metric_name] = score_to_float(metric(target))
            del metric
            if torch is not None and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    return directions


def build_summary(rows, dataset_names, metrics):
    summary_rows = []
    for dataset in dataset_names:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for metric_name in metrics:
            scores = [float(row[metric_name]) for row in dataset_rows]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric_name,
                    "mean": mean(scores),
                    "std": population_std(scores),
                    "count": len(scores),
                }
            )
    return summary_rows


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, rows, summary_rows, metrics, directions, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_image_scores.csv", rows, ["dataset", "filename", "path", "width", "height", *metrics])
    write_csv(output_dir / "summary_scores.csv", summary_rows, ["dataset", "metric", "mean", "std", "count"])

    summary_json = {
        "metrics": list(metrics),
        "metric_directions": directions,
        "summary": summary_rows,
        **dict(metadata or {}),
    }
    with (output_dir / "summary_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)
    return summary_json


def evaluate_dataset_dirs(
    dataset_dirs,
    output_dir,
    metrics=None,
    device="cpu",
    expected_counts=None,
    max_samples_per_dataset=None,
    sample_seed=42,
):
    metrics = list(metrics or DEFAULT_OMGSR_METRICS)
    expected_counts = expected_counts or {}
    images_by_dataset = collect_images(dataset_dirs)
    validate_expected_counts(images_by_dataset, expected_counts)
    total_counts = {name: len(images) for name, images in images_by_dataset.items()}
    evaluated_images = sample_images(
        images_by_dataset,
        max_samples_per_dataset=max_samples_per_dataset,
        sample_seed=sample_seed,
    )
    rows = build_rows(evaluated_images)
    directions = evaluate_metrics(rows, metrics, device)
    summary_rows = build_summary(rows, list(images_by_dataset.keys()), metrics)
    return write_outputs(
        output_dir,
        rows,
        summary_rows,
        metrics,
        directions,
        metadata={
            "total_image_counts": total_counts,
            "evaluated_image_counts": {
                name: len(images) for name, images in evaluated_images.items()
            },
            "max_samples_per_dataset": max_samples_per_dataset,
            "sample_seed": int(sample_seed) if max_samples_per_dataset is not None else None,
            "metric_input_policy": {
                name: {"min_short_side": size, "resize": "bicubic_keep_aspect"}
                for name, size in METRIC_MIN_SHORT_SIDE.items()
                if name in {metric.lower() for metric in metrics}
            },
        },
    )
