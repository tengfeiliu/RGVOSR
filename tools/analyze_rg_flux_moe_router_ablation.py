import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compare_rg_flux_pairing_metrics import (  # noqa: E402
    compare_metric_rows,
    load_csv_rows,
    load_metric_directions,
)


def _percentile(sorted_values, quantile):
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean_ci(values, samples=2000, seed=42):
    values = [float(value) for value in values]
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(int(seed))
    means = []
    for _ in range(int(samples)):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _indexed_metric_rows(rows, source):
    indexed = {}
    for row in rows:
        key = (str(row.get("dataset") or ""), str(row.get("filename") or ""))
        if not all(key) or key in indexed:
            raise ValueError(f"Invalid or duplicate metric identity in {source}: {key}")
        indexed[key] = row
    return indexed


def compare_mode_to_baseline(
    baseline_metrics_dir,
    candidate_metrics_dir,
    baseline_mode,
    candidate_mode,
    bootstrap_samples=2000,
    seed=42,
):
    baseline_metrics_dir = Path(baseline_metrics_dir)
    candidate_metrics_dir = Path(candidate_metrics_dir)
    baseline_rows = load_csv_rows(baseline_metrics_dir / "per_image_scores.csv")
    candidate_rows = load_csv_rows(candidate_metrics_dir / "per_image_scores.csv")
    directions = load_metric_directions(baseline_metrics_dir / "summary_scores.json")
    candidate_directions = load_metric_directions(candidate_metrics_dir / "summary_scores.json")
    if directions != candidate_directions:
        raise ValueError(f"Metric directions differ for {candidate_mode}.")

    baseline_index = _indexed_metric_rows(baseline_rows, baseline_mode)
    candidate_index = _indexed_metric_rows(candidate_rows, candidate_mode)
    if set(baseline_index) != set(candidate_index):
        raise ValueError(
            f"Image sets differ between {baseline_mode} and {candidate_mode}."
        )

    base_rows = compare_metric_rows(baseline_rows, candidate_rows, directions)
    output = []
    for summary in base_rows:
        dataset = summary["dataset"]
        metric = summary["metric"]
        direction = summary["direction"]
        advantages = []
        for key in sorted(key for key in baseline_index if key[0] == dataset):
            baseline_value = float(baseline_index[key][metric])
            candidate_value = float(candidate_index[key][metric])
            raw = baseline_value - candidate_value
            advantages.append(raw if direction == "higher_better" else -raw)
        ci_low, ci_high = bootstrap_mean_ci(
            advantages,
            samples=bootstrap_samples,
            seed=int(seed) + sum(ord(char) for char in f"{candidate_mode}:{dataset}:{metric}"),
        )
        sorted_advantages = sorted(advantages)
        output.append(
            {
                "baseline_mode": baseline_mode,
                "candidate_mode": candidate_mode,
                "dataset": dataset,
                "metric": metric,
                "direction": direction,
                "count": summary["count"],
                "baseline_mean": summary["matched_mean"],
                "candidate_mean": summary["shuffled_mean"],
                "baseline_advantage_mean": summary["matched_advantage"],
                "baseline_advantage_median": _percentile(sorted_advantages, 0.5),
                "baseline_win_rate": summary["matched_win_rate"],
                "tie_rate": summary["tie_rate"],
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
            }
        )
    return output


def _image_index(mode_dir):
    manifest_path = Path(mode_dir) / "inference_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    indexed = {}
    for dataset in manifest["datasets"]:
        dataset_name = dataset["name"]
        output_dir = Path(dataset["output_dir"])
        for path in sorted(output_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                indexed[(dataset_name, path.name)] = path
    return indexed


def onehot_output_diversity(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    modes = sorted(path for path in checkpoint_root.glob("onehot_e*") if path.is_dir())
    indexes = {path.name: _image_index(path) for path in modes}
    rows = []
    for left_index, left_mode in enumerate(sorted(indexes)):
        for right_mode in sorted(indexes)[left_index + 1 :]:
            left = indexes[left_mode]
            right = indexes[right_mode]
            if set(left) != set(right):
                raise ValueError(f"One-hot image sets differ: {left_mode} vs {right_mode}")
            by_dataset = {}
            for key in sorted(left):
                left_image = np.asarray(Image.open(left[key]).convert("RGB"), dtype=np.float32) / 255.0
                right_image = np.asarray(Image.open(right[key]).convert("RGB"), dtype=np.float32) / 255.0
                if left_image.shape != right_image.shape:
                    raise ValueError(f"One-hot output shapes differ for {key}.")
                mae = float(np.mean(np.abs(left_image - right_image)))
                rmse = float(np.sqrt(np.mean((left_image - right_image) ** 2)))
                psnr = float("inf") if rmse == 0.0 else -20.0 * math.log10(rmse)
                by_dataset.setdefault(key[0], []).append((mae, psnr))
            for dataset, values in sorted(by_dataset.items()):
                rows.append(
                    {
                        "left_mode": left_mode,
                        "right_mode": right_mode,
                        "dataset": dataset,
                        "count": len(values),
                        "mean_pixel_mae": sum(value[0] for value in values) / len(values),
                        "mean_pairwise_psnr": sum(value[1] for value in values) / len(values),
                    }
                )
    return rows


def analyze_checkpoint(checkpoint_root, baseline_mode="learned_top2", bootstrap_samples=2000, seed=42):
    checkpoint_root = Path(checkpoint_root)
    baseline_metrics = checkpoint_root / baseline_mode / "metrics"
    if not baseline_metrics.exists():
        raise FileNotFoundError(f"Baseline metrics do not exist: {baseline_metrics}")
    comparison_rows = []
    for mode_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        if mode_dir.name == baseline_mode or not (mode_dir / "metrics" / "per_image_scores.csv").exists():
            continue
        comparison_rows.extend(
            compare_mode_to_baseline(
                baseline_metrics,
                mode_dir / "metrics",
                baseline_mode,
                mode_dir.name,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )

    comparison_csv = checkpoint_root / "router_ablation_comparison.csv"
    comparison_json = checkpoint_root / "router_ablation_comparison.json"
    fields = list(comparison_rows[0]) if comparison_rows else []
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparison_rows)
    with comparison_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline_mode": baseline_mode,
                "interpretation": "baseline_advantage > 0 favors learned_top2 after metric direction normalization",
                "comparisons": comparison_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    diversity_rows = onehot_output_diversity(checkpoint_root)
    diversity_csv = checkpoint_root / "onehot_output_diversity.csv"
    with diversity_csv.open("w", newline="", encoding="utf-8") as handle:
        if diversity_rows:
            writer = csv.DictWriter(handle, fieldnames=list(diversity_rows[0]))
            writer.writeheader()
            writer.writerows(diversity_rows)
    return comparison_csv, comparison_json, diversity_csv


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Compare MoE Router ablation metrics image by image.")
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--baseline_mode", default="learned_top2")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    outputs = analyze_checkpoint(
        args.checkpoint_root,
        baseline_mode=args.baseline_mode,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    for output in outputs:
        print(f"[router_ablation] saved analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
