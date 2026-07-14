import argparse
import csv
import json
import math
from pathlib import Path


IDENTITY_FIELDS = {"dataset", "filename", "path", "width", "height"}


def load_csv_rows(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_metric_directions(summary_path):
    summary_path = Path(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    directions = payload.get("metric_directions")
    if not isinstance(directions, dict) or not directions:
        raise ValueError(f"Missing metric_directions in {summary_path}")
    return directions


def mean(values):
    return sum(values) / len(values)


def population_std(values):
    if len(values) <= 1:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def index_rows(rows, source_name):
    indexed = {}
    for row in rows:
        key = (str(row.get("dataset") or ""), str(row.get("filename") or ""))
        if not all(key):
            raise ValueError(f"{source_name} contains a row without dataset/filename: {row}")
        if key in indexed:
            raise ValueError(f"{source_name} contains duplicate dataset/filename: {key}")
        indexed[key] = row
    return indexed


def compare_metric_rows(matched_rows, shuffled_rows, directions):
    matched = index_rows(matched_rows, "matched metrics")
    shuffled = index_rows(shuffled_rows, "shuffled metrics")
    matched_keys = set(matched)
    shuffled_keys = set(shuffled)
    if matched_keys != shuffled_keys:
        missing_shuffled = sorted(matched_keys - shuffled_keys)[:5]
        missing_matched = sorted(shuffled_keys - matched_keys)[:5]
        raise ValueError(
            "Matched and shuffled metric rows do not contain the same images. "
            f"missing_from_shuffled={missing_shuffled}, missing_from_matched={missing_matched}"
        )

    output_rows = []
    datasets = sorted({dataset for dataset, _ in matched_keys})
    for dataset in datasets:
        dataset_keys = sorted(key for key in matched_keys if key[0] == dataset)
        for metric, direction in directions.items():
            if direction not in {"higher_better", "lower_better"}:
                raise ValueError(f"Unsupported metric direction for {metric}: {direction}")
            matched_values = []
            shuffled_values = []
            raw_deltas = []
            advantages = []
            for key in dataset_keys:
                try:
                    matched_value = float(matched[key][metric])
                    shuffled_value = float(shuffled[key][metric])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid metric '{metric}' for image {key}") from exc
                if not math.isfinite(matched_value) or not math.isfinite(shuffled_value):
                    raise ValueError(f"Non-finite metric '{metric}' for image {key}")
                raw_delta = matched_value - shuffled_value
                advantage = raw_delta if direction == "higher_better" else -raw_delta
                matched_values.append(matched_value)
                shuffled_values.append(shuffled_value)
                raw_deltas.append(raw_delta)
                advantages.append(advantage)

            count = len(dataset_keys)
            wins = sum(value > 0.0 for value in advantages)
            ties = sum(value == 0.0 for value in advantages)
            output_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "direction": direction,
                    "count": count,
                    "matched_mean": mean(matched_values),
                    "shuffled_mean": mean(shuffled_values),
                    "matched_minus_shuffled": mean(raw_deltas),
                    "matched_advantage": mean(advantages),
                    "paired_advantage_std": population_std(advantages),
                    "matched_win_rate": wins / count,
                    "tie_rate": ties / count,
                }
            )
    return output_rows


def write_comparison_outputs(output_dir, rows, matched_metrics_dir, shuffled_metrics_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pairing_comparison.csv"
    json_path = output_dir / "pairing_comparison.json"
    fieldnames = [
        "dataset",
        "metric",
        "direction",
        "count",
        "matched_mean",
        "shuffled_mean",
        "matched_minus_shuffled",
        "matched_advantage",
        "paired_advantage_std",
        "matched_win_rate",
        "tie_rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "matched_metrics_dir": str(matched_metrics_dir),
        "shuffled_metrics_dir": str(shuffled_metrics_dir),
        "interpretation": "matched_advantage > 0 means matched suggestions outperform shuffled suggestions",
        "comparisons": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return csv_path, json_path


def compare_pairing_metrics(matched_metrics_dir, shuffled_metrics_dir, output_dir):
    matched_metrics_dir = Path(matched_metrics_dir)
    shuffled_metrics_dir = Path(shuffled_metrics_dir)
    matched_rows = load_csv_rows(matched_metrics_dir / "per_image_scores.csv")
    shuffled_rows = load_csv_rows(shuffled_metrics_dir / "per_image_scores.csv")
    matched_directions = load_metric_directions(matched_metrics_dir / "summary_scores.json")
    shuffled_directions = load_metric_directions(shuffled_metrics_dir / "summary_scores.json")
    if matched_directions != shuffled_directions:
        raise ValueError(
            "Matched and shuffled evaluations use different metrics or metric directions: "
            f"matched={matched_directions}, shuffled={shuffled_directions}"
        )
    rows = compare_metric_rows(matched_rows, shuffled_rows, matched_directions)
    return write_comparison_outputs(
        output_dir=output_dir,
        rows=rows,
        matched_metrics_dir=matched_metrics_dir,
        shuffled_metrics_dir=shuffled_metrics_dir,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare matched and cross-image shuffled suggestion metrics image by image."
    )
    parser.add_argument("--matched_metrics_dir", required=True)
    parser.add_argument("--shuffled_metrics_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    csv_path, json_path = compare_pairing_metrics(
        matched_metrics_dir=args.matched_metrics_dir,
        shuffled_metrics_dir=args.shuffled_metrics_dir,
        output_dir=args.output_dir,
    )
    print(f"Saved paired comparison CSV: {csv_path}")
    print(f"Saved paired comparison JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
