"""Summarize the automatic multiround evaluation of G_SFT versus G_RL.

Each row in the output is aligned by (round, dataset, metric).  ``oriented_delta``
is positive when the final NFT policy improves a metric in its declared
direction, so NIQE and other lower-is-better metrics remain directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _read_trends(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric trend CSV does not exist: {path}")
    result = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["round"]), row["dataset"], row["metric"])
            if key in result:
                raise ValueError(f"Duplicate metric trend row in {path}: {key}")
            result[key] = row
    return result


def summarize(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = _read_trends(baseline_path)
    candidate = _read_trends(candidate_path)
    common_keys = sorted(set(baseline) & set(candidate), key=lambda item: (int(item[0]), item[1], item[2]))
    if not common_keys:
        raise RuntimeError("No shared (round, dataset, metric) entries between SFT and RL evaluation")

    records = []
    per_round = defaultdict(list)
    for key in common_keys:
        before, after = baseline[key], candidate[key]
        direction = after["direction"] or before["direction"]
        if direction not in {"higher_better", "lower_better"}:
            raise ValueError(f"Unknown metric direction for {key}: {direction}")
        baseline_mean, candidate_mean = float(before["mean"]), float(after["mean"])
        raw_delta = candidate_mean - baseline_mean
        oriented_delta = raw_delta if direction == "higher_better" else -raw_delta
        record = {
            "round": int(key[0]),
            "dataset": key[1],
            "metric": key[2],
            "direction": direction,
            "g_sft_mean": baseline_mean,
            "g_rl_mean": candidate_mean,
            "raw_delta": raw_delta,
            "oriented_delta": oriented_delta,
        }
        records.append(record)
        per_round[record["round"]].append(oriented_delta)
    return {
        "baseline": "G_SFT",
        "candidate": "G_RL",
        "record_count": len(records),
        "mean_oriented_delta_by_round": {
            str(round_index): sum(values) / len(values) for round_index, values in sorted(per_round.items())
        },
        "records": records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft_metric_trends", required=True)
    parser.add_argument("--rl_metric_trends", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = summarize(Path(args.sft_metric_trends), Path(args.rl_metric_trends))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
