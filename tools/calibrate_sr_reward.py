"""Fit portable NR-IQA normalization statistics for automatic reward_v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.sr_reward_v1 import PyIQAMetricEnsemble, RewardCalibration
from rl_sr.artifact_store import read_jsonl, write_json_atomic


def _image(path):
    with Image.open(path) as value:
        value.load()
        return to_tensor(value.convert("RGB")).unsqueeze(0)


def main(args):
    rows = read_jsonl(args.rollout_jsonl)
    evaluator = PyIQAMetricEnsemble(tuple(args.metrics), device=args.device)
    scores = {name: [] for name in args.metrics}
    valid = 0
    for row in rows:
        runtime = row.get("runtime") or {}
        path = runtime.get("executed_sr")
        if not path or not Path(path).is_file():
            continue
        values = evaluator(_image(path))
        for name, value in values.items():
            scores[name].append(float(value[0]))
        valid += 1
    calibration = RewardCalibration.fit(scores, evaluator.directions)
    if not calibration.metric_calibrations:
        raise RuntimeError(
            f"Could not calibrate from {valid} outputs; at least 8 valid outputs are needed per metric."
        )
    payload = calibration.as_dict() | {
        "source_rollout_count": valid,
        "metrics": list(args.metrics),
        "metric_directions": evaluator.directions,
        "note": "Only metric values are persisted; paths and image contents are not part of calibration identity.",
    }
    write_json_atomic(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--metrics", nargs="+", default=["clipiqa", "musiq", "maniqa"])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
