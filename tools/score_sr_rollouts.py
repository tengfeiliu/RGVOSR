"""Stage D: score K output-level candidates with automatic reward_v1."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.sr_reward_v1 import AutoSRRewardV1, RewardCalibration
from rl_sr.artifact_store import read_jsonl, write_jsonl_atomic
from rl_sr.schema import RolloutRecord, StateRecord
from losses.output_nft import group_optimality_probabilities


def _image(path):
    with Image.open(path) as value:
        value.load()
        return to_tensor(value.convert("RGB")).unsqueeze(0)


def _state_index(state_jsonl):
    return {record.state_id: record for record in read_jsonl(state_jsonl, StateRecord)}


def main(args):
    states = _state_index(args.state_jsonl)
    calibration = None
    if args.calibration_json:
        calibration = RewardCalibration.from_dict(json.loads(Path(args.calibration_json).read_text(encoding="utf-8")))
    reward = AutoSRRewardV1(
        calibration=calibration,
        quality_metrics=tuple(args.quality_metrics),
        quality_device=args.device,
        enable_quality=not args.no_quality,
    )
    groups = defaultdict(list)
    for row in read_jsonl(args.rollout_jsonl):
        rollout = RolloutRecord.from_dict(row)
        if rollout.state_id not in states:
            raise KeyError(f"Rollout {rollout.rollout_id} references unknown state {rollout.state_id}")
        groups[rollout.state_id].append(rollout)
    scored = []
    for state_id, rollouts in groups.items():
        state = states[state_id]
        if not state.runtime.original_lr or not state.runtime.current_sr:
            raise ValueError(f"State {state_id} lacks original_lr/current_sr runtime files")
        candidates = []
        for rollout in rollouts:
            path = rollout.runtime.executed_sr
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"Missing candidate for rollout {rollout.rollout_id}: {path}")
            candidates.append(_image(path))
        candidate = torch.cat(candidates, dim=0)
        original = _image(state.runtime.original_lr)
        current = _image(state.runtime.current_sr)
        results = reward.score(original, current.expand_as(candidate), candidate)
        totals = torch.tensor([result.total for result in results])
        probabilities = group_optimality_probabilities(totals)
        for rollout, result, probability in zip(rollouts, results, probabilities):
            probability = 0.5 + result.confidence * (float(probability) - 0.5)
            if result.hard_violation:
                probability = 0.0
            payload = rollout.as_dict()
            payload.update(
                {
                    "reward_version": result.version,
                    "reward_vector": {
                        "quality": result.quality,
                        "faithfulness": result.faithfulness,
                        "local_improvement": result.local_improvement,
                        "violation": result.violation,
                        **result.diagnostics,
                    },
                    "total_reward": result.total,
                    "advantage": result.total - float(totals.mean()),
                    "optimality_probability": probability,
                    "confidence": result.confidence,
                    "hard_violation": result.hard_violation,
                    "rollout_id": "",
                }
            )
            scored.append(RolloutRecord.from_dict(payload))
    write_jsonl_atomic(args.output_jsonl, scored)
    summary = {
        "rollout_count": len(scored),
        "state_count": len(groups),
        "hard_violation_count": sum(record.hard_violation for record in scored),
        "mean_reward": sum(record.total_reward for record in scored) / max(len(scored), 1),
        "mean_confidence": sum(record.confidence for record in scored) / max(len(scored), 1),
        "reward_version": scored[0].reward_version if scored else None,
    }
    Path(args.output_jsonl).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_jsonl", required=True)
    parser.add_argument("--rollout_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--calibration_json", default=None)
    parser.add_argument("--quality_metrics", nargs="+", default=["clipiqa", "musiq", "maniqa"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_quality", action="store_true", help="Diagnostic only; does not replace a calibrated quality run.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
