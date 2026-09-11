"""Portable helpers to fit and persist automatic NR-IQA calibration."""

from __future__ import annotations

from .sr_reward_v1 import RewardCalibration


def fit_calibration_from_rows(rows):
    scores = {}
    directions = {}
    for row in rows:
        metric_scores = row.get("metric_scores") or {}
        metric_directions = row.get("metric_directions") or {}
        for name, value in metric_scores.items():
            scores.setdefault(str(name), []).append(float(value))
            if name in metric_directions:
                directions[str(name)] = metric_directions[name]
    return RewardCalibration.fit(scores, directions)
