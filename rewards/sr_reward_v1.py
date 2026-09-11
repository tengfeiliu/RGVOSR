"""Automatic reward_v1 for multi-round output-level SR.

The first reward version is deliberately label-free.  It measures whether a
candidate improves the current SR result while preserving evidence in the
original LR input.  It is *not* a learned preference model and therefore does
not claim human calibration.  ``RewardCalibration`` makes heterogeneous
NR-IQA scores comparable using a deterministic calibration split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F


REWARD_VERSION = "sr_reward_v1_auto_1"


def _as_bchw(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError("Reward images must be torch tensors")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] not in {1, 3}:
        raise ValueError(f"Expected BCHW image with one or three channels, got {tuple(image.shape)}")
    return image.float().clamp(0.0, 1.0)


def _gray(image: torch.Tensor) -> torch.Tensor:
    image = _as_bchw(image)
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _sobel(image: torch.Tensor) -> torch.Tensor:
    gray = _gray(image)
    kernel_x = gray.new_tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    return torch.sqrt(
        F.conv2d(gray, kernel_x, padding=1).square()
        + F.conv2d(gray, kernel_y, padding=1).square()
        + 1.0e-12
    )


def _blur(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    image = _as_bchw(image)
    return F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _per_item_mean(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).mean(dim=1)


def _resize_like(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(_as_bchw(image), size=reference.shape[-2:], mode="bicubic", align_corners=False)


@dataclass(frozen=True)
class MetricCalibration:
    lower_better: bool
    p05: float
    p95: float

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        span = max(float(self.p95) - float(self.p05), 1.0e-6)
        result = (value - float(self.p05)) / span
        if self.lower_better:
            result = 1.0 - result
        return result.clamp(0.0, 1.0)


@dataclass(frozen=True)
class RewardCalibration:
    """Portable calibration statistics; only metric values, never file paths."""

    metric_calibrations: dict[str, MetricCalibration] = field(default_factory=dict)
    version: str = REWARD_VERSION

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "metric_calibrations": {
                name: asdict(value) for name, value in sorted(self.metric_calibrations.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "RewardCalibration":
        metrics = {
            str(name): MetricCalibration(**values)
            for name, values in (payload.get("metric_calibrations") or {}).items()
        }
        return cls(metric_calibrations=metrics, version=str(payload.get("version") or REWARD_VERSION))

    @classmethod
    def fit(
        cls,
        metric_scores: Mapping[str, Iterable[float]],
        directions: Mapping[str, str | bool],
        low_quantile: float = 0.05,
        high_quantile: float = 0.95,
    ) -> "RewardCalibration":
        if not 0.0 <= low_quantile < high_quantile <= 1.0:
            raise ValueError("Calibration quantiles must satisfy 0 <= low < high <= 1")
        calibrations = {}
        for name, values in metric_scores.items():
            values = torch.as_tensor(list(values), dtype=torch.float32)
            values = values[torch.isfinite(values)]
            if values.numel() < 8:
                continue
            direction = directions.get(name, "higher_better")
            lower_better = bool(direction is True or str(direction).lower() == "lower_better")
            calibrations[str(name)] = MetricCalibration(
                lower_better=lower_better,
                p05=float(torch.quantile(values, low_quantile)),
                p95=float(torch.quantile(values, high_quantile)),
            )
        return cls(metric_calibrations=calibrations)


class PyIQAMetricEnsemble:
    """Lazy PyIQA wrapper; absence of PyIQA is reported, never silently faked."""

    def __init__(self, metrics=("clipiqa", "musiq", "maniqa"), device="cuda"):
        self.metric_names = tuple(str(metric) for metric in metrics)
        self.device = str(device)
        self._metrics = None
        self._directions = None

    def _load(self):
        if self._metrics is not None:
            return
        try:
            import pyiqa
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyIQA is required for NR-IQA quality terms. Install the project evaluation dependency, "
                "or set reward.quality_metrics=[] for a diagnostic-only dry run."
            ) from exc
        from metrics.rg_sr_metrics import metric_direction

        self._metrics = {}
        self._directions = {}
        for name in self.metric_names:
            metric = pyiqa.create_metric(name, device=self.device)
            metric.eval()
            self._metrics[name] = metric
            self._directions[name] = metric_direction(name, metric)

    @property
    def directions(self):
        self._load()
        return dict(self._directions)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self._load()
        image = _as_bchw(image).to(self.device)
        output = {}
        for name, metric in self._metrics.items():
            value = metric(image)
            output[name] = value.detach().float().reshape(image.shape[0], -1).mean(dim=1).cpu()
        return output


@dataclass(frozen=True)
class RewardResult:
    version: str
    total: float
    quality: float
    faithfulness: float
    local_improvement: float
    violation: float
    confidence: float
    hard_violation: bool
    diagnostics: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


class AutoSRRewardV1:
    """Reward ensemble used before any human or VLM preference labels exist.

    ``candidate`` and ``current_sr`` are evaluated as an output transition.  A
    reward has no knowledge of filenames: score inputs are images only.  This
    keeps reward calculations portable and prevents accidental path leakage.
    """

    def __init__(
        self,
        calibration: RewardCalibration | None = None,
        quality_metrics=("clipiqa", "musiq", "maniqa"),
        quality_device="cuda",
        weights: Mapping[str, float] | None = None,
        enable_quality: bool = True,
    ):
        self.calibration = calibration or RewardCalibration()
        self.quality_metrics = tuple(quality_metrics)
        self.quality_device = quality_device
        self.enable_quality = bool(enable_quality and self.quality_metrics)
        self.quality_evaluator = None
        self.weights = {
            "quality": 1.00,
            "faithfulness": 1.20,
            "local_improvement": 0.65,
            "violation": 1.35,
            **dict(weights or {}),
        }

    def _quality_delta(self, candidate: torch.Tensor, current_sr: torch.Tensor):
        if not self.enable_quality:
            return candidate.new_zeros(candidate.shape[0]), {}, 0.0
        if self.quality_evaluator is None:
            self.quality_evaluator = PyIQAMetricEnsemble(self.quality_metrics, self.quality_device)
        candidate_scores = self.quality_evaluator(candidate)
        current_scores = self.quality_evaluator(current_sr)
        metric_deltas = {}
        normalized = []
        for name in self.quality_metrics:
            if name not in candidate_scores or name not in current_scores:
                continue
            calibration = self.calibration.metric_calibrations.get(name)
            if calibration is None:
                # A bounded difference is safer than treating raw metric scales as comparable.
                direction = self.quality_evaluator.directions[name]
                sign = -1.0 if direction == "lower_better" else 1.0
                delta = torch.tanh(sign * (candidate_scores[name] - current_scores[name]))
            else:
                delta = calibration.normalize(candidate_scores[name]) - calibration.normalize(current_scores[name])
            delta = delta.to(candidate.device)
            delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
            normalized.append(delta)
            metric_deltas[f"quality_delta_{name}"] = float(delta.mean())
        if not normalized:
            return candidate.new_zeros(candidate.shape[0]), metric_deltas, 0.0
        stacked = torch.stack([value.to(candidate.device) for value in normalized], dim=1)
        score = stacked.median(dim=1).values
        agreement = 1.0 - float(stacked.std(dim=1, unbiased=False).mean().clamp(0.0, 1.0))
        return score, metric_deltas, agreement

    def _faithfulness_delta(self, original_lr, current_sr, candidate):
        original_lr = _as_bchw(original_lr).to(candidate.device)
        candidate_lr = _resize_like(candidate, original_lr)
        current_lr = _resize_like(current_sr, original_lr)
        def discrepancy(value):
            low = _per_item_mean((_blur(value) - _blur(original_lr)).abs())
            edge = _per_item_mean((_sobel(value) - _sobel(original_lr)).abs())
            chroma = _per_item_mean(
                (F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
                 - F.avg_pool2d(original_lr, kernel_size=3, stride=1, padding=1)).abs()
            )
            return 0.55 * low + 0.30 * edge + 0.15 * chroma
        current_error = discrepancy(current_lr)
        candidate_error = discrepancy(candidate_lr)
        return (current_error - candidate_error).clamp(-1.0, 1.0), candidate_error

    def _local_transition(self, original_lr, current_sr, candidate):
        current_sr = _as_bchw(current_sr)
        candidate = _as_bchw(candidate).to(current_sr.device)
        anchor_up = _resize_like(original_lr, current_sr)
        parent_residual = (_blur(current_sr) - _blur(anchor_up)).abs().mean(dim=1, keepdim=True)
        parent_edges = _sobel(current_sr)
        # Regions where parent contains strong unexplained residual/detail are
        # allowed to change; stable regions receive a collateral-edit penalty.
        risk = parent_residual + 0.18 * parent_edges
        threshold = risk.flatten(1).quantile(0.70, dim=1).view(-1, 1, 1, 1)
        risk_mask = (risk >= threshold).float()
        changed = (candidate - current_sr).abs().mean(dim=1, keepdim=True)
        sharp_gain = (_sobel(candidate) - _sobel(current_sr)).clamp(min=0.0)
        risk_gain = (sharp_gain * risk_mask).flatten(1).mean(dim=1)
        collateral = (changed * (1.0 - risk_mask)).flatten(1).mean(dim=1)
        return (risk_gain - 1.6 * collateral).clamp(-1.0, 1.0), risk_mask.mean(dim=(1, 2, 3)), collateral

    def _violations(self, original_lr, current_sr, candidate):
        candidate = _as_bchw(candidate)
        current_sr = _as_bchw(current_sr).to(candidate.device)
        anchor_up = _resize_like(original_lr, candidate)
        low_source_error = _per_item_mean((_blur(candidate) - _blur(anchor_up)).abs())
        chroma_shift = _per_item_mean(
            (candidate - current_sr).mean(dim=(-2, -1), keepdim=True).abs()
        )
        laplace = candidate.new_tensor(((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))).view(1, 1, 3, 3)
        ringing = _per_item_mean(F.conv2d(_gray(candidate), laplace, padding=1).abs())
        collateral = _per_item_mean((candidate - current_sr).abs())
        # Thresholds are explicitly inspectable and should be rechecked on the
        # target data; they are not hidden learned policy decisions.
        violation = (
            1.8 * (low_source_error - 0.11).clamp_min(0.0)
            + 2.0 * (chroma_shift - 0.045).clamp_min(0.0)
            + 0.55 * (ringing - 0.25).clamp_min(0.0)
            + 0.75 * (collateral - 0.12).clamp_min(0.0)
        ).clamp(0.0, 1.0)
        hard = (low_source_error > 0.22) | (chroma_shift > 0.12) | (collateral > 0.30)
        return violation, hard, {
            "low_source_error": low_source_error,
            "chroma_shift": chroma_shift,
            "ringing": ringing,
            "collateral_change": collateral,
        }

    @torch.no_grad()
    def score(self, original_lr, current_sr, candidate) -> list[RewardResult]:
        original_lr = _as_bchw(original_lr)
        current_sr = _as_bchw(current_sr)
        candidate = _as_bchw(candidate)
        if current_sr.shape != candidate.shape:
            raise ValueError("current_sr and candidate must have matching B/C/H/W geometry")
        if original_lr.shape[0] not in {1, candidate.shape[0]}:
            raise ValueError("original_lr batch must be one or match candidate batch")
        if original_lr.shape[0] == 1 and candidate.shape[0] > 1:
            original_lr = original_lr.expand(candidate.shape[0], -1, -1, -1)
        candidate = candidate.to(current_sr.device)
        original_lr = original_lr.to(current_sr.device)
        quality, quality_details, agreement = self._quality_delta(candidate, current_sr)
        faithfulness, faith_error = self._faithfulness_delta(original_lr, current_sr, candidate)
        local, risk_coverage, _ = self._local_transition(original_lr, current_sr, candidate)
        violation, hard, violation_details = self._violations(original_lr, current_sr, candidate)
        # Agreement is high only if independent NR-IQA metrics support the same
        # direction.  Deterministic terms keep a usable confidence when PyIQA is
        # deliberately disabled for a dry run.
        validity = torch.isfinite(quality + faithfulness + local + violation).float()
        q_conf = candidate.new_full((candidate.shape[0],), agreement if self.enable_quality else 0.45)
        confidence = (0.40 * validity + 0.35 * q_conf + 0.25 * (1.0 - violation)).clamp(0.0, 1.0)
        total = (
            self.weights["quality"] * quality
            + self.weights["faithfulness"] * faithfulness
            + self.weights["local_improvement"] * local
            - self.weights["violation"] * violation
        )
        total = torch.where(hard, total.new_full(total.shape, -1.0), total)
        results = []
        for index in range(candidate.shape[0]):
            diagnostics = {
                "faithfulness_error": float(faith_error[index]),
                "risk_coverage": float(risk_coverage[index]),
                **{name: float(value[index]) for name, value in violation_details.items()},
                **quality_details,
            }
            results.append(
                RewardResult(
                    version=REWARD_VERSION,
                    total=float(total[index]),
                    quality=float(quality[index]),
                    faithfulness=float(faithfulness[index]),
                    local_improvement=float(local[index]),
                    violation=float(violation[index]),
                    confidence=float(confidence[index]),
                    hard_violation=bool(hard[index]),
                    diagnostics=diagnostics,
                )
            )
        return results
