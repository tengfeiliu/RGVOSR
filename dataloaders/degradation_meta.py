import json
import math
from pathlib import Path

import numpy as np


REASONING_KEYS = (
    "degradation_analysis",
    "texture_edge_analysis",
    "semantic_risk_analysis",
    "sr_strategy",
)

DEGRADATION_KEYS = (
    "blur",
    "noise",
    "jpeg",
    "ringing",
    "texture_loss",
    "text_region_risk",
    "color_shift",
    "hallucination_risk",
)

PHYSICAL_DEGRADATION_KEYS = (
    "blur",
    "noise",
    "jpeg",
    "ringing",
    "texture_loss",
    "color_shift",
)

SUGGESTION_VOCAB = {
    "reduce blur",
    "suppress noise",
    "suppress JPEG artifacts",
    "reduce ringing artifacts",
    "recover fine textures",
    "enhance edge sharpness",
    "preserve global structure",
    "preserve color consistency",
    "improve text readability",
    "avoid hallucinated details",
    "avoid over-sharpening",
    "preserve face identity",
    "preserve repeated patterns",
}

SCORE_WEIGHTS = {
    "blur": 0.16,
    "noise": 0.14,
    "jpeg": 0.12,
    "ringing": 0.12,
    "texture_loss": 0.18,
    "text_region_risk": 0.12,
    "color_shift": 0.06,
    "hallucination_risk": 0.10,
}


def clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def mean_numeric(value, default=0.0):
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return float(np.mean(value))
    if isinstance(value, (list, tuple)):
        vals = [mean_numeric(v, default=None) for v in value]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_jsonable(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def normalize_range(value, low, high, invert=False):
    if high == low:
        return 0.0
    val = (mean_numeric(value) - low) / (high - low)
    if invert:
        val = 1.0 - val
    return clamp01(val)


def jpeg_strength(quality, min_quality=30.0, max_quality=95.0):
    return normalize_range(quality, min_quality, max_quality, invert=True)


def noise_strength(noise_meta):
    if not noise_meta:
        return 0.0
    kind = noise_meta.get("type")
    if kind == "gaussian":
        return normalize_range(noise_meta.get("sigma"), 1.0, 30.0)
    if kind == "poisson":
        return normalize_range(noise_meta.get("scale"), 0.05, 3.0)
    return 0.0


def resize_down_strength(resize_meta):
    if not resize_meta:
        return 0.0
    scale = mean_numeric(resize_meta.get("scale"), default=1.0)
    if scale >= 1.0:
        return 0.0
    return clamp01((1.0 - scale) / 0.85)


def kernel_blur_strength(kernel):
    if kernel is None:
        return 0.0
    if hasattr(kernel, "detach"):
        arr = kernel.detach().float().cpu().numpy()
    else:
        arr = np.asarray(kernel, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    weights = np.abs(arr.astype(np.float64))
    total = float(weights.sum())
    if total <= 1e-12:
        return 0.0
    h, w = weights.shape
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    second_moment = float((weights * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / total)
    return clamp01(math.sqrt(max(second_moment, 0.0)) / 8.0)


def _stage_strengths(meta):
    stages = []
    for key in ("stage1", "stage2"):
        stage = meta.get(key)
        if isinstance(stage, dict):
            stages.append(stage)
    return stages


def build_degradation_vector_from_meta(meta):
    stages = _stage_strengths(meta)

    blur_vals = []
    noise_vals = []
    jpeg_vals = []
    resize_vals = []
    for stage in stages:
        kernel = stage.get("kernel")
        if isinstance(kernel, dict) and stage.get("blur_applied", True):
            blur_vals.append(clamp01(kernel.get("strength", 0.0)))
        noise_vals.append(noise_strength(stage.get("noise")))
        jpeg_vals.append(jpeg_strength(stage.get("jpeg_quality")))
        resize_vals.append(resize_down_strength(stage.get("resize")))

    final_sinc = meta.get("final_sinc", {})
    final_sinc_strength = clamp01(final_sinc.get("strength", 0.0)) if final_sinc.get("applied") else 0.0
    blur = clamp01(max(blur_vals or [0.0]) * 0.80 + final_sinc_strength * 0.20)
    noise = clamp01(max(noise_vals or [0.0]))
    jpeg = clamp01(max(jpeg_vals or [0.0]))

    scale_final = mean_numeric(meta.get("scale_final"), default=1.0)
    fixed_downsample = clamp01((scale_final - 1.0) / 7.0) if scale_final > 1 else 0.0
    downsample = clamp01(max(resize_vals or [0.0]) * 0.55 + fixed_downsample * 0.45)

    ringing = clamp01(final_sinc_strength * 0.45 + jpeg * 0.35 + downsample * 0.20)
    texture_loss = clamp01(blur * 0.35 + downsample * 0.25 + jpeg * 0.20 + noise * 0.20)

    color_shift = 0.0
    if meta.get("gray", {}).get("applied"):
        color_shift = max(color_shift, 0.7)
    if meta.get("color_jitter", {}).get("applied"):
        color_shift = max(color_shift, 0.5)

    return {
        "blur": blur,
        "noise": noise,
        "jpeg": jpeg,
        "ringing": ringing,
        "texture_loss": texture_loss,
        "text_region_risk": 0.0,
        "color_shift": clamp01(color_shift),
        "hallucination_risk": 0.0,
    }


def normalize_vector(vector):
    vector = vector or {}
    return {key: round(clamp01(vector.get(key, 0.0)), 6) for key in DEGRADATION_KEYS}


def compute_score(vector):
    vector = normalize_vector(vector)
    risk = 0.0
    weight_sum = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        risk += vector[key] * weight
        weight_sum += weight
    normalized_risk = risk / max(weight_sum, 1e-8)
    return int(round(100.0 * (1.0 - clamp01(normalized_risk))))


def filter_suggestions(suggestions):
    if not isinstance(suggestions, (list, tuple)):
        return []
    filtered = []
    for item in suggestions:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item in SUGGESTION_VOCAB and item not in filtered:
            filtered.append(item)
    return filtered


def physical_suggestions_from_vector(vector):
    vector = normalize_vector(vector)
    suggestions = []
    thresholds = [
        ("blur", "reduce blur", 0.35),
        ("noise", "suppress noise", 0.30),
        ("jpeg", "suppress JPEG artifacts", 0.30),
        ("ringing", "reduce ringing artifacts", 0.30),
        ("texture_loss", "recover fine textures", 0.35),
        ("color_shift", "preserve color consistency", 0.30),
    ]
    for key, suggestion, threshold in thresholds:
        if vector[key] >= threshold:
            suggestions.append(suggestion)
    if vector["blur"] >= 0.25 and "enhance edge sharpness" not in suggestions:
        suggestions.append("enhance edge sharpness")
    return suggestions[:5]


def default_reasoning():
    return {
        "degradation_analysis": "Physical degradation values are derived from the synthetic degradation pipeline.",
        "texture_edge_analysis": "Texture and edge risk are estimated from blur, downsampling, noise, JPEG, and ringing parameters.",
        "semantic_risk_analysis": "Semantic risk was not provided by a vision-language analyzer.",
        "sr_strategy": "Use the physical degradation vector for restoration-aware conditioning or filtering.",
    }


def default_semantic_result(physical_vector=None):
    physical_vector = normalize_vector(physical_vector)
    return {
        "reasoning": default_reasoning(),
        "suggestions": physical_suggestions_from_vector(physical_vector) or ["preserve global structure"],
        "degradation_vector": {
            "text_region_risk": 0.0,
            "hallucination_risk": clamp01(physical_vector["texture_loss"] * 0.4),
        },
    }


def merge_analysis_result(physical_vector, semantic_result):
    physical_vector = physical_vector or {}
    vector = {key: clamp01(physical_vector.get(key, 0.0)) for key in PHYSICAL_DEGRADATION_KEYS}

    semantic_result = semantic_result or default_semantic_result(vector)
    semantic_vector = semantic_result.get("degradation_vector", {})
    vector["text_region_risk"] = clamp01(semantic_vector.get("text_region_risk", 0.0))
    vector["hallucination_risk"] = max(
        clamp01(semantic_vector.get("hallucination_risk", 0.0)),
        clamp01(vector["texture_loss"] * 0.4),
    )

    ordered_vector = normalize_vector(vector)
    reasoning = semantic_result.get("reasoning") or default_reasoning()
    reasoning = {key: str(reasoning.get(key, "")) for key in REASONING_KEYS}
    suggestions = filter_suggestions(semantic_result.get("suggestions"))
    if not suggestions:
        suggestions = physical_suggestions_from_vector(ordered_vector) or ["preserve global structure"]

    return {
        "reasoning": reasoning,
        "suggestions": suggestions,
        "score": compute_score(ordered_vector),
        "degradation_vector": ordered_vector,
    }


def make_cache_record(hq_path, lq_path, raw_degradation_params, result):
    return {
        "hq_path": str(hq_path),
        "lq_path": str(lq_path),
        "raw_degradation_params": to_jsonable(raw_degradation_params),
        "result": result,
    }


def read_jsonl_paths(path, key="hq_path"):
    seen = set()
    path = Path(path)
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = payload.get(key) or payload.get("image_path")
            if value:
                seen.add(str(value))
    return seen
