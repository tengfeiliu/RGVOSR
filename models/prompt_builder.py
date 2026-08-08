import json
import re

from models.router_condition import (
    ROUTER_CONDITION_KEYS,
    ROUTER_CONDITION_VERSION,
    extract_router_condition,
)


DEFAULT_SR_PROMPT = (
    "Super-resolve this low-quality image into a high-quality realistic image.\n\n"
    "Requirements:\n"
    "Preserve the original layout, structure, identity, repeated patterns, and color consistency.\n"
    "Avoid hallucinated details, over-sharpening, and semantic changes."
)

PROMPT_VARIANTS = (
    "fixed",
    "suggestion",
    "iqa",
    "iqa_suggestion",
    "condition8_text",
)
IQA_FIELDS = (
    "distortion_location",
    "distortion_severity",
    "distortion_type",
    "overall_quality",
)
PROFILE_WORD_LIMITS = {
    "caption": 60,
    "distortion_location": 60,
    "distortion_severity": 60,
    "distortion_type": 30,
    "overall_quality": 80,
    "suggestion": 100,
}

_CONDITION8_LEVELS = (
    (0.0, "no visible"),
    (0.1, "subtle"),
    (0.25, "mild"),
    (0.5, "moderate"),
    (0.75, "severe"),
    (1.0, "extreme"),
)
_CONDITION8_PHRASES = {
    "blur": "blur",
    "noise": "noise",
    "compression": "compression artifacts",
    "ringing_aliasing": "ringing and aliasing artifacts",
    "texture_loss": "texture loss and detail loss",
    "photometric": "color distortion and exposure inconsistency",
    "structure_risk": "risk to text, faces, thin lines, repeated patterns, and geometry",
    "hallucination_risk": "risk of hallucinated or false details",
}
CONDITION8_NEUTRAL_TEXT = (
    "No specific degradation condition was recognized; apply balanced restoration "
    "while preserving structure, color, identity, and natural detail."
)


def _safe_text(value, default=""):
    if value is None:
        return default
    try:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False).strip()
        return str(value).strip()
    except Exception:
        return default


def normalize_prompt_variant(prompt_variant):
    if prompt_variant is None:
        return None
    normalized = str(prompt_variant).strip().lower().replace("-", "_")
    if normalized not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unsupported prompt_variant '{prompt_variant}'. Expected one of: {', '.join(PROMPT_VARIANTS)}"
        )
    return normalized


def normalize_bounded_text(value, max_words):
    """Normalize model-authored text and apply a deterministic word budget."""
    text = re.sub(r"\s+", " ", _safe_text(value)).strip()
    if not text:
        return ""

    # Remove exact repeated sentences while preserving their first occurrence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        normalized = sentence.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            unique_sentences.append(normalized)
    text = " ".join(unique_sentences)

    words = text.split()
    if len(words) > int(max_words):
        text = " ".join(words[: int(max_words)]).rstrip(" ,;:")
    return text


def _condition8_level(value):
    value = max(0.0, min(1.0, float(value)))
    return min(_CONDITION8_LEVELS, key=lambda item: abs(item[0] - value))[1]


def condition8_to_canonical_text(condition) -> str:
    """Convert text8_v1 values into deterministic Text-Encoder input text."""
    if str(condition.version) != ROUTER_CONDITION_VERSION:
        raise ValueError(
            f"Unsupported condition8 version '{condition.version}'; "
            f"expected {ROUTER_CONDITION_VERSION}"
        )
    clauses = []
    for key, value, valid in zip(
        ROUTER_CONDITION_KEYS,
        condition.values,
        condition.valid_mask,
    ):
        if not bool(valid):
            continue
        level = _condition8_level(value)
        clauses.append(f"{level} {_CONDITION8_PHRASES[key]}")
    if not clauses:
        return CONDITION8_NEUTRAL_TEXT
    return "; ".join(clauses) + "."


def build_condition8_text(profile: dict) -> str:
    condition = extract_router_condition(
        profile,
        version=ROUTER_CONDITION_VERSION,
    )
    return condition8_to_canonical_text(condition)


def normalized_prompt_profile(profile):
    profile = profile if isinstance(profile, dict) else {}
    source_iqa = profile.get("iqa")
    source_iqa = source_iqa if isinstance(source_iqa, dict) else {}
    return {
        "caption": normalize_bounded_text(
            profile.get("caption"), PROFILE_WORD_LIMITS["caption"]
        ),
        "iqa": {
            field: normalize_bounded_text(
                source_iqa.get(field), PROFILE_WORD_LIMITS[field]
            )
            for field in IQA_FIELDS
        },
        "suggestion": normalize_bounded_text(
            profile.get("suggestion"), PROFILE_WORD_LIMITS["suggestion"]
        ),
    }


def validate_prompt_profile(profile, prompt_variant, include_caption=False):
    prompt_variant = normalize_prompt_variant(prompt_variant)
    if prompt_variant is None:
        raise ValueError("prompt_variant must be explicit when validating a profile")
    if prompt_variant == "fixed" and include_caption:
        raise ValueError("prompt_variant='fixed' cannot be combined with include_caption=true")

    normalized = normalized_prompt_profile(profile)
    missing = []
    if include_caption and not normalized["caption"]:
        missing.append("caption")
    if prompt_variant in {"iqa", "iqa_suggestion"}:
        missing.extend(
            f"iqa.{field}"
            for field in IQA_FIELDS
            if not normalized["iqa"][field]
        )
    if prompt_variant in {"suggestion", "iqa_suggestion"} and not normalized["suggestion"]:
        missing.append("suggestion")
    if missing:
        raise ValueError(
            f"Profile is missing fields required by prompt variant '{prompt_variant}': "
            + ", ".join(missing)
        )
    return normalized


def _append_iqa(parts, iqa):
    parts.extend(["", "IQA profile:"])
    for key in IQA_FIELDS:
        value = iqa.get(key)
        text = _safe_text(value)
        if text:
            parts.extend([f"{key}:", text])


def build_sr_prompt(
    profile: dict,
    use_prompt: bool = True,
    use_suggestions: bool = True,
    prompt_variant=None,
    include_caption: bool = False,
) -> str:
    prompt_variant = normalize_prompt_variant(prompt_variant)
    profile = profile if isinstance(profile, dict) else {}
    iaa = profile.get("iaa")
    iaa = iaa if isinstance(iaa, dict) else {}

    if prompt_variant is not None:
        if prompt_variant == "fixed":
            if include_caption:
                raise ValueError(
                    "prompt_variant='fixed' cannot be combined with include_caption=true"
                )
            return DEFAULT_SR_PROMPT
        normalized = validate_prompt_profile(
            profile,
            prompt_variant=prompt_variant,
            include_caption=include_caption,
        )
        parts = [DEFAULT_SR_PROMPT]
        if include_caption:
            parts.extend(["", "Image description:", normalized["caption"]])
        if prompt_variant == "condition8_text":
            parts.extend(
                [
                    "",
                    "Canonical degradation and fidelity condition:",
                    build_condition8_text(profile),
                ]
            )
        if prompt_variant in {"iqa", "iqa_suggestion"}:
            _append_iqa(parts, normalized["iqa"])
        if prompt_variant in {"suggestion", "iqa_suggestion"}:
            parts.extend(["", "Restoration suggestion:", normalized["suggestion"]])
        return "\n".join(parts)

    if not use_prompt:
        if include_caption:
            raise ValueError("include_caption=true requires use_prompt=true")
        return DEFAULT_SR_PROMPT

    normalized = normalized_prompt_profile(profile)
    if include_caption:
        parts = [DEFAULT_SR_PROMPT]
        if not normalized["caption"]:
            raise ValueError("Profile is missing fields required by prompt: caption")
        parts.extend(["", "Image description:", normalized["caption"]])
        if any(normalized["iqa"].values()):
            _append_iqa(parts, normalized["iqa"])
        if use_suggestions and normalized["suggestion"]:
            parts.extend(["", "Restoration suggestion:", normalized["suggestion"]])
        comprehensive = _safe_text(iaa.get("comprehensive"))
        if comprehensive:
            parts.extend(["", "IAA comprehensive:", comprehensive])
        return "\n".join(parts)

    # Preserve the original no-variant builder for existing full-image configs.
    parts = [
        "Super-resolve this low-quality image into a high-quality realistic image.",
        "",
        "IQA profile:",
    ]
    for key, value in normalized["iqa"].items():
        if value:
            parts.extend([f"{key}:", value])
    if use_suggestions and normalized["suggestion"]:
        parts.extend(["", "Restoration suggestion:", normalized["suggestion"]])
    comprehensive = _safe_text(iaa.get("comprehensive"))
    if comprehensive:
        parts.extend(["", "IAA comprehensive:", comprehensive])
    parts.extend(
        [
            "",
            "Requirements:",
            "Preserve the original layout, structure, identity, repeated patterns, and color consistency.",
            "Avoid hallucinated details, over-sharpening, and semantic changes.",
        ]
    )
    return "\n".join(parts)
