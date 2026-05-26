"""Local structure and contamination validators for IAA/IQA profiles."""

import re
from typing import Any


IAA_FORBIDDEN_TERMS = [
    "blur",
    "blurry",
    "blurriness",
    "low resolution",
    "resolution",
    "pixelation",
    "noise",
    "grain",
    "compression",
    "artifact",
    "artifacts",
    "sharpness",
    "focus",
    "detail loss",
    "texture loss",
    "fidelity",
    "distortion",
    "overexposure",
    "underexposure",
    "exposure defect",
    "image quality",
    "technical quality",
]

IQA_FORBIDDEN_TERMS = [
    "composition",
    "framing",
    "layout",
    "balance",
    "symmetry",
    "rhythm",
    "leading lines",
    "focal point",
    "visual hierarchy",
    "artistic",
    "aesthetic",
    "creativity",
    "originality",
    "theme",
    "storytelling",
    "narrative",
    "mood",
    "atmosphere",
    "emotion",
    "viewer",
    "engagement",
    "memorability",
    "gestalt",
]


def _term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)


def find_forbidden_terms(text: str, forbidden_terms: list[str]) -> list[str]:
    """Return forbidden terms found in text, preserving configured term names."""
    text = str(text or "")
    matches = []
    for term in forbidden_terms:
        if _term_pattern(term).search(text):
            matches.append(term)
    return matches


def contains_forbidden_in_iaa(text: str) -> list[str]:
    """Return IQA/quality terms that are forbidden inside IAA text."""
    return find_forbidden_terms(text, IAA_FORBIDDEN_TERMS)


def contains_forbidden_in_iqa(text: str) -> list[str]:
    """Return aesthetic terms that are forbidden inside IQA text."""
    return find_forbidden_terms(text, IQA_FORBIDDEN_TERMS)


def iter_string_fields(obj: Any, path: str = ""):
    """Yield (path, text) for every string nested inside obj."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            yield from iter_string_fields(value, child)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child = f"{path}[{index}]"
            yield from iter_string_fields(value, child)


def validate_profile_structure(profile: dict) -> list[str]:
    """Return structure warnings for a profile object."""
    errors = []
    if not isinstance(profile, dict):
        return ["profile must be an object"]
    if not isinstance(profile.get("iaa"), dict):
        errors.append("profile.iaa is missing or is not an object")
    if not isinstance(profile.get("iqa"), dict):
        errors.append("profile.iqa is missing or is not an object")
    if "ista" not in profile:
        errors.append("profile.ista is missing")
    return errors


def _violations(section: Any, root_path: str, forbidden_terms: list[str]) -> list[dict]:
    result = []
    for path, text in iter_string_fields(section, root_path):
        terms = find_forbidden_terms(text, forbidden_terms)
        if terms:
            result.append({"path": path, "terms": terms, "text": text})
    return result


def validate_strict_separation(profile: dict) -> dict:
    """Validate that IAA and IQA text do not contain each other's forbidden concepts."""
    profile = profile if isinstance(profile, dict) else {}
    iaa_violations = _violations(profile.get("iaa", {}), "profile.iaa", IAA_FORBIDDEN_TERMS)
    iqa_violations = _violations(profile.get("iqa", {}), "profile.iqa", IQA_FORBIDDEN_TERMS)
    return {
        "valid": not iaa_violations and not iqa_violations,
        "iaa_violations": iaa_violations,
        "iqa_violations": iqa_violations,
    }


def split_bullets(text: str) -> list[str]:
    """Split bullet-style or sentence-style text into clean items."""
    text = str(text or "").strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 or any(re.match(r"^[-*•]\s+", line) for line in lines):
        return [re.sub(r"^[-*•]\s*", "", line).strip() for line in lines if re.sub(r"^[-*•]\s*", "", line).strip()]
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


def remove_duplicate_bullets(text: str) -> str:
    """Remove duplicate bullet/sentence items and return bullet-style text."""
    seen = set()
    kept = []
    for bullet in split_bullets(text):
        key = re.sub(r"\s+", " ", bullet).strip().lower()
        if key and key not in seen:
            seen.add(key)
            kept.append(bullet)
    return "\n".join(f"- {item}" for item in kept)
