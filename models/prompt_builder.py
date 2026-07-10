import json


DEFAULT_SR_PROMPT = (
    "Super-resolve this low-quality image into a high-quality realistic image.\n\n"
    "Requirements:\n"
    "Preserve the original layout, structure, identity, repeated patterns, and color consistency.\n"
    "Avoid hallucinated details, over-sharpening, and semantic changes."
)

PROMPT_VARIANTS = ("fixed", "suggestion", "iqa", "iqa_suggestion")


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


def _append_iqa(parts, iqa):
    parts.extend(["", "IQA profile:"])
    for key, value in iqa.items():
        text = _safe_text(value)
        if text:
            parts.extend([f"{key}:", text])


def build_sr_prompt(
    profile: dict,
    use_prompt: bool = True,
    use_suggestions: bool = True,
    prompt_variant=None,
) -> str:
    prompt_variant = normalize_prompt_variant(prompt_variant)
    profile = profile if isinstance(profile, dict) else {}
    iqa = profile.get("iqa")
    iqa = iqa if isinstance(iqa, dict) else {}
    iaa = profile.get("iaa")
    iaa = iaa if isinstance(iaa, dict) else {}
    suggestion = _safe_text(profile.get("suggestion"))

    if prompt_variant is not None:
        if prompt_variant == "fixed":
            return DEFAULT_SR_PROMPT
        parts = [DEFAULT_SR_PROMPT]
        if prompt_variant in {"iqa", "iqa_suggestion"}:
            _append_iqa(parts, iqa)
        if prompt_variant in {"suggestion", "iqa_suggestion"} and suggestion:
            parts.extend(["", "Restoration suggestion:", suggestion])
        return "\n".join(parts)

    if not use_prompt:
        return DEFAULT_SR_PROMPT

    parts = [
        "Super-resolve this low-quality image into a high-quality realistic image.",
        "",
        "IQA profile:",
    ]

    for key, value in iqa.items():
        text = _safe_text(value)
        if text:
            parts.extend([f"{key}:", text])

    if use_suggestions and suggestion:
        parts.extend(["", "Restoration suggestion:", suggestion])

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
