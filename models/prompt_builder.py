import json


DEFAULT_SR_PROMPT = (
    "Super-resolve this low-quality image into a high-quality realistic image.\n\n"
    "Requirements:\n"
    "Preserve the original layout, structure, identity, repeated patterns, and color consistency.\n"
    "Avoid hallucinated details, over-sharpening, and semantic changes."
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


def build_sr_prompt(profile: dict, use_prompt: bool = True, use_suggestions: bool = True) -> str:
    if not use_prompt:
        return DEFAULT_SR_PROMPT

    profile = profile if isinstance(profile, dict) else {}
    iqa = profile.get("iqa")
    iqa = iqa if isinstance(iqa, dict) else {}
    iaa = profile.get("iaa")
    iaa = iaa if isinstance(iaa, dict) else {}

    parts = [
        "Super-resolve this low-quality image into a high-quality realistic image.",
        "",
        "IQA profile:",
    ]

    for key, value in iqa.items():
        text = _safe_text(value)
        if text:
            parts.extend([f"{key}:", text])

    suggestion = _safe_text(profile.get("suggestion"))
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
