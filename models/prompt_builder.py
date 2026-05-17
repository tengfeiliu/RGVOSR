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
        return str(value).strip()
    except Exception:
        return default


def build_sr_prompt(result: dict, use_prompt: bool = True, use_suggestions: bool = True) -> str:
    if not use_prompt:
        return DEFAULT_SR_PROMPT

    result = result if isinstance(result, dict) else {}
    reasoning = result.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}

    degradation_analysis = _safe_text(reasoning.get("degradation_analysis"))
    texture_edge_analysis = _safe_text(reasoning.get("texture_edge_analysis"))
    semantic_risk_analysis = _safe_text(reasoning.get("semantic_risk_analysis"))
    sr_strategy = _safe_text(reasoning.get("sr_strategy"))

    parts = [
        "Super-resolve this low-quality image into a high-quality realistic image.",
        "",
        "Image degradation analysis:",
        degradation_analysis,
        "",
        "Texture and edge analysis:",
        texture_edge_analysis,
        "",
        "Semantic restoration risk:",
        semantic_risk_analysis,
        "",
        "Super-resolution strategy:",
        sr_strategy,
    ]

    suggestions = result.get("suggestions")
    if use_suggestions and isinstance(suggestions, (list, tuple)):
        clean_suggestions = [_safe_text(item) for item in suggestions]
        clean_suggestions = [item for item in clean_suggestions if item]
        if clean_suggestions:
            parts.extend(["", "Restoration suggestions:"])
            parts.extend([f"- {item}" for item in clean_suggestions])

    parts.extend(
        [
            "",
            "Requirements:",
            "Preserve the original layout, structure, identity, repeated patterns, and color consistency.",
            "Avoid hallucinated details, over-sharpening, and semantic changes.",
        ]
    )

    return "\n".join(parts)
