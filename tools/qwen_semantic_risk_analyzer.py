import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.degradation_meta import (
    REASONING_KEYS,
    SUGGESTION_VOCAB,
    clamp01,
    filter_suggestions,
    read_jsonl_paths,
)


DEFAULT_MODEL = "qwen2.5-vl-72b-instruct"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SEMANTIC_PROMPT = """You are an expert in image degradation analysis and super-resolution restoration.

Given a low-resolution image, analyze semantic restoration risks and generate restoration-oriented guidance for a super-resolution model.

Physical degradation values such as blur, noise, JPEG artifacts, ringing, texture loss, and color shift will be supplied by a separate synthetic degradation pipeline. Do not estimate those physical values. Focus on visible semantic risks: text readability, faces or identity, buildings, repeated patterns, thin structures, and hallucination risk.

You must return a valid JSON object only. Do not include markdown, explanations, code fences, or extra text.

The JSON schema must be exactly:

{
  "reasoning": {
    "degradation_analysis": "...",
    "texture_edge_analysis": "...",
    "semantic_risk_analysis": "...",
    "sr_strategy": "..."
  },
  "suggestions": [
    "..."
  ],
  "score": 0,
  "degradation_vector": {
    "text_region_risk": 0.0,
    "hallucination_risk": 0.0
  }
}

Rules:
1. The input image is a low-resolution image for super-resolution.
2. The score field is ignored by downstream code, but must be a number.
3. text_region_risk and hallucination_risk must be numbers in [0, 1].
4. Suggestions must be selected from this controlled vocabulary when applicable:
   - reduce blur
   - suppress noise
   - suppress JPEG artifacts
   - reduce ringing artifacts
   - recover fine textures
   - enhance edge sharpness
   - preserve global structure
   - preserve color consistency
   - improve text readability
   - avoid hallucinated details
   - avoid over-sharpening
   - preserve face identity
   - preserve repeated patterns
5. Return 1 to 5 suggestions, ordered by importance.
6. Do not invent scene content. Focus on visible semantic risk and SR restoration strategy.
7. If text, faces, buildings, repeated patterns, or thin structures are visible, mention the corresponding restoration risk and use a non-zero risk value.
"""

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_json_object(raw_text):
    if raw_text is None:
        raise ValueError("empty response")
    text = str(raw_text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error = None
    for candidate in candidates:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate.strip())
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"invalid JSON response: {last_error}")


def _normalize_reasoning(value):
    if not isinstance(value, dict):
        raise ValueError("reasoning must be an object")
    normalized = {}
    for key in REASONING_KEYS:
        item = value.get(key, "")
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"reasoning.{key} is missing or empty")
        normalized[key] = item.strip()
    return normalized


def _semantic_contradiction(reasoning, suggestions, text_risk, hallucination_risk):
    joined_reasoning = " ".join(reasoning.values()).lower()
    risky_text_terms = ("text", "readability", "letter", "character", "ocr")
    hallucination_terms = (
        "hallucination",
        "face",
        "identity",
        "building",
        "thin structure",
        "repeated pattern",
        "global structure",
    )

    # if "improve text readability" in suggestions and text_risk <= 0.0001:
    #     return "text suggestion with zero text_region_risk"
    # if any(term in joined_reasoning for term in risky_text_terms) and text_risk <= 0.0001:
    #     return "text reasoning with zero text_region_risk"

    hallucination_suggestions = {
        "avoid hallucinated details",
        "preserve face identity",
        "preserve global structure",
        "preserve repeated patterns",
    }
    # if hallucination_suggestions.intersection(suggestions) and hallucination_risk <= 0.03:
    #     return "semantic-risk suggestion with zero hallucination_risk"
    # if any(term in joined_reasoning for term in hallucination_terms) and hallucination_risk <= 0.03:
    #     return "semantic-risk reasoning with zero hallucination_risk"
    return None


def normalize_semantic_result(payload):
    if not isinstance(payload, dict):
        raise ValueError("semantic response must be an object")

    reasoning = _normalize_reasoning(payload.get("reasoning"))
    suggestions = filter_suggestions(payload.get("suggestions"))
    if not suggestions:
        raise ValueError("suggestions are empty after vocabulary filtering")

    vector = payload.get("degradation_vector")
    if not isinstance(vector, dict):
        raise ValueError("degradation_vector must be an object")
    text_risk = clamp01(vector.get("text_region_risk", 0.0))
    hallucination_risk = clamp01(vector.get("hallucination_risk", 0.0))

    contradiction = _semantic_contradiction(reasoning, set(suggestions), text_risk, hallucination_risk)
    if contradiction:
        raise ValueError(contradiction)

    return {
        "reasoning": reasoning,
        "suggestions": suggestions[:5],
        "score": payload.get("score", 0),
        "degradation_vector": {
            "text_region_risk": text_risk,
            "hallucination_risk": hallucination_risk,
        },
    }


def image_to_data_url(image_path):
    image_path = Path(image_path)
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    with image_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def analyze_image(image_path, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, api_key=None, prompt=SEMANTIC_PROMPT):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for Qwen-VL analysis.") from exc

    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
    return normalize_semantic_result(parse_json_object(content)), content


def list_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_path]
    if input_path.is_file():
        images = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    images.append(Path(line))
        return images
    images = []
    for path in input_path.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic SR risks with Qwen2.5-VL.")
    parser.add_argument("--input", required=True, help="Image path, directory, or txt list.")
    parser.add_argument("--output", required=True, help="Valid semantic-analysis JSONL path.")
    parser.add_argument("--invalid-output", required=True, help="Invalid-response JSONL path.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    images = list_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]

    seen = set()
    if args.resume:
        seen.update(read_jsonl_paths(args.output, key="image_path"))
        seen.update(read_jsonl_paths(args.invalid_output, key="image_path"))

    for image_path in images:
        image_key = str(image_path)
        if image_key in seen:
            continue
        try:
            result, raw_response = analyze_image(image_path, model=args.model, base_url=args.base_url)
            append_jsonl(
                args.output,
                {
                    "image_path": image_key,
                    "model": args.model,
                    "result": result,
                    "raw_response": raw_response,
                },
            )
        except Exception as exc:
            append_jsonl(
                args.invalid_output,
                {
                    "image_path": image_key,
                    "reason": str(exc),
                    "raw_response": "",
                },
            )


if __name__ == "__main__":
    main()
