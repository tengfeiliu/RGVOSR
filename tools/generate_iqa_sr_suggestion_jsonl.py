"""Generate conservative IQA-only SR suggestions in a new JSONL file.

This tool intentionally does not use or modify the existing profile cleaner. It
copies every input record and replaces only
``unipercept_raw.profile.suggestion`` in the output copy.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from profile_cleaner.json_utils import parse_json_strict_or_extract
from models.prompt_builder import PROFILE_WORD_LIMITS


DEFAULT_MODEL = "qwen2.5-vl-72b-instruct"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
IQA_FIELDS = (
    "distortion_type",
    "distortion_location",
    "distortion_severity",
    "overall_quality",
)
ALLOWED_TYPES = (
    "blur",
    "noise",
    "compression_artifacts",
    "ringing",
    "aliasing_pixelation",
    "edge_texture_loss",
)
ALLOWED_LEVELS = ("mild", "moderate")
SUGGESTION_MAX_WORDS = int(PROFILE_WORD_LIMITS["suggestion"])
TARGET_MAX_WORDS = 8

PRESERVATION_SUFFIX = (
    "Preserve the original exposure, color relationships, geometry, and semantic content. "
    "Do not invent unsupported details."
)
FALLBACK_SUGGESTION = (
    "Apply conservative 4x super-resolution using only source-supported information. "
    + PRESERVATION_SUFFIX
)

ACTION_TEXT = {
    ("blur", "mild"): "Mildly reduce visible blur while preserving stable boundaries.",
    ("blur", "moderate"): "Moderately reduce visible blur while preserving stable boundaries.",
    ("noise", "mild"): "Mildly suppress visible noise while preserving natural texture.",
    ("noise", "moderate"): "Moderately suppress visible noise while preserving natural texture.",
    (
        "compression_artifacts",
        "mild",
    ): "Mildly suppress visible compression artifacts without changing source structure.",
    (
        "compression_artifacts",
        "moderate",
    ): "Moderately suppress visible compression artifacts without changing source structure.",
    ("ringing", "mild"): "Mildly reduce visible ringing and halo artifacts.",
    ("ringing", "moderate"): "Moderately reduce visible ringing and halo artifacts.",
    (
        "aliasing_pixelation",
        "mild",
    ): "Mildly reduce visible aliasing and pixelation on source-supported edges.",
    (
        "aliasing_pixelation",
        "moderate",
    ): "Moderately reduce visible aliasing and pixelation on source-supported edges.",
    (
        "edge_texture_loss",
        "mild",
    ): "Conservatively recover source-supported edge and texture detail.",
    (
        "edge_texture_loss",
        "moderate",
    ): "Conservatively recover source-supported edge and texture detail.",
}
LOCATION_ACTION_TEXT = {
    ("blur", "mild"): "Mildly reduce blur affecting {target}.",
    ("blur", "moderate"): "Moderately reduce blur affecting {target}.",
    ("noise", "mild"): "Mildly suppress noise in {target}.",
    ("noise", "moderate"): "Moderately suppress noise in {target}.",
    (
        "compression_artifacts",
        "mild",
    ): "Mildly suppress compression artifacts in {target}.",
    (
        "compression_artifacts",
        "moderate",
    ): "Moderately suppress compression artifacts in {target}.",
    ("ringing", "mild"): "Mildly reduce ringing around {target}.",
    ("ringing", "moderate"): "Moderately reduce ringing around {target}.",
    (
        "aliasing_pixelation",
        "mild",
    ): "Mildly reduce aliasing and pixelation around {target}.",
    (
        "aliasing_pixelation",
        "moderate",
    ): "Moderately reduce aliasing and pixelation around {target}.",
    (
        "edge_texture_loss",
        "mild",
    ): "Conservatively recover source-supported edge and texture detail in {target}.",
    (
        "edge_texture_loss",
        "moderate",
    ): "Conservatively recover source-supported edge and texture detail in {target}.",
}

CATEGORY_PATTERNS = {
    "blur": re.compile(
        r"\b(?:blur|blurred|blurry|blurriness|defocus|motion\s+blur|soft\s+focus|"
        r"low\s+sharpness|lack\s+of\s+sharpness|limited\s+sharpness|soft\s+edges?)\b",
        re.IGNORECASE,
    ),
    "noise": re.compile(r"\b(?:noise|grain|grainy|noisy)\b", re.IGNORECASE),
    "compression_artifacts": re.compile(
        r"\b(?:compression|jpeg|blockiness|blocking|deblocking|quantization)\b",
        re.IGNORECASE,
    ),
    "ringing": re.compile(r"\b(?:ringing|halo|halos)\b", re.IGNORECASE),
    "aliasing_pixelation": re.compile(
        r"\b(?:aliasing|jaggies|jagged|pixelation|pixelated|low[-\s]+resolution)\b",
        re.IGNORECASE,
    ),
    "edge_texture_loss": re.compile(
        r"\b(?:texture\s+loss|detail\s+loss|loss\s+of\s+(?:fine\s+)?detail|"
        r"lost\s+(?:fine\s+)?detail|edge\s+(?:merging|softening)|"
        r"poor\s+edge\s+clarity|reduced\s+edge\s+acuity|washed[-\s]+out\s+texture)\b",
        re.IGNORECASE,
    ),
}
NEGATIVE_EVIDENCE = re.compile(
    r"\b(?:none|absent|negligible|minimal|not\s+visible|not\s+present|"
    r"no\s+(?:visible\s+)?(?:blur|noise|compression|ringing|aliasing|pixelation))\b",
    re.IGNORECASE,
)
MODERATE_EVIDENCE = re.compile(
    r"\b(?:moderate|moderately|severe|severely|strong|pronounced|pervasive|"
    r"substantial|significant|extensive|heavy|high)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
TARGET_FORBIDDEN_PATTERN = re.compile(
    r"\b(?:mild|mildly|slight|slightly|minor|moderate|moderately|severe|"
    r"severely|strong|strongly|reduce|remove|suppress|recover|restore|"
    r"sharpen|denoise|deblur|improve|enhance|correct|fix|reconstruct|"
    r"retouch|recolor|relight|upscale|super[-\s]?resolution)\b",
    re.IGNORECASE,
)


SYSTEM_PROMPT_PREFIX = """You are a conservative IQA-to-super-resolution instruction compiler.

Convert a UniPercept IQA report into a short, reliable restoration suggestion for fidelity-first 4x super-resolution with FLUX.2 [klein]. The paired HR target preserves the original content, exposure, colors, lighting, geometry, composition, and semantic identity.

Hard rules:
1. Use only distortion_type, distortion_location, distortion_severity, and overall_quality from the supplied IQA report. Ignore IAA and all aesthetic information.
2. Never beautify, retouch, redesign, relight, recolor, or reinterpret the image.
3. Allowed degradation types are only: blur, noise, compression_artifacts, ringing, aliasing_pixelation, edge_texture_loss.
4. Never propose changing exposure, brightness, highlights, shadows, contrast, dynamic range, saturation, white balance, color, lighting, composition, or mood.
5. Select a degradation only when positively supported by distortion_type or distortion_severity. distortion_location and overall_quality may confirm it but cannot introduce it.
6. Evidence described as none, absent, negligible, minimal, not visible, or not present is insufficient and must be omitted.
7. If fields conflict, omit the uncertain degradation rather than guessing.
8. Select at most three unique degradation types, ordered by restoration priority.
9. Use level mild for slight, mild, minor, or limited degradation. Use level moderate for moderate, severe, strong, pervasive, or substantial degradation. Never use a level stronger than moderate.
"""

LOCATION_ENABLED_PROMPT = """
10. For each selected degradation, optionally extract one concise crop-local target from distortion_location.
11. target must be a contiguous quote of at most 8 English words from location_evidence. It should name only the visibly supported subject, region, or relative spatial location, without degradation, severity, or restoration-action words.
12. location_evidence must be one complete positive-evidence clause copied from distortion_location, must support the same degradation type, and must not be paraphrased.
13. Do not infer identities, names, hidden objects, unreadable text, or locations unsupported by distortion_location. Use empty strings when no reliable crop-local target exists.
"""

LOCATION_DISABLED_PROMPT = """
10. Return empty target and location_evidence strings for every selected degradation.
11. Never mention scene content, object names, or spatial locations in the final suggestion. It must remain valid after a random crop.
12. Do not infer or copy crop-local target phrases into any other response field.
13. Keep selected_degradations limited to globally applicable degradation actions.
"""

SYSTEM_PROMPT_SUFFIX = f"""
14. The final suggestion must contain at most three operations and {SUGGESTION_MAX_WORDS} words, and end with: "Preserve the original exposure, color relationships, geometry, and semantic content. Do not invent unsupported details."
15. If no degradation is sufficiently supported, use the conservative fallback given in the schema example.

Return valid JSON only with exactly this schema:
{{
  "selected_degradations": [
    {{
      "type": "blur | noise | compression_artifacts | ringing | aliasing_pixelation | edge_texture_loss",
      "level": "mild | moderate",
      "evidence_fields": ["distortion_type", "distortion_location", "distortion_severity", "overall_quality"],
      "target": "",
      "location_evidence": ""
    }}
  ],
  "rejected_claims": [
    {{"claim": "", "reason": ""}}
  ],
  "suggestion": ""
}}

The suggestion wording will be deterministically validated and compiled downstream. Accuracy of selected_degradations and location evidence is more important than producing many actions."""


def build_system_prompt(include_location: bool = True) -> str:
    location_rules = (
        LOCATION_ENABLED_PROMPT
        if include_location
        else LOCATION_DISABLED_PROMPT
    )
    return SYSTEM_PROMPT_PREFIX + location_rules + SYSTEM_PROMPT_SUFFIX


SYSTEM_PROMPT = build_system_prompt(include_location=True)


def render_user_prompt(
    iqa: dict[str, Any],
    include_location: bool = True,
) -> str:
    payload = {field: iqa.get(field, "") for field in IQA_FIELDS}
    mode = (
        "crop-local, location-aware"
        if include_location
        else "crop-invariant, location-free"
    )
    return (
        f"Convert this UniPercept IQA report into a conservative, {mode} 4x "
        "super-resolution suggestion. Return valid JSON only.\n\nIQA report:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(str(text or "")))


def split_evidence_clauses(text: Any) -> list[str]:
    return [
        item.strip(" -*\t")
        for item in re.split(r"[.;\n]+", str(text or ""))
        if item.strip(" -*\t")
    ]


def positive_clauses(text: Any, degradation_type: str) -> list[str]:
    pattern = CATEGORY_PATTERNS[degradation_type]
    return [
        clause
        for clause in split_evidence_clauses(text)
        if pattern.search(clause) and not NEGATIVE_EVIDENCE.search(clause)
    ]


def evidence_fields_for(iqa: dict[str, Any], degradation_type: str) -> list[str]:
    return [field for field in IQA_FIELDS if positive_clauses(iqa.get(field), degradation_type)]


def normalize_comparison_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.strip(" \t\r\n.,;:!?\"“”").casefold()


def word_tokens(value: Any) -> list[str]:
    return [
        match.group(0).casefold()
        for match in WORD_PATTERN.finditer(str(value or ""))
    ]


def contiguous_token_spans(
    container: Any,
    candidate: Any,
) -> list[tuple[int, int]]:
    container_text = str(container or "")
    container_matches = list(WORD_PATTERN.finditer(container_text))
    container_tokens = [
        match.group(0).casefold()
        for match in container_matches
    ]
    candidate_tokens = word_tokens(candidate)
    if not candidate_tokens or len(candidate_tokens) > len(container_tokens):
        return []
    width = len(candidate_tokens)
    return [
        (
            container_matches[index].start(),
            container_matches[index + width - 1].end(),
        )
        for index in range(len(container_tokens) - width + 1)
        if container_tokens[index : index + width] == candidate_tokens
    ]


def contains_contiguous_tokens(container: Any, candidate: Any) -> bool:
    return bool(contiguous_token_spans(container, candidate))


def target_is_linked_to_degradation(
    clause: str,
    target: str,
    degradation_type: str,
) -> bool:
    target_spans = contiguous_token_spans(clause, target)
    degradation_spans = [
        match.span()
        for match in CATEGORY_PATTERNS[degradation_type].finditer(clause)
    ]
    other_degradation_spans = [
        match.span()
        for other_type, pattern in CATEGORY_PATTERNS.items()
        if other_type != degradation_type
        for match in pattern.finditer(clause)
    ]
    for target_start, target_end in target_spans:
        for degradation_start, degradation_end in degradation_spans:
            between_start = min(target_start, degradation_start)
            between_end = max(target_end, degradation_end)
            if not any(
                other_start >= between_start and other_end <= between_end
                for other_start, other_end in other_degradation_spans
            ):
                return True
    return False


def validate_location_target(
    item: dict[str, Any],
    iqa: dict[str, Any],
    degradation_type: str,
    include_location: bool = True,
) -> dict[str, str]:
    if not include_location:
        return {
            "target": "",
            "location_evidence": "",
            "location_status": "disabled",
            "location_reason": "location output disabled by configuration",
        }

    target = re.sub(r"\s+", " ", str(item.get("target") or "")).strip()
    location_evidence = re.sub(
        r"\s+",
        " ",
        str(item.get("location_evidence") or ""),
    ).strip()
    if not target and not location_evidence:
        return {
            "target": "",
            "location_evidence": "",
            "location_status": "missing",
            "location_reason": "no crop-local location was supplied",
        }
    if not target or not location_evidence:
        return {
            "target": "",
            "location_evidence": "",
            "location_status": "rejected",
            "location_reason": (
                "target and location_evidence must either both be present "
                "or both be empty"
            ),
        }
    if count_words(target) > TARGET_MAX_WORDS:
        return {
            "target": "",
            "location_evidence": "",
            "location_status": "rejected",
            "location_reason": (
                f"target exceeds {TARGET_MAX_WORDS} words"
            ),
        }

    location_clauses = positive_clauses(
        iqa.get("distortion_location"),
        degradation_type,
    )
    normalized_evidence = normalize_comparison_text(location_evidence)
    matching_clause = next(
        (
            clause
            for clause in location_clauses
            if normalize_comparison_text(clause) == normalized_evidence
        ),
        None,
    )
    if matching_clause is None:
        return {
            "target": "",
            "location_evidence": "",
            "location_status": "rejected",
            "location_reason": (
                "location_evidence is not an exact positive-evidence clause "
                "for this degradation in distortion_location"
            ),
        }
    if not contains_contiguous_tokens(matching_clause, target):
        return {
            "target": "",
            "location_evidence": matching_clause,
            "location_status": "rejected",
            "location_reason": (
                "target is not a contiguous phrase from location_evidence"
            ),
        }
    if (
        TARGET_FORBIDDEN_PATTERN.search(target)
        or MODERATE_EVIDENCE.search(target)
        or NEGATIVE_EVIDENCE.search(target)
        or any(pattern.search(target) for pattern in CATEGORY_PATTERNS.values())
    ):
        return {
            "target": "",
            "location_evidence": matching_clause,
            "location_status": "rejected",
            "location_reason": (
                "target contains degradation, severity, or restoration-action "
                "language"
            ),
        }
    if not target_is_linked_to_degradation(
        matching_clause,
        target,
        degradation_type,
    ):
        return {
            "target": "",
            "location_evidence": matching_clause,
            "location_status": "rejected",
            "location_reason": (
                "target is separated from this degradation by evidence for "
                "another degradation type"
            ),
        }

    return {
        "target": target,
        "location_evidence": matching_clause,
        "location_status": "accepted",
        "location_reason": "",
    }


def normalized_level(iqa: dict[str, Any], degradation_type: str, requested_level: str) -> str:
    if requested_level != "moderate":
        return "mild"

    severity_clauses = positive_clauses(iqa.get("distortion_severity"), degradation_type)
    if any(MODERATE_EVIDENCE.search(clause) for clause in severity_clauses):
        return "moderate"

    type_is_supported = bool(positive_clauses(iqa.get("distortion_type"), degradation_type))
    severity_is_globally_moderate = bool(MODERATE_EVIDENCE.search(str(iqa.get("distortion_severity", ""))))
    return "moderate" if type_is_supported and severity_is_globally_moderate else "mild"


def normalize_selected_degradations(
    payload: dict[str, Any],
    iqa: dict[str, Any],
    include_location: bool = True,
) -> list[dict[str, Any]]:
    selected = payload.get("selected_degradations")
    if not isinstance(selected, list):
        raise ValueError("selected_degradations must be a list")

    normalized = []
    seen = set()
    for item in selected:
        if not isinstance(item, dict):
            continue
        degradation_type = str(item.get("type") or "").strip()
        requested_level = str(item.get("level") or "mild").strip().lower()
        if degradation_type not in ALLOWED_TYPES or requested_level not in ALLOWED_LEVELS:
            continue
        if degradation_type in seen:
            continue

        supporting_fields = evidence_fields_for(iqa, degradation_type)
        if not ({"distortion_type", "distortion_severity"} & set(supporting_fields)):
            continue

        location = validate_location_target(
            item,
            iqa,
            degradation_type,
            include_location=include_location,
        )
        normalized.append(
            {
                "type": degradation_type,
                "level": normalized_level(iqa, degradation_type, requested_level),
                "evidence_fields": supporting_fields,
                **location,
            }
        )
        seen.add(degradation_type)
        if len(normalized) == 3:
            break
    return normalized


def compiled_action(
    item: dict[str, Any],
    include_location: bool = True,
) -> str:
    key = (item["type"], item["level"])
    use_location = (
        include_location
        and item.get("location_status") == "accepted"
        and bool(str(item.get("target") or "").strip())
    )
    if use_location:
        return LOCATION_ACTION_TEXT[key].format(
            target=str(item["target"]).strip()
        )
    return ACTION_TEXT[key]


def compile_suggestion(
    selected: list[dict[str, Any]],
    include_location: bool = True,
) -> str:
    if not selected:
        return FALLBACK_SUGGESTION

    localized = [bool(include_location) for _ in selected]

    def build():
        actions = [
            compiled_action(
                item,
                include_location=localized[index],
            )
            for index, item in enumerate(selected)
        ]
        return " ".join(actions + [PRESERVATION_SUFFIX])

    suggestion = build()
    if count_words(suggestion) <= SUGGESTION_MAX_WORDS:
        return suggestion

    # Preserve the highest-priority locations. Lower-priority actions fall
    # back to their generic deterministic wording before failing the record.
    for index in range(len(localized) - 1, -1, -1):
        localized[index] = False
        suggestion = build()
        if count_words(suggestion) <= SUGGESTION_MAX_WORDS:
            return suggestion

    raise ValueError(
        "compiled suggestion exceeds "
        f"{SUGGESTION_MAX_WORDS} words: {count_words(suggestion)}"
    )


def get_profile(record: dict[str, Any]) -> dict[str, Any]:
    unipercept_raw = record.get("unipercept_raw")
    if not isinstance(unipercept_raw, dict):
        raise ValueError("record is missing unipercept_raw")
    profile = unipercept_raw.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("record is missing unipercept_raw.profile")
    return profile


def get_iqa(record: dict[str, Any]) -> dict[str, Any]:
    iqa = get_profile(record).get("iqa")
    if not isinstance(iqa, dict):
        raise ValueError("record is missing unipercept_raw.profile.iqa")
    if not any(str(iqa.get(field) or "").strip() for field in IQA_FIELDS):
        raise ValueError("IQA fields are empty")
    return iqa


def replace_suggestion(
    record: dict[str, Any],
    suggestion: str,
    status: str = "complete",
) -> dict[str, Any]:
    output = copy.deepcopy(record)
    get_profile(output)["suggestion"] = str(suggestion)
    annotation_status = output.setdefault("annotation_status", {})
    if not isinstance(annotation_status, dict):
        annotation_status = {}
        output["annotation_status"] = annotation_status
    annotation_status["suggestion"] = str(status)
    return output


class QwenIqaSuggestionClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_retries: int = 3,
        include_location: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_retries = max(1, int(max_retries))
        self.include_location = bool(include_location)

    def complete(self, iqa: dict[str, Any]) -> dict[str, Any]:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=512,
                    messages=[
                        {
                            "role": "system",
                            "content": build_system_prompt(
                                self.include_location
                            ),
                        },
                        {
                            "role": "user",
                            "content": render_user_prompt(
                                iqa,
                                include_location=self.include_location,
                            ),
                        },
                    ],
                    extra_body={"enable_thinking": False},
                )
                content = response.choices[0].message.content or ""
                return parse_json_strict_or_extract(content)
            except Exception as exc:  # API SDKs expose several provider-specific errors.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"Qwen request failed after {self.max_retries} attempts: {last_error}")


def convert_record(
    record: dict[str, Any],
    client,
    include_location: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if include_location is None:
        include_location = bool(getattr(client, "include_location", True))
    iqa = get_iqa(record)
    payload = client.complete(iqa)
    selected = normalize_selected_degradations(
        payload,
        iqa,
        include_location=include_location,
    )
    suggestion = compile_suggestion(
        selected,
        include_location=include_location,
    )
    return replace_suggestion(record, suggestion), {
        "sample_id": record.get("sample_id"),
        "hq_path": record.get("hq_path"),
        "include_location": bool(include_location),
        "selected_degradations": selected,
        "suggestion": suggestion,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL record at {path}:{line_no} is not an object")
            records.append(item)
    return records


def record_resume_key(record: dict[str, Any]) -> tuple[str, str]:
    sample_id = str(record.get("sample_id") or "").strip()
    if sample_id:
        return "sample_id", sample_id
    hq_path = str(record.get("hq_path") or "").strip()
    if not hq_path:
        raise ValueError("Record is missing both sample_id and hq_path")
    return "hq_path", hq_path


def completed_record_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    completed = set()
    for item in load_jsonl(path):
        key = record_resume_key(item)
        if key in completed:
            raise ValueError(
                f"Resume output contains duplicate {key[0]}: {key[1]}"
            )
        completed.add(key)
    return completed


def completed_hq_paths(path: Path) -> set[str]:
    """Backward-compatible helper for legacy callers without sample_id."""
    return {
        value
        for key_type, value in completed_record_keys(path)
        if key_type == "hq_path"
    }


def append_jsonl(handle, record: dict[str, Any]):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def append_error(path: Path, payload: dict[str, Any], lock: threading.Lock):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate crop-local, location-aware IQA-only SR suggestions "
            "into a new JSONL file."
        )
    )
    parser.add_argument("--input", required=True, help="Source JSONL; it is never modified.")
    parser.add_argument("--output", required=True, help="New JSONL receiving copied records.")
    parser.add_argument("--model", default=os.getenv("IQA_SR_SUGGESTION_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--api-key", default=None, help="Prefer an environment variable instead.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--include-location",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include validated crop-local targets in suggestions. Use "
            "--no-include-location for legacy random-crop data."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append only missing records when output is partial. sample_id is "
            "preferred; legacy records fall back to hq_path."
        ),
    )
    parser.add_argument(
        "--error-log",
        default=None,
        help="Optional sidecar JSONL. Defaults to <output>.errors.jsonl.",
    )
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Optional new JSONL containing selected degradation evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_path}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Qwen API key is missing. Set {args.api_key_env} or pass --api-key."
        )

    records = load_jsonl(input_path)
    if args.limit:
        records = records[: args.limit]
    completed = completed_record_keys(output_path) if args.resume else set()
    pending = [
        record for record in records if record_resume_key(record) not in completed
    ]
    if not pending:
        print(f"[iqa_sr_suggestion] Nothing to do: {output_path}", flush=True)
        return 0

    client = QwenIqaSuggestionClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
        include_location=args.include_location,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = Path(args.error_log) if args.error_log else Path(str(output_path) + ".errors.jsonl")
    audit_path = Path(args.audit_output) if args.audit_output else None
    if audit_path and audit_path.exists() and not args.resume:
        raise FileExistsError(f"Audit output already exists; refusing to overwrite: {audit_path}")
    error_lock = threading.Lock()

    def safe_convert(record):
        try:
            return (
                *convert_record(
                    record,
                    client,
                    include_location=args.include_location,
                ),
                None,
            )
        except Exception as exc:
            fallback = replace_suggestion(
                record,
                FALLBACK_SUGGESTION,
                status="fallback",
            )
            audit = {
                "sample_id": record.get("sample_id"),
                "hq_path": record.get("hq_path"),
                "include_location": bool(args.include_location),
                "selected_degradations": [],
                "suggestion": FALLBACK_SUGGESTION,
                "fallback": True,
            }
            return fallback, audit, str(exc)

    first_record, first_audit, first_error = safe_convert(pending[0])
    mode = "a" if output_path.exists() else "x"
    audit_mode = "a" if audit_path and audit_path.exists() else "x"
    with output_path.open(mode, encoding="utf-8") as output_handle:
        audit_handle = audit_path.open(audit_mode, encoding="utf-8") if audit_path else None
        try:
            append_jsonl(output_handle, first_record)
            if audit_handle:
                append_jsonl(audit_handle, first_audit)
            if first_error:
                append_error(
                    error_path,
                    {
                        "sample_id": first_record.get("sample_id"),
                        "hq_path": first_record.get("hq_path"),
                        "error": first_error,
                        "fallback_suggestion": FALLBACK_SUGGESTION,
                    },
                    error_lock,
                )
            print(
                f"[iqa_sr_suggestion] 1/{len(pending)} "
                f"sample_id={pending[0].get('sample_id')} "
                f"hq_path={pending[0].get('hq_path')} fallback={bool(first_error)}",
                flush=True,
            )

            remaining = pending[1:]
            if args.workers == 1:
                converted_iter = map(safe_convert, remaining)
            else:
                executor = ThreadPoolExecutor(max_workers=args.workers)
                converted_iter = executor.map(safe_convert, remaining)

            try:
                for offset, (converted, audit, error) in enumerate(converted_iter, 2):
                    append_jsonl(output_handle, converted)
                    if audit_handle:
                        append_jsonl(audit_handle, audit)
                    if error:
                        append_error(
                            error_path,
                            {
                                "sample_id": converted.get("sample_id"),
                                "hq_path": converted.get("hq_path"),
                                "error": error,
                                "fallback_suggestion": FALLBACK_SUGGESTION,
                            },
                            error_lock,
                        )
                    print(
                        f"[iqa_sr_suggestion] {offset}/{len(pending)} "
                        f"hq_path={converted.get('hq_path')} fallback={bool(error)}",
                        flush=True,
                    )
            finally:
                if args.workers != 1:
                    executor.shutdown(wait=True)
        finally:
            if audit_handle:
                audit_handle.close()

    print(
        f"[iqa_sr_suggestion] Wrote new records={len(pending)} output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
