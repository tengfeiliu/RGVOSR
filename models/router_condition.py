import hashlib
import json
import math
import re
from dataclasses import dataclass

import torch


ROUTER_CONDITION_VERSION = "text8_v1"
ROUTER_CONDITION_KEYS = (
    "blur",
    "noise",
    "compression",
    "ringing_aliasing",
    "texture_loss",
    "photometric",
    "structure_risk",
    "hallucination_risk",
)

# Expert semantics: structure/deblur, artifact suppression, texture recovery,
# fidelity guard. Rows can be overridden from configuration.
DEFAULT_EXPERT_SCORE_MATRIX = (
    (1.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.9, 0.7, 0.0, 0.0, 0.0, 0.0),
    (0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.8, 1.0),
)

_KEY_PATTERNS = {
    "blur": (
        r"\bblur(?:red|ry|ring)?\b",
        r"\bsoft(?:ness| focus)?\b",
        r"\bdefocus(?:ed)?\b",
        r"\bout[- ]of[- ]focus\b",
        r"\bmotion smear(?:ing)?\b",
    ),
    "noise": (
        r"\bnoise\b",
        r"\bnoisy\b",
        r"\bgrain(?:y|iness)?\b",
        r"\bspeckl(?:e|ing)\b",
        r"\bchroma(?:tic)? noise\b",
    ),
    "compression": (
        r"\bjpe?g\b",
        r"\bcompression(?: artifacts?)?\b",
        r"\bblock(?:ing|iness|y| artifacts?)\b",
        r"\bmacroblocks?\b",
    ),
    "ringing_aliasing": (
        r"\br(?:ing|inging)(?: artifacts?)?\b",
        r"\bhalos?\b",
        r"\balias(?:ing|ed)?\b",
        r"\bjagged(?: edges?)?\b",
        r"\bmoire\b",
        r"\bstair[- ]?stepp(?:ing|ed)\b",
    ),
    "texture_loss": (
        r"\btexture loss\b",
        r"\blost textures?\b",
        r"\bdetail loss\b",
        r"\blost details?\b",
        r"\black(?:ing)? (?:of )?(?:fine )?details?\b",
        r"\bover[- ]?smooth(?:ed|ing|ness)?\b",
        r"\bpixelat(?:ed|ion)\b",
        r"\blow[- ]resolution\b",
        r"\brecover (?:fine )?textures?\b",
        r"\brecover\b[^.;,!?]{0,64}\b(?:edge and )?texture details?\b",
        r"\brestore (?:fine )?details?\b",
        r"\benhance edge sharpness\b",
    ),
    "photometric": (
        r"\bcolor (?:cast|shift|distortion)\b",
        r"\bwhite balance\b",
        r"\blow contrast\b",
        r"\bwashed[- ]out\b",
        r"\bunder[- ]?expos(?:ed|ure)\b",
        r"\bover[- ]?expos(?:ed|ure)\b",
        r"\bdesaturat(?:ed|ion)\b",
        r"\bsaturation (?:loss|shift)\b",
        r"\bcolor consistency\b",
    ),
    "structure_risk": (
        r"\btexts?\b",
        r"\breadability\b",
        r"\bfaces?\b",
        r"\bidentity\b",
        r"\bthin lines?\b",
        r"\brepeated patterns?\b",
        r"\bgeometr(?:y|ic)\b",
        r"\bglobal structure\b",
        r"\bstructural (?:detail|fidelity|integrity)\b",
    ),
    "hallucination_risk": (
        r"\bhallucinat(?:e|ed|ion|ions)\b",
        r"\bfalse details?\b",
        r"\bartificial textures?\b",
        r"\bover[- ]?sharpen(?:ed|ing)?\b",
        r"\bsemantic changes?\b",
        r"\bidentity changes?\b",
    ),
}

_SEVERITY_PATTERNS = (
    (r"\b(?:very slight|very subtle)\b", 0.1),
    (r"\b(?:extreme|extremely|dominant|critical)\b", 1.0),
    (r"\b(?:severe|severely|strong|strongly|heavy|heavily|major)\b", 0.75),
    (r"\b(?:moderate|moderately|noticeable|noticeably|medium)\b", 0.5),
    (r"\b(?:mild|mildly|minor|slight|slightly)\b", 0.25),
    (r"\b(?:subtle|subtly|minimal)\b", 0.1),
)
_NEGATION_PATTERNS = (
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bfree of\b",
    r"\bno evidence of\b",
    r"\blittle evidence of\b",
    r"\bnot visible\b",
    r"\bno visible\b",
)


def _safe_text(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _split_segments(text):
    text = _safe_text(text)
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(
            r"(?:[.;,!?\n]+|\b(?:but|while|whereas)\b|"
            r"\band\b(?=\s+(?:no|not|without|very|mild|moderate|severe|strong|heavy|subtle)))",
            text,
        )
        if part.strip()
    ]


def _match_distance(match, anchor_start, anchor_end):
    if match.end() <= anchor_start:
        return anchor_start - match.end()
    if match.start() >= anchor_end:
        return match.start() - anchor_end
    return 0


def _local_severity(segment, key_match, default):
    candidates = []
    for pattern in _NEGATION_PATTERNS:
        for match in re.finditer(pattern, segment):
            distance = _match_distance(match, key_match.start(), key_match.end())
            before_key = match.end() <= key_match.start()
            explicit_after_key = pattern in {
                r"\babsent\b",
                r"\bnot visible\b",
                r"\bno visible\b",
            }
            # A preservation clause such as "compression artifacts without
            # changing source structure" must not negate the degradation.
            if distance <= 20 and (before_key or explicit_after_key):
                candidates.append((distance, match.start() > key_match.start(), match.start(), 0.0))
    for pattern, value in _SEVERITY_PATTERNS:
        for match in re.finditer(pattern, segment):
            distance = _match_distance(match, key_match.start(), key_match.end())
            if distance <= 32:
                candidates.append((distance, match.start() > key_match.start(), match.start(), value))
    if not candidates:
        return float(default)
    candidates.sort(key=lambda item: item[:3])
    return float(candidates[0][3])


def _global_severity(text, default=0.5):
    text = _safe_text(text)
    matches = []
    for pattern, value in _SEVERITY_PATTERNS:
        matches.extend((match.start(), value) for match in re.finditer(pattern, text))
    return float(matches[0][1]) if matches else float(default)


def _extract_text_features(text, global_default=0.5):
    values = {key: 0.0 for key in ROUTER_CONDITION_KEYS}
    valid = {key: False for key in ROUTER_CONDITION_KEYS}
    for segment in _split_segments(text):
        for key, patterns in _KEY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, segment):
                    value = _local_severity(segment, match, global_default)
                    values[key] = max(values[key], value)
                    valid[key] = True
    return values, valid


def _strip_suggestion_boilerplate(text):
    text = _safe_text(text)
    for pattern in (
        r"\bpreserve the original exposure,? color relationships,? geometry,? and semantic content\b\. ?",
        r"\bdo not invent unsupported details\b\. ?",
        r"\bwithout changing (?:the )?(?:source )?(?:structure|content|geometry)\b",
        r"\bpreserve (?:the )?(?:original )?(?:global )?(?:geometry|structure|semantic content)\b",
        r"\bmaintain (?:the )?(?:original )?(?:global )?(?:geometry|structure|semantic content)\b",
        r"\bwithout (?:changing|altering) (?:the )?(?:original )?(?:semantic content|geometry|structure)\b",
    ):
        text = re.sub(pattern, " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class RouterCondition:
    values: tuple
    valid_mask: tuple
    confidence: float
    source_hash: str
    version: str = ROUTER_CONDITION_VERSION

    def as_dict(self):
        return {
            "values": list(self.values),
            "valid_mask": list(self.valid_mask),
            "confidence": float(self.confidence),
            "source": "iqa_suggestion_text",
            "source_hash": self.source_hash,
            "extractor_version": self.version,
            "keys": list(ROUTER_CONDITION_KEYS),
        }


def extract_router_condition(profile, version=ROUTER_CONDITION_VERSION):
    if version != ROUTER_CONDITION_VERSION:
        raise ValueError(
            f"Unsupported router condition extractor version: {version}; "
            f"expected {ROUTER_CONDITION_VERSION}"
        )
    profile = profile if isinstance(profile, dict) else {}
    iqa = profile.get("iqa")
    iqa = iqa if isinstance(iqa, dict) else {}
    suggestion_raw = _safe_text(profile.get("suggestion"))
    suggestion = _strip_suggestion_boilerplate(suggestion_raw)

    severity_default = _global_severity(iqa.get("distortion_severity"), default=0.5)
    severity_values, severity_valid = _extract_text_features(
        iqa.get("distortion_severity"),
        global_default=severity_default,
    )
    type_values, type_valid = _extract_text_features(
        iqa.get("distortion_type"),
        # A generic field such as "severe" applies to all listed types. Once
        # key-specific severities exist, each explicit key overrides a neutral
        # type default instead, preventing cross-degradation leakage.
        global_default=0.5 if any(severity_valid.values()) else severity_default,
    )
    quality_values, quality_valid = _extract_text_features(
        iqa.get("overall_quality"),
        global_default=severity_default,
    )
    suggestion_values, suggestion_valid = _extract_text_features(suggestion, global_default=0.5)

    iqa_values = {}
    iqa_valid = {}
    for key in ROUTER_CONDITION_KEYS:
        # Explicit per-degradation severity overrides a generic type mention.
        if severity_valid[key]:
            iqa_values[key] = severity_values[key]
        elif type_valid[key]:
            iqa_values[key] = type_values[key]
        elif quality_valid[key]:
            iqa_values[key] = quality_values[key]
        else:
            iqa_values[key] = 0.0
        iqa_valid[key] = severity_valid[key] or type_valid[key] or quality_valid[key]

    location_values, location_valid = _extract_text_features(
        iqa.get("distortion_location"),
        global_default=0.5,
    )
    if location_valid["structure_risk"]:
        iqa_values["structure_risk"] = max(
            iqa_values["structure_risk"],
            location_values["structure_risk"],
        )
        iqa_valid["structure_risk"] = True

    values = []
    masks = []
    confidence_sum = 0.0
    risk_keys = {"structure_risk", "hallucination_risk"}
    for key in ROUTER_CONDITION_KEYS:
        has_iqa = bool(iqa_valid[key])
        has_suggestion = bool(suggestion_valid[key])
        if key in risk_keys:
            value = max(iqa_values[key], suggestion_values[key])
        elif has_iqa and has_suggestion:
            value = 0.8 * iqa_values[key] + 0.2 * suggestion_values[key]
        elif has_iqa:
            value = iqa_values[key]
        elif has_suggestion:
            value = 0.6 * suggestion_values[key]
        else:
            value = 0.0
        valid = has_iqa or has_suggestion
        values.append(max(0.0, min(1.0, float(value))))
        masks.append(1.0 if valid else 0.0)
        if valid:
            confidence_sum += 0.95 if has_iqa and has_suggestion else (0.9 if has_iqa else 0.65)

    # Coverage-aware: sparse text conditions receive weaker teacher supervision.
    confidence = confidence_sum / len(ROUTER_CONDITION_KEYS)
    source_payload = {
        "iqa": {key: _safe_text(iqa.get(key)) for key in sorted(iqa)},
        "suggestion": suggestion_raw,
        "version": version,
    }
    source_hash = hashlib.sha256(
        json.dumps(source_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Router condition extraction produced a non-finite or out-of-range value")
    return RouterCondition(
        values=tuple(values),
        valid_mask=tuple(masks),
        confidence=float(confidence),
        source_hash=source_hash,
        version=version,
    )


def router_condition_tensors(
    profile,
    device=None,
    dtype=torch.float32,
    version=ROUTER_CONDITION_VERSION,
):
    condition = extract_router_condition(profile, version=version)
    values = torch.tensor(condition.values, device=device, dtype=dtype)
    valid_mask = torch.tensor(condition.valid_mask, device=device, dtype=dtype)
    confidence = torch.tensor(condition.confidence, device=device, dtype=dtype)
    return values, valid_mask, confidence


def condition_to_expert_scores(condition, score_matrix=None):
    if condition.shape[-1] != len(ROUTER_CONDITION_KEYS):
        raise ValueError(
            f"Expected router condition dimension {len(ROUTER_CONDITION_KEYS)}, "
            f"got {condition.shape[-1]}"
        )
    matrix = score_matrix if score_matrix is not None else DEFAULT_EXPERT_SCORE_MATRIX
    matrix = torch.as_tensor(matrix, device=condition.device, dtype=condition.dtype)
    if matrix.ndim != 2 or matrix.shape[1] != condition.shape[-1]:
        raise ValueError(
            f"Expert score matrix must have shape [E, {condition.shape[-1]}], "
            f"got {tuple(matrix.shape)}"
        )
    return condition @ matrix.transpose(0, 1)


def condition_to_expert_target(condition, temperature=0.7, score_matrix=None):
    temperature = max(float(temperature), 1e-6)
    scores = condition_to_expert_scores(condition.float(), score_matrix=score_matrix)
    return torch.softmax(scores / temperature, dim=-1).to(dtype=condition.dtype)
