"""LLM-backed profile cleaning orchestration."""

import copy
import re
import time
from typing import Any

from .config import IAA_PLACEHOLDER, IQA_PLACEHOLDER
from .json_utils import parse_json_strict_or_extract
from .prompts import render_json_repair_prompt, render_prompt_b
from .validators import (
    IAA_FORBIDDEN_TERMS,
    IQA_FORBIDDEN_TERMS,
    find_forbidden_terms,
    iter_string_fields,
    remove_duplicate_bullets,
    split_bullets,
    validate_profile_structure,
    validate_strict_separation,
)


IAA_TARGET_CHARS = 100
IQA_TARGET_MIN_CHARS = 350
IQA_TARGET_MAX_CHARS = 400
IQA_FIELD_ORDER = ("distortion_location", "distortion_severity", "distortion_type", "overall_quality")


class ProfileCleaner:
    """Clean a single UniPercept profile using LLM prompts and local fallback checks."""

    def __init__(self, llm_client, max_retries=2, verbose=False):
        self.llm_client = llm_client
        self.max_retries = int(max_retries)
        self.verbose = bool(verbose)

    def _log(self, message: str):
        if self.verbose:
            print(f"[profile_cleaner] {message}", flush=True)

    def _complete(self, stage: str, prompt: str) -> str:
        self._log(f"{stage} start prompt_chars={len(prompt)}")
        started = time.time()
        response = self.llm_client.complete(prompt)
        self._log(f"{stage} done seconds={time.time() - started:.1f} response_chars={len(response or '')}")
        return response

    def _parse_or_repair_json(self, raw_output: str, stage: str) -> dict:
        try:
            return parse_json_strict_or_extract(raw_output)
        except ValueError as first_error:
            repair_prompt = render_json_repair_prompt(raw_output)
            self._log(f"JSON repair start after {stage}: {first_error}")
            repaired = self._complete("JSON repair", repair_prompt)
            try:
                return parse_json_strict_or_extract(repaired)
            except ValueError as second_error:
                raise ValueError(f"Unable to parse or repair JSON: {first_error}; repair failed: {second_error}") from second_error

    def _call_prompt_b(self, profile: dict) -> dict:
        raw = self._complete("Prompt B", render_prompt_b(profile))
        return self._parse_or_repair_json(raw, "Prompt B")

    def clean_one(self, profile: dict) -> dict:
        """Clean one bare profile dictionary and return a cleaned copy."""
        structure_errors = validate_profile_structure(profile)
        if structure_errors and not isinstance(profile, dict):
            raise ValueError("; ".join(structure_errors))

        original = copy.deepcopy(profile)
        cleaned = self._call_prompt_b(original)
        cleaned = self._restore_ista_if_needed(original, cleaned)

        report = validate_strict_separation(cleaned)
        if not report["valid"]:
            self._log(
                "Local fallback repair start "
                f"iaa_violations={len(report['iaa_violations'])} "
                f"iqa_violations={len(report['iqa_violations'])}"
            )
            cleaned = self.local_fallback_repair(cleaned)
            repaired_report = validate_strict_separation(cleaned)
            self._log(f"Local fallback repair done valid={repaired_report['valid']}")

        cleaned = normalize_profile_lengths(cleaned)
        final_report = validate_strict_separation(cleaned)
        if not final_report["valid"]:
            self._log(
                "Local fallback repair start after length normalization "
                f"iaa_violations={len(final_report['iaa_violations'])} "
                f"iqa_violations={len(final_report['iqa_violations'])}"
            )
            cleaned = normalize_profile_lengths(self.local_fallback_repair(cleaned))
        else:
            self._log("Strict separation valid")
        return self._restore_ista_if_needed(original, cleaned)

    def clean_many(self, profiles: list[dict]) -> list[dict]:
        """Clean a list of bare profile dictionaries."""
        return [self.clean_one(profile) for profile in profiles]

    def _restore_ista_if_needed(self, original: dict, cleaned: dict) -> dict:
        if isinstance(original, dict) and "ista" in original:
            cleaned = copy.deepcopy(cleaned)
            cleaned["ista"] = copy.deepcopy(original["ista"])
        return cleaned

    def local_fallback_repair(self, profile: dict) -> dict:
        """Conservatively delete contaminated IAA/IQA sentences and fill empty fields."""
        repaired = copy.deepcopy(profile)
        if isinstance(repaired.get("iaa"), dict):
            repaired["iaa"] = _repair_text_tree(repaired["iaa"], IAA_FORBIDDEN_TERMS, IAA_PLACEHOLDER)
        if isinstance(repaired.get("iqa"), dict):
            repaired["iqa"] = _repair_text_tree(repaired["iqa"], IQA_FORBIDDEN_TERMS, IQA_PLACEHOLDER)
        return repaired


def normalize_profile_lengths(profile: dict) -> dict:
    """Normalize cleaned profile text lengths without changing non-IAA/IQA sections."""
    normalized = copy.deepcopy(profile)
    if isinstance(normalized.get("iaa"), dict):
        normalized["iaa"] = compact_iaa_summary(normalized["iaa"])
    if isinstance(normalized.get("iqa"), dict):
        normalized["iqa"] = normalize_iqa_length(normalized["iqa"])
    return normalized


def compact_iaa_summary(iaa: dict) -> dict:
    """Place a short IAA summary in comprehensive and empty other string fields."""
    compacted = copy.deepcopy(iaa)
    if not isinstance(compacted, dict):
        return compacted

    target_key = "comprehensive" if "comprehensive" in compacted else next(iter(compacted), "comprehensive")
    candidates = []
    if isinstance(compacted.get(target_key), str):
        candidates.extend(_valid_items_from_text(compacted[target_key], IAA_FORBIDDEN_TERMS))
    for key, value in compacted.items():
        if key == target_key:
            continue
        candidates.extend(_valid_items_from_value(value, IAA_FORBIDDEN_TERMS))

    summary = _short_summary(candidates, IAA_TARGET_CHARS) or IAA_PLACEHOLDER
    summary = _truncate_text(summary, IAA_TARGET_CHARS)

    for key, value in list(compacted.items()):
        if key == target_key:
            compacted[key] = summary
        elif isinstance(value, str):
            compacted[key] = ""

    if target_key not in compacted:
        compacted[target_key] = summary
    return compacted


def normalize_iqa_length(iqa: dict) -> dict:
    """Trim IQA text to the 350-400 character target when enough content exists."""
    normalized = copy.deepcopy(iqa)
    if not isinstance(normalized, dict):
        return normalized
    if _string_tree_length(normalized) <= IQA_TARGET_MAX_CHARS:
        return normalized

    ordered_keys = [key for key in IQA_FIELD_ORDER if key in normalized]
    ordered_keys.extend(key for key in normalized if key not in ordered_keys)
    assignments: dict[str, list[str]] = {key: [] for key in ordered_keys}

    for key in ordered_keys:
        for item in _valid_items_from_value(normalized.get(key), IQA_FORBIDDEN_TERMS):
            if _append_if_within_budget(assignments, key, item):
                continue
            if _assignments_length(assignments) < IQA_TARGET_MIN_CHARS:
                _append_truncated_to_budget(assignments, key, item)
            break

    for key in ordered_keys:
        if isinstance(normalized.get(key), str):
            normalized[key] = _format_bullets(assignments.get(key, []))

    if not any(isinstance(value, str) and value.strip() for value in normalized.values()):
        target_key = "overall_quality" if "overall_quality" in normalized else next(iter(normalized), "overall_quality")
        normalized[target_key] = IQA_PLACEHOLDER
    return normalized


def _repair_text_tree(value: Any, forbidden_terms: list[str], placeholder: str):
    if isinstance(value, str):
        return _repair_text_value(value, forbidden_terms, placeholder)
    if isinstance(value, dict):
        return {key: _repair_text_tree(item, forbidden_terms, placeholder) for key, item in value.items()}
    if isinstance(value, list):
        return [_repair_text_tree(item, forbidden_terms, placeholder) for item in value]
    return value


def _repair_text_value(text: str, forbidden_terms: list[str], placeholder: str) -> str:
    kept = []
    for item in split_bullets(text):
        if not find_forbidden_terms(item, forbidden_terms):
            kept.append(item)
    if not kept:
        return placeholder
    return remove_duplicate_bullets("\n".join(f"- {item}" for item in kept))


def _valid_items_from_value(value: Any, forbidden_terms: list[str]) -> list[str]:
    items = []
    if isinstance(value, str):
        return _valid_items_from_text(value, forbidden_terms)
    for _, text in iter_string_fields(value):
        items.extend(_valid_items_from_text(text, forbidden_terms))
    return items


def _valid_items_from_text(text: str, forbidden_terms: list[str]) -> list[str]:
    items = []
    seen = set()
    for item in split_bullets(text):
        item = _clean_item(item)
        key = re.sub(r"\s+", " ", item).strip().lower()
        if item and key not in seen and not find_forbidden_terms(item, forbidden_terms):
            seen.add(key)
            items.append(item)
    return items


def _short_summary(candidates: list[str], max_chars: int) -> str:
    summary_parts = []
    seen = set()
    for item in candidates:
        if not item:
            continue
        key = re.sub(r"\s+", " ", item).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        candidate = " ".join(summary_parts + [item])
        if len(candidate) <= max_chars:
            summary_parts.append(item)
            if len(summary_parts) >= 2:
                break
        elif not summary_parts:
            return _truncate_text(item, max_chars)
    return " ".join(summary_parts)


def _truncate_text(text: str, max_chars: int) -> str:
    text = _clean_item(text)
    if len(text) <= max_chars:
        return text
    trimmed = text[: max(max_chars, 0)].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0].rstrip()
    return trimmed.rstrip(" ,;:-")


def _clean_item(item: str) -> str:
    return re.sub(r"\s+", " ", str(item or "")).strip()


def _string_tree_length(value: Any) -> int:
    texts = [_clean_item(text) for _, text in iter_string_fields(value)]
    return len(" ".join(text for text in texts if text))


def _assignments_length(assignments: dict[str, list[str]]) -> int:
    return _string_tree_length({key: _format_bullets(items) for key, items in assignments.items()})


def _append_if_within_budget(assignments: dict[str, list[str]], key: str, item: str) -> bool:
    assignments[key].append(item)
    if _assignments_length(assignments) <= IQA_TARGET_MAX_CHARS:
        return True
    assignments[key].pop()
    return False


def _append_truncated_to_budget(assignments: dict[str, list[str]], key: str, item: str) -> bool:
    available = IQA_TARGET_MAX_CHARS - _assignments_length(assignments)
    if available < 24:
        return False
    for length in range(min(len(item), available), 23, -1):
        truncated = _truncate_text(item, length)
        if truncated and _append_if_within_budget(assignments, key, truncated):
            return True
    return False


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)
