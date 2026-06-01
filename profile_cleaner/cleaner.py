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


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[’'.-][A-Za-z0-9]+)*")
IAA_MAX_WORDS = 50
IQA_MAX_WORDS = 350
SUGGESTION_MAX_WORDS = 80
IQA_FIELD_ORDER = ("distortion_location", "distortion_severity", "distortion_type", "overall_quality")


class ProfileCleaner:
    """Clean a single UniPercept profile using LLM prompts and local fallback checks."""

    def __init__(self, llm_client, max_retries=2, verbose=False, enable_required_iqa_fallback=True):
        self.llm_client = llm_client
        self.max_retries = int(max_retries)
        self.verbose = bool(verbose)
        self.enable_required_iqa_fallback = bool(enable_required_iqa_fallback)

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
        raw = self._complete("Prompt B", render_prompt_b(profile_without_ista(profile)))
        return self._parse_or_repair_json(raw, "Prompt B")

    def clean_one(self, profile: dict) -> dict:
        """Clean one bare profile dictionary and return a cleaned copy."""
        structure_errors = validate_profile_structure(profile)
        if structure_errors and not isinstance(profile, dict):
            raise ValueError("; ".join(structure_errors))

        original = copy.deepcopy(profile)
        cleaned = self._call_prompt_b(original)
        cleaned = self._restore_ista_if_needed(original, cleaned)
        cleaned = self._ensure_required_iqa_fields(cleaned, original)

        report = validate_strict_separation(cleaned)
        if not report["valid"]:
            self._log(
                "Local fallback repair start "
                f"iaa_violations={len(report['iaa_violations'])} "
                f"iqa_violations={len(report['iqa_violations'])}"
            )
            cleaned = self.local_fallback_repair(cleaned)
            cleaned = self._ensure_required_iqa_fields(cleaned, original)
            repaired_report = validate_strict_separation(cleaned)
            self._log(f"Local fallback repair done valid={repaired_report['valid']}")

        cleaned = normalize_profile_lengths(cleaned)
        cleaned = self._ensure_required_iqa_fields(cleaned, original)
        final_report = validate_strict_separation(cleaned)
        if not final_report["valid"]:
            self._log(
                "Local fallback repair start after length normalization "
                f"iaa_violations={len(final_report['iaa_violations'])} "
                f"iqa_violations={len(final_report['iqa_violations'])}"
            )
            cleaned = normalize_profile_lengths(self.local_fallback_repair(cleaned))
            cleaned = self._ensure_required_iqa_fields(cleaned, original)
        else:
            self._log("Strict separation valid")
        return self._restore_ista_if_needed(original, cleaned)

    def _ensure_required_iqa_fields(self, cleaned: dict, original: dict) -> dict:
        if not self.enable_required_iqa_fallback:
            return cleaned
        return ensure_required_iqa_fields(cleaned, original)

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
    """Normalize cleaned profile word budgets without changing non-profile sections."""
    normalized = copy.deepcopy(profile)
    if isinstance(normalized.get("iaa"), dict):
        normalized["iaa"] = compact_iaa_summary(normalized["iaa"])
    if isinstance(normalized.get("iqa"), dict):
        normalized["iqa"] = normalize_iqa_length(normalized["iqa"])
    normalized = normalize_suggestion_length(normalized)
    return normalized


def profile_without_ista(profile: dict) -> dict:
    """Return a prompt payload without ISTA, which is restored after LLM cleaning."""
    if not isinstance(profile, dict):
        return profile
    stripped = copy.deepcopy(profile)
    stripped.pop("ista", None)
    return stripped


def ensure_required_iqa_fields(profile: dict, original_profile: dict | None = None) -> dict:
    """Ensure the core IQA fields remain present and non-empty after LLM cleanup."""
    repaired = copy.deepcopy(profile)
    if not isinstance(repaired, dict):
        return repaired
    if not isinstance(repaired.get("iqa"), dict):
        repaired["iqa"] = {}

    current_iqa = repaired["iqa"]
    original_iqa = {}
    if isinstance(original_profile, dict) and isinstance(original_profile.get("iqa"), dict):
        original_iqa = original_profile["iqa"]

    for key in IQA_FIELD_ORDER:
        if not _has_text(current_iqa.get(key)):
            current_iqa[key] = _iqa_replacement_text(key, original_iqa, current_iqa)
    return repaired


def _has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return any(str(text or "").strip() for _, text in iter_string_fields(value))


def _iqa_replacement_text(key: str, original_iqa: dict, current_iqa: dict) -> str:
    candidates = _valid_items_from_value(original_iqa.get(key), IQA_FORBIDDEN_TERMS)
    if not candidates:
        candidates = _valid_items_from_value(current_iqa.get(key), IQA_FORBIDDEN_TERMS)
    if not candidates:
        candidates = _valid_items_from_value(original_iqa, IQA_FORBIDDEN_TERMS)
    if not candidates:
        return IQA_PLACEHOLDER
    return _format_bullets(candidates[:3])


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

    summary = _short_summary(candidates, IAA_MAX_WORDS) or IAA_PLACEHOLDER
    summary = _truncate_text_to_words(summary, IAA_MAX_WORDS)

    for key, value in list(compacted.items()):
        if key == target_key:
            compacted[key] = summary
        elif isinstance(value, str):
            compacted[key] = ""

    if target_key not in compacted:
        compacted[target_key] = summary
    return compacted


def normalize_iqa_length(iqa: dict) -> dict:
    """Trim IQA text to the configured word budget when enough content exists."""
    normalized = copy.deepcopy(iqa)
    if not isinstance(normalized, dict):
        return normalized
    if count_words(normalized) <= IQA_MAX_WORDS:
        return normalized

    ordered_keys = [key for key in IQA_FIELD_ORDER if key in normalized]
    ordered_keys.extend(key for key in normalized if key not in ordered_keys)
    assignments: dict[str, list[str]] = {key: [] for key in ordered_keys}
    items_by_key = {key: _valid_items_from_value(normalized.get(key), IQA_FORBIDDEN_TERMS) for key in ordered_keys}
    item_indices = {key: 0 for key in ordered_keys}

    required_keys_with_items = [key for key in IQA_FIELD_ORDER if key in items_by_key and items_by_key[key]]
    if required_keys_with_items:
        reserve_words = max(1, IQA_MAX_WORDS // len(required_keys_with_items))
        for key in required_keys_with_items:
            available = IQA_MAX_WORDS - count_words(assignments)
            if available <= 0:
                break
            item = _truncate_text_to_words(items_by_key[key][0], min(reserve_words, available))
            if item:
                assignments[key].append(item)
            item_indices[key] = 1

    made_progress = True
    while made_progress and count_words(assignments) < IQA_MAX_WORDS:
        made_progress = False
        for key in ordered_keys:
            index = item_indices.get(key, 0)
            if index >= len(items_by_key.get(key, [])):
                continue
            item = items_by_key[key][index]
            if _append_if_within_word_budget(assignments, key, item, IQA_MAX_WORDS):
                item_indices[key] = index + 1
                made_progress = True
                continue
            if _append_truncated_to_word_budget(assignments, key, item, IQA_MAX_WORDS):
                item_indices[key] = index + 1
                made_progress = True
            else:
                item_indices[key] = len(items_by_key[key])

    for key in ordered_keys:
        if isinstance(normalized.get(key), str):
            normalized[key] = _format_bullets(assignments.get(key, []))

    if not any(isinstance(value, str) and value.strip() for value in normalized.values()):
        target_key = "overall_quality" if "overall_quality" in normalized else next(iter(normalized), "overall_quality")
        normalized[target_key] = IQA_PLACEHOLDER
    return normalized


def normalize_suggestion_length(profile: dict) -> dict:
    """Trim profile-level suggestion to the configured word budget."""
    normalized = copy.deepcopy(profile)
    if isinstance(normalized.get("suggestion"), str):
        normalized["suggestion"] = _truncate_text_to_words(normalized["suggestion"], SUGGESTION_MAX_WORDS)
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


def _short_summary(candidates: list[str], max_words: int) -> str:
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
        if count_words(candidate) <= max_words:
            summary_parts.append(item)
            if len(summary_parts) >= 2:
                break
        elif not summary_parts:
            return _truncate_text_to_words(item, max_words)
    return " ".join(summary_parts)


def _truncate_text_to_words(text: str, max_words: int) -> str:
    text = _clean_item(text)
    if max_words <= 0:
        return ""
    matches = list(WORD_PATTERN.finditer(text))
    if len(matches) <= max_words:
        return text
    trimmed = text[: matches[max_words - 1].end()].rstrip()
    return trimmed.rstrip(" ,;:-")


def _clean_item(item: str) -> str:
    return re.sub(r"\s+", " ", str(item or "")).strip()


def count_words(value: Any) -> int:
    """Count English-like words in a string or nested string fields."""
    if isinstance(value, str):
        return len(WORD_PATTERN.findall(value))
    return sum(count_words(text) for _, text in iter_string_fields(value))


def _append_if_within_word_budget(assignments: dict[str, list[str]], key: str, item: str, max_words: int) -> bool:
    assignments[key].append(item)
    if count_words(assignments) <= max_words:
        return True
    assignments[key].pop()
    return False


def _append_truncated_to_word_budget(assignments: dict[str, list[str]], key: str, item: str, max_words: int) -> bool:
    available = max_words - count_words(assignments)
    if available <= 0:
        return False
    truncated = _truncate_text_to_words(item, available)
    if truncated and _append_if_within_word_budget(assignments, key, truncated, max_words):
        return True
    return False


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)
