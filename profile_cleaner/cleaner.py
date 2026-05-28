"""LLM-backed profile cleaning orchestration."""

import copy
import time
from typing import Any

from .config import IAA_PLACEHOLDER, IQA_PLACEHOLDER
from .json_utils import parse_json_strict_or_extract
from .prompts import render_json_repair_prompt, render_prompt_b, render_prompt_c
from .validators import (
    IAA_FORBIDDEN_TERMS,
    IQA_FORBIDDEN_TERMS,
    find_forbidden_terms,
    remove_duplicate_bullets,
    split_bullets,
    validate_profile_structure,
    validate_strict_separation,
)


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

    def _call_prompt_c(self, profile: dict) -> dict:
        raw = self._complete("Prompt C", render_prompt_c(profile))
        return self._parse_or_repair_json(raw, "Prompt C")

    def clean_one(self, profile: dict) -> dict:
        """Clean one bare profile dictionary and return a cleaned copy."""
        structure_errors = validate_profile_structure(profile)
        if structure_errors and not isinstance(profile, dict):
            raise ValueError("; ".join(structure_errors))

        original = copy.deepcopy(profile)
        cleaned = self._call_prompt_b(original)
        cleaned = self._call_prompt_c(cleaned)

        for _ in range(max(self.max_retries, 0)):
            report = validate_strict_separation(cleaned)
            if report["valid"]:
                self._log("Strict separation valid")
                return self._restore_ista_if_needed(original, cleaned)
            self._log(
                "Strict separation retry "
                f"iaa_violations={len(report['iaa_violations'])} "
                f"iqa_violations={len(report['iqa_violations'])}"
            )
            cleaned = self._call_prompt_c(cleaned)

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
