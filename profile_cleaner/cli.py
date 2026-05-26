"""Command line interface for cleaning nested UniPercept profiles."""

import argparse
import copy
import json
import os
from pathlib import Path

from .cleaner import ProfileCleaner
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_ERROR_LOG,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    PROFILE_PATH,
)
from .json_utils import load_json_or_jsonl, save_json_or_jsonl
from .llm_client import LLMClient
from .validators import validate_profile_structure, validate_strict_separation


def build_parser():
    parser = argparse.ArgumentParser(description="Clean UniPercept IAA/IQA profiles in JSON or JSONL records.")
    parser.add_argument("--input", required=True, help="Input JSON, JSONL, or directory.")
    parser.add_argument("--output", required=True, help="Output JSON, JSONL, or directory.")
    parser.add_argument("--jsonl", action="store_true", help="Treat files as JSONL.")
    parser.add_argument("--recursive", action="store_true", help="Recursively process a directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    parser.add_argument("--model", default=os.getenv("PROFILE_CLEANER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("PROFILE_CLEANER_TEMPERATURE", DEFAULT_TEMPERATURE)))
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--dry-run", action="store_true", help="Validate records without calling the LLM or writing outputs.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--error-log", default=DEFAULT_ERROR_LOG)
    return parser


def build_cleaner(args):
    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
    )
    return ProfileCleaner(client, max_retries=args.max_retries, verbose=args.verbose)


def get_nested_profile(record: dict):
    value = record
    for key in PROFILE_PATH:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, dict) else None


def set_nested_profile(record: dict, profile: dict):
    target = record
    for key in PROFILE_PATH[:-1]:
        target = target[key]
    target[PROFILE_PATH[-1]] = profile


def profile_summary(profile):
    if not isinstance(profile, dict):
        return {"type": type(profile).__name__}
    return {
        "iaa_fields": len(profile.get("iaa") or {}) if isinstance(profile.get("iaa"), dict) else 0,
        "iqa_fields": len(profile.get("iqa") or {}) if isinstance(profile.get("iqa"), dict) else 0,
        "has_ista": "ista" in profile,
    }


def append_error(error_log: Path, input_file: Path, item_index: int, error: str, profile):
    error_log = Path(error_log)
    error_log.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_file": str(input_file),
        "item_index": item_index,
        "error": str(error),
        "profile_summary": profile_summary(profile),
    }
    with error_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def clean_records(records, cleaner, input_file: Path, error_log: Path, dry_run=False, verbose=False):
    cleaned = []
    for index, record in enumerate(records):
        output_record = copy.deepcopy(record)
        profile = get_nested_profile(output_record)
        if profile is None:
            if verbose:
                print(f"{input_file}:{index} missing unipercept_raw.profile")
            if not dry_run:
                append_error(error_log, input_file, index, "missing unipercept_raw.profile", None)
            cleaned.append(output_record)
            continue

        structure_errors = validate_profile_structure(profile)
        if dry_run:
            report = validate_strict_separation(profile)
            if verbose:
                print(f"{input_file}:{index} structure={structure_errors} valid={report['valid']}")
            cleaned.append(output_record)
            continue

        try:
            if structure_errors:
                append_error(error_log, input_file, index, "; ".join(structure_errors), profile)
            cleaned_profile = cleaner.clean_one(profile)
            set_nested_profile(output_record, cleaned_profile)
        except Exception as exc:
            append_error(error_log, input_file, index, str(exc), profile)
        cleaned.append(output_record)
    return cleaned


def iter_input_output_files(input_path: Path, output_path: Path, recursive=False, jsonl=False):
    if input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        suffixes = {".jsonl"} if jsonl else {".json", ".jsonl"}
        for file_path in sorted(item for item in input_path.glob(pattern) if item.is_file() and item.suffix.lower() in suffixes):
            rel = file_path.relative_to(input_path)
            yield file_path, output_path / rel
    else:
        yield input_path, output_path


def process_file(input_file: Path, output_file: Path, cleaner, args):
    file_jsonl = args.jsonl or input_file.suffix.lower() == ".jsonl"
    records = load_json_or_jsonl(input_file, jsonl=file_jsonl)
    cleaned = clean_records(
        records,
        cleaner,
        input_file=input_file,
        error_log=Path(args.error_log),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    if not args.dry_run:
        save_json_or_jsonl(cleaned, output_file, jsonl=file_jsonl, overwrite=args.overwrite)
    return len(records)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        parser.error(f"Input does not exist: {input_path}")
    if input_path.is_dir() and not args.recursive:
        parser.error("--recursive is required for directory input")

    cleaner = None if args.dry_run else build_cleaner(args)
    total = 0
    for input_file, output_file in iter_input_output_files(input_path, output_path, args.recursive, args.jsonl):
        total += process_file(input_file, output_file, cleaner, args)
    if args.verbose or args.dry_run:
        print(f"Processed {total} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
