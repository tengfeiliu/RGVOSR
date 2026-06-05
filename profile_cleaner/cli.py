"""Command line interface for cleaning nested UniPercept profiles."""

import argparse
import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


_ERROR_LOG_LOCK = threading.Lock()


def positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description="Clean UniPercept IAA/IQA profiles in JSON or JSONL records.")
    parser.add_argument("--input", required=True, help="Input JSON, JSONL, or directory.")
    parser.add_argument("--output", required=True, help="Output JSON, JSONL, or directory.")
    parser.add_argument("--jsonl", action="store_true", help="Treat files as JSONL.")
    parser.add_argument("--recursive", action="store_true", help="Recursively process a directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    parser.add_argument("--model", default=os.getenv("PROFILE_CLEANER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=float(os.getenv("PROFILE_CLEANER_TEMPERATURE", DEFAULT_TEMPERATURE)))
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--limit", type=int, default=0, help="Maximum records to process from each input file. 0 means no limit.")
    parser.add_argument("--workers", type=positive_int, default=4, help="Number of records to clean concurrently.")
    parser.add_argument("--dry-run", action="store_true", help="Validate records without calling the LLM or writing outputs.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--error-log", default=DEFAULT_ERROR_LOG)
    parser.add_argument(
        "--required-iqa-fallback",
        dest="required_iqa_fallback",
        action="store_true",
        default=False,
        help="Fill empty required IQA fields from the original profile after LLM cleanup.",
    )
    parser.add_argument(
        "--no-required-iqa-fallback",
        dest="required_iqa_fallback",
        action="store_false",
        help="Keep empty required IQA fields from the LLM output for debugging.",
    )
    return parser


def build_cleaner(args):
    client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
    )
    return ProfileCleaner(
        client,
        max_retries=args.max_retries,
        verbose=True,
        enable_required_iqa_fallback=args.required_iqa_fallback,
    )


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
    with _ERROR_LOG_LOCK:
        with error_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def clean_record(record, index, total, cleaner, input_file: Path, error_log: Path, dry_run=False, verbose=False, progress=True):
    if progress:
        print(f"[profile_cleaner] {input_file}: record {index + 1}/{total} start", flush=True)
    output_record = copy.deepcopy(record)
    profile = get_nested_profile(output_record)
    if profile is None:
        if verbose or progress:
            print(f"{input_file}:{index} missing unipercept_raw.profile")
        if not dry_run:
            append_error(error_log, input_file, index, "missing unipercept_raw.profile", None)
        return output_record

    structure_errors = validate_profile_structure(profile)
    if dry_run:
        report = validate_strict_separation(profile)
        if verbose:
            print(f"{input_file}:{index} structure={structure_errors} valid={report['valid']}")
        elif progress:
            print(f"[profile_cleaner] {input_file}: record {index + 1}/{total} dry-run valid={report['valid']}", flush=True)
        return output_record

    try:
        if structure_errors:
            append_error(error_log, input_file, index, "; ".join(structure_errors), profile)
        cleaned_profile = cleaner.clean_one(profile)
        set_nested_profile(output_record, cleaned_profile)
        if progress:
            print(f"[profile_cleaner] {input_file}: record {index + 1}/{total} cleaned", flush=True)
    except Exception as exc:
        append_error(error_log, input_file, index, str(exc), profile)
        if progress:
            print(f"[profile_cleaner] {input_file}: record {index + 1}/{total} failed: {exc}", flush=True)
    return output_record


def clean_records(records, cleaner, input_file: Path, error_log: Path, dry_run=False, verbose=False, progress=True, workers=1):
    total = len(records)
    if workers <= 1 or total <= 1:
        return [
            clean_record(record, index, total, cleaner, input_file, error_log, dry_run, verbose, progress)
            for index, record in enumerate(records)
        ]

    cleaned = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(clean_record, record, index, total, cleaner, input_file, error_log, dry_run, verbose, progress): (
                index,
                record,
            )
            for index, record in enumerate(records)
        }
        for future in as_completed(futures):
            index, record = futures[future]
            try:
                cleaned[index] = future.result()
            except Exception as exc:
                if not dry_run:
                    append_error(error_log, input_file, index, str(exc), get_nested_profile(record))
                if progress:
                    print(f"[profile_cleaner] {input_file}: record {index + 1}/{total} future failed: {exc}", flush=True)
                cleaned[index] = copy.deepcopy(record)

    missing_indexes = [index for index, record in enumerate(cleaned) if record is None]
    if missing_indexes:
        raise RuntimeError(f"Missing cleaned records for indexes: {missing_indexes}")
    return cleaned


def append_jsonl_record(handle, record: dict):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def get_hq_path(record: dict):
    hq_path = record.get("hq_path") if isinstance(record, dict) else None
    return hq_path if hq_path else None


def load_completed_hq_paths(output_file: Path) -> set:
    records = load_json_or_jsonl(output_file, jsonl=True)
    completed = set()
    for record in records:
        hq_path = get_hq_path(record)
        if hq_path:
            completed.add(hq_path)
    return completed


def iter_input_output_files(input_path: Path, output_path: Path, recursive=False, jsonl=False):
    if input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        suffixes = {".jsonl"} if jsonl else {".json", ".jsonl"}
        for file_path in sorted(item for item in input_path.glob(pattern) if item.is_file() and item.suffix.lower() in suffixes):
            rel = file_path.relative_to(input_path)
            yield file_path, output_path / rel
    else:
        yield input_path, output_path


def clean_jsonl_record(record, index, total, cleaner, input_file: Path, error_log: Path, verbose=False):
    hq_path = get_hq_path(record)
    cleaned_record = clean_record(
        record,
        index,
        total,
        cleaner,
        input_file,
        error_log,
        False,
        verbose,
        True,
    )
    return index, hq_path, cleaned_record


def write_jsonl_result(handle, record: dict, hq_path, completed_hq_paths: set, written_hq_paths: set):
    if hq_path and hq_path in written_hq_paths:
        raise RuntimeError(f"Duplicate hq_path write attempted: {hq_path}")
    append_jsonl_record(handle, record)
    if hq_path:
        completed_hq_paths.add(hq_path)
        written_hq_paths.add(hq_path)


def process_file(input_file: Path, output_file: Path, cleaner, args):
    file_jsonl = args.jsonl or input_file.suffix.lower() == ".jsonl"
    records = load_json_or_jsonl(input_file, jsonl=file_jsonl)
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[profile_cleaner] Processing file {input_file} -> {output_file} records={len(records)}", flush=True)
    if file_jsonl and not args.dry_run:
        error_log = Path(args.error_log)
        completed_hq_paths = set()
        resume = output_file.exists() and not args.overwrite
        if resume:
            completed_hq_paths = load_completed_hq_paths(output_file)
            print(
                f"[profile_cleaner] Resuming {output_file}: completed_hq_paths={len(completed_hq_paths)}",
                flush=True,
            )
        scheduled_hq_paths = set()
        written_hq_paths = set()
        scheduled_count = 0
        appended_count = 0
        failed_future_count = 0
        output_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        skipped = 0
        total = len(records)
        with output_file.open(mode, encoding="utf-8") as handle:
            if args.workers <= 1:
                for index, record in enumerate(records):
                    hq_path = get_hq_path(record)
                    if hq_path and hq_path in completed_hq_paths:
                        skipped += 1
                        print(
                            f"[profile_cleaner] {input_file}: record {index + 1}/{total} skipped hq_path={hq_path}",
                            flush=True,
                        )
                        continue
                    if hq_path and hq_path in scheduled_hq_paths:
                        skipped += 1
                        print(
                            f"[profile_cleaner] {input_file}: record {index + 1}/{total} skipped duplicate hq_path={hq_path}",
                            flush=True,
                        )
                        continue
                    if hq_path:
                        scheduled_hq_paths.add(hq_path)
                    scheduled_count += 1
                    try:
                        cleaned_record = clean_record(
                            record,
                            index,
                            total,
                            cleaner,
                            input_file=input_file,
                            error_log=error_log,
                            dry_run=False,
                            verbose=args.verbose,
                            progress=True,
                        )
                    except Exception as exc:
                        failed_future_count += 1
                        append_error(error_log, input_file, index, str(exc), get_nested_profile(record))
                        print(
                            f"[profile_cleaner] {input_file}: record {index + 1}/{total} future failed: {exc}",
                            flush=True,
                        )
                        cleaned_record = copy.deepcopy(record)
                    write_jsonl_result(handle, cleaned_record, hq_path, completed_hq_paths, written_hq_paths)
                    appended_count += 1
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {}
                    for index, record in enumerate(records):
                        hq_path = get_hq_path(record)
                        if hq_path and hq_path in completed_hq_paths:
                            skipped += 1
                            print(
                                f"[profile_cleaner] {input_file}: record {index + 1}/{total} skipped hq_path={hq_path}",
                                flush=True,
                            )
                            continue
                        if hq_path and hq_path in scheduled_hq_paths:
                            skipped += 1
                            print(
                                f"[profile_cleaner] {input_file}: record {index + 1}/{total} skipped duplicate hq_path={hq_path}",
                                flush=True,
                            )
                            continue
                        if hq_path:
                            scheduled_hq_paths.add(hq_path)
                        scheduled_count += 1
                        future = executor.submit(
                            clean_jsonl_record,
                            record,
                            index,
                            total,
                            cleaner,
                            input_file,
                            error_log,
                            args.verbose,
                        )
                        futures[future] = (index, record, hq_path)
                    for future in as_completed(futures):
                        index, record, hq_path = futures[future]
                        try:
                            _, hq_path, cleaned_record = future.result()
                        except Exception as exc:
                            failed_future_count += 1
                            append_error(error_log, input_file, index, str(exc), get_nested_profile(record))
                            print(
                                f"[profile_cleaner] {input_file}: record {index + 1}/{total} future failed: {exc}",
                                flush=True,
                            )
                            cleaned_record = copy.deepcopy(record)
                        write_jsonl_result(handle, cleaned_record, hq_path, completed_hq_paths, written_hq_paths)
                        appended_count += 1

        if appended_count != scheduled_count:
            raise RuntimeError(
                f"JSONL write count mismatch for {output_file}: appended={appended_count} scheduled={scheduled_count}"
            )
        print(
            f"[profile_cleaner] Wrote {output_file} appended={appended_count} skipped={skipped} failed_futures={failed_future_count}",
            flush=True,
        )
        return len(records)

    cleaned = clean_records(
        records,
        cleaner,
        input_file=input_file,
        error_log=Path(args.error_log),
        dry_run=args.dry_run,
        verbose=args.verbose,
        progress=True,
        workers=args.workers,
    )
    if not args.dry_run:
        save_json_or_jsonl(cleaned, output_file, jsonl=file_jsonl, overwrite=args.overwrite)
        print(f"[profile_cleaner] Wrote {output_file}", flush=True)
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
    print(f"[profile_cleaner] Processed {total} records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
