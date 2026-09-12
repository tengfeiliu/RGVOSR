"""Create an inference input list directly from ``data.jsonl_path``.

The file keeps the JSONL's original LQ path spelling so the existing condition
lookup still finds its paired caption/profile record.  Absolute paths are never
written into RL artifact identities; this list is only a runtime input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield line_number, row


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def build_manifest(data_jsonl_path: Path, output_path: Path, label: str) -> dict:
    if not data_jsonl_path.is_file():
        raise FileNotFoundError(f"Paired data JSONL does not exist: {data_jsonl_path}")

    lq_paths: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for line_number, row in _read_jsonl(data_jsonl_path):
        lq_path = row.get("lq_path")
        if not isinstance(lq_path, str) or not lq_path.strip():
            raise ValueError(f"Missing lq_path at {data_jsonl_path}:{line_number}")
        # Preserve the source spelling in the manifest.  It is how the legacy
        # JSONL condition matcher identifies a record.  Existence checks accept
        # paths relative to the current repository as well as the JSONL folder.
        source_value = lq_path.strip()
        source_path = Path(source_value).expanduser()
        candidates = [source_path]
        if not source_path.is_absolute():
            candidates.append(data_jsonl_path.parent / source_path)
        existing = next((candidate for candidate in candidates if candidate.is_file()), None)
        if existing is None:
            missing.append(source_value)
            continue
        dedupe_key = str(existing.resolve())
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            # A JSONL-relative path must become usable from the repository CWD
            # where the inference process runs. This remains runtime-only; it
            # never contributes to RL artifact identity.
            lq_paths.append(source_value if source_path.is_absolute() or source_path.is_file() else dedupe_key)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" … (+{len(missing) - 5} more)"
        raise FileNotFoundError(
            f"{len(missing)} lq_path entries from {data_jsonl_path} do not exist: {preview}{suffix}"
        )
    if not lq_paths:
        raise RuntimeError(f"No LQ inputs found in {data_jsonl_path}")

    _atomic_write(output_path, "\n".join(lq_paths) + "\n")
    summary = {
        "label": label,
        "data_jsonl_path": str(data_jsonl_path),
        "input_count": len(lq_paths),
        "input_list": str(output_path),
        "path_policy": "original_jsonl_values_runtime_only",
    }
    _atomic_write(output_path.with_suffix(".manifest.json"), json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl_path", required=True, help="The paired data.jsonl_path from the config.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_manifest(Path(args.data_jsonl_path).expanduser(), Path(args.output).expanduser(), args.label)
    print(json.dumps(result, ensure_ascii=False, indent=2))
