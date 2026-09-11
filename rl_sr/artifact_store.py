"""Atomic, portable JSONL storage for RL-SR artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from rl_sr.schema import canonical_json


T = TypeVar("T")


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: str | Path, records: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.as_dict() if hasattr(record, "as_dict") else record
            handle.write(canonical_json(payload) + "\n")
    temporary.replace(path)


def read_jsonl(path: str | Path, record_type: type[T] | None = None) -> list[T | dict[str, Any]]:
    rows: list[T | dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if record_type is not None:
                if not hasattr(record_type, "from_dict"):
                    raise TypeError(f"{record_type} does not implement from_dict")
                rows.append(record_type.from_dict(payload))
            else:
                rows.append(payload)
    return rows
