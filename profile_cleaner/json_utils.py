"""JSON and JSONL helpers for profile cleaner inputs and LLM outputs."""

import json
import re
from pathlib import Path
from typing import Any


def safe_json_dumps(obj: Any) -> str:
    """Serialize JSON with stable Unicode-preserving formatting."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def load_json_or_jsonl(path: Path, jsonl: bool = False) -> list[dict]:
    """Load a JSON object/list or JSONL records from path."""
    path = Path(path)
    if jsonl:
        items = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL item at {path}:{line_no} is not an object")
                items.append(item)
        return items

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError(f"JSON list contains non-object items: {path}")
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"JSON root must be an object or list of objects: {path}")


def save_json_or_jsonl(items: list[dict], path: Path, jsonl: bool = False, overwrite: bool = False):
    """Save records as JSON or JSONL, refusing to overwrite unless requested."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return

    payload = items if len(items) != 1 else items[0]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _strip_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from text."""
    if text is None:
        raise ValueError("empty JSON text")
    text = _strip_fence(str(text))
    if not text:
        raise ValueError("empty JSON text")
    if text.lstrip().startswith("{"):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("No complete JSON object found")


def parse_json_strict_or_extract(text: str) -> dict:
    """Parse JSON directly, or extract the first JSON object and parse it."""
    if text is None:
        raise ValueError("empty JSON text")
    raw = _strip_fence(str(text))
    candidates = [raw]
    try:
        extracted = extract_json_object(raw)
    except ValueError:
        extracted = None
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    last_error = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            raise ValueError("JSON root must be an object")
        return parsed

    raise ValueError(f"invalid JSON object: {last_error}")
