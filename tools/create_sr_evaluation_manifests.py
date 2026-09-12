"""Build named RealLQ/RealLR evaluation manifests from the RL-SR YAML config.

The iterative runner accepts ``--dataset_dirs NAME=INPUT``. This tool turns the
configured inference JSONL records into one ``.txt`` input list per named
dataset. It never copies, filters, or replaces the source condition JSONL:
inference continues to read the original file for caption/IQA/suggestion lookup.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                row = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield line_number, row


def _resolve_path(value, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _dataset_specs(config: dict, repo_root: Path) -> list[dict]:
    evaluation = ((config.get("rl_sr") or {}).get("evaluation") or {})
    default_jsonl_path = evaluation.get("jsonl_path")
    raw_specs = evaluation.get("datasets")
    if raw_specs is None:
        # Backward-compatible fallback for older configs with a single optional
        # evaluation JSONL. New configs should always use named datasets.
        jsonl_path = default_jsonl_path or (config.get("data") or {}).get("jsonl_path")
        raw_specs = [{"name": "evaluation", "jsonl_path": jsonl_path}]
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("rl_sr.evaluation.datasets must be a non-empty list")

    parsed = []
    names = set()
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(f"rl_sr.evaluation.datasets[{index}] must be a mapping")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError(f"rl_sr.evaluation.datasets[{index}].name is required")
        if name in names:
            raise ValueError(f"Duplicate evaluation dataset name: {name}")
        if any(char in name for char in "=/\\"):
            raise ValueError(f"Evaluation dataset name contains an unsupported character: {name}")
        jsonl_value = raw.get("jsonl_path") or default_jsonl_path
        if not jsonl_value:
            raise ValueError(f"rl_sr.evaluation.datasets[{index}].jsonl_path is required")
        dataset_filter = raw.get("dataset_filter", name)
        expected_count = raw.get("expected_count", None)
        if expected_count is not None:
            expected_count = int(expected_count)
            if expected_count <= 0:
                raise ValueError(f"expected_count for {name} must be positive")
        parsed.append(
            {
                "name": name,
                "jsonl_path": _resolve_path(jsonl_value, repo_root),
                "dataset_filter": None if dataset_filter is None else str(dataset_filter),
                "expected_count": expected_count,
            }
        )
        names.add(name)
    return parsed


def _record_dataset_names(row: dict) -> list[str]:
    """Return dataset labels preserved by the known inference-cache schemas."""
    values = []
    for source in (row, row.get("raw_degradation_params"), row.get("result")):
        if not isinstance(source, dict):
            continue
        for key in ("dataset_name", "dataset"):
            value = source.get(key)
            if value is not None and str(value).strip():
                normalized = str(value).strip()
                if normalized not in values:
                    values.append(normalized)
    return values


def _matches_dataset(row: dict, dataset_filter: str | None) -> str | None:
    """Match a configured dataset name without changing the source JSONL.

    Older inference caches sometimes omit the top-level dataset label but keep
    the original evaluation directory in ``lq_path``.  Accept that directory
    token as a compatibility fallback.  An explicit metadata match always wins.
    """
    if dataset_filter is None:
        return "unfiltered"
    expected = str(dataset_filter).strip().casefold()
    if any(value.casefold() == expected for value in _record_dataset_names(row)):
        return "metadata"
    lq_path = row.get("lq_path")
    if isinstance(lq_path, str) and expected in lq_path.replace("\\", "/").casefold():
        return "lq_path"
    return None


def _selection_diagnostic(rows: list[dict]) -> dict:
    labels = []
    paths = []
    for row in rows:
        for label in _record_dataset_names(row):
            if label not in labels:
                labels.append(label)
        lq_path = row.get("lq_path")
        if isinstance(lq_path, str) and lq_path not in paths:
            paths.append(lq_path)
    return {"dataset_labels": labels[:20], "lq_path_examples": paths[:5]}


def _source_path_candidates(raw_path: str, jsonl_path: Path) -> list[Path]:
    path = Path(raw_path).expanduser()
    return [path] if path.is_absolute() else [path, jsonl_path.parent / path]


def build_manifests(config_path: Path, repo_root: Path, output_dir: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    datasets = []
    source_jsonl_paths = set()
    for spec in _dataset_specs(config, repo_root):
        jsonl_path = spec["jsonl_path"]
        source_jsonl_paths.add(str(jsonl_path))
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Evaluation JSONL does not exist for {spec['name']}: {jsonl_path}")
        source_rows = list(_read_jsonl(jsonl_path))
        input_values, seen_inputs, matching_rows = [], set(), 0
        match_sources = {"metadata": 0, "lq_path": 0, "unfiltered": 0}
        for line_number, row in source_rows:
            match_source = _matches_dataset(row, spec["dataset_filter"])
            if match_source is None:
                continue
            matching_rows += 1
            match_sources[match_source] += 1
            lq_path = row.get("lq_path")
            if not isinstance(lq_path, str) or not lq_path.strip():
                raise ValueError(f"Missing lq_path for {spec['name']} at {jsonl_path}:{line_number}")
            source_value = lq_path.strip()
            existing = next(
                (candidate for candidate in _source_path_candidates(source_value, jsonl_path) if candidate.is_file()),
                None,
            )
            if existing is None:
                raise FileNotFoundError(
                    f"Evaluation LQ image does not exist for {spec['name']} at {jsonl_path}:{line_number}: {source_value}"
                )
            dedupe_key = str(existing.resolve())
            if dedupe_key in seen_inputs:
                continue
            seen_inputs.add(dedupe_key)
            raw_path = Path(source_value).expanduser()
            # The runner executes at repository root. Preserve the JSONL value
            # when it already resolves there; otherwise write a runtime absolute
            # path for a JSONL-relative image. The original JSONL remains intact.
            input_values.append(source_value if raw_path.is_absolute() or raw_path.is_file() else dedupe_key)
        if matching_rows == 0:
            filter_text = spec["dataset_filter"]
            diagnostic = _selection_diagnostic([row for _, row in source_rows])
            raise ValueError(
                f"No rows for evaluation dataset '{spec['name']}' with dataset_filter='{filter_text}' in {jsonl_path}. "
                f"Observed dataset labels: {diagnostic['dataset_labels'] or '<none>'}; "
                f"lq_path examples: {diagnostic['lq_path_examples'] or '<none>'}."
            )
        if spec["expected_count"] is not None and len(input_values) != spec["expected_count"]:
            raise ValueError(
                f"Evaluation count mismatch for {spec['name']}: expected {spec['expected_count']}, found {len(input_values)}"
            )
        input_list = output_dir / f"{spec['name']}_lq_inputs.txt"
        _atomic_write(input_list, "\n".join(input_values) + "\n")
        datasets.append(
            {
                "name": spec["name"],
                "input_list": str(input_list),
                "jsonl_path": str(jsonl_path),
                "dataset_filter": spec["dataset_filter"],
                "match_sources": {key: value for key, value in match_sources.items() if value},
                "input_count": len(input_values),
                "expected_count": spec["expected_count"],
            }
        )

    if len(source_jsonl_paths) != 1:
        raise ValueError(
            "All rl_sr.evaluation.datasets must reference the same evaluation.jsonl_path. "
            "The iterative runner deliberately keeps one original condition JSONL."
        )
    condition_jsonl = next(iter(source_jsonl_paths))
    manifest = {
        "schema_version": "rl_sr_evaluation_v1",
        "condition_jsonl": str(condition_jsonl),
        "datasets": datasets,
        "path_policy": "source_jsonl_is_read_only_runtime_input_not_artifact_identity",
    }
    _atomic_write(output_dir / "evaluation_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def _load_manifest(output_dir: Path) -> dict:
    path = output_dir / "evaluation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--print_dataset_dirs", action="store_true")
    parser.add_argument("--print_condition_jsonl", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if args.print_dataset_dirs or args.print_condition_jsonl:
        manifest = _load_manifest(output_dir)
        if args.print_dataset_dirs:
            for dataset in manifest["datasets"]:
                print(f"{dataset['name']}={dataset['input_list']}")
        if args.print_condition_jsonl:
            print(manifest["condition_jsonl"])
        return
    manifest = build_manifests(
        config_path=Path(args.config).expanduser().resolve(),
        repo_root=Path(args.repo_root).expanduser().resolve(),
        output_dir=output_dir,
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "datasets"}, ensure_ascii=False, indent=2))
    print(json.dumps(manifest["datasets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
