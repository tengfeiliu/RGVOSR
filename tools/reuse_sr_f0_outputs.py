"""Select and copy existing F0 y_1 outputs without loading the SR model or IQA.

Source outputs are read-only. Runtime paths are used for matching/loading only;
adapter fingerprints use the existing content/relative-key hashing scheme.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_f0_cache(source_dir, lineage, checkpoint, sampling, conditions):
    """Validate only requested samples; IQA completion is not a prerequisite.

    Return selected source rows keyed by the requested (dataset, sample_id),
    plus producer/sampling provenance. No files are created by validation.
    """
    from rl_sr.schema import adapter_content_sha256

    source_dir = Path(source_dir).expanduser().resolve()
    if not (source_dir / "sample_lineage.jsonl").is_file():
        nested = source_dir / "01_f0_round1_train_state"
        if nested.is_dir():
            source_dir = nested
    run_manifest = _read_json(source_dir / "iterative_manifest.json")
    round_manifest = _read_json(source_dir / "round_01" / "inference_manifest.json")
    actual_sampling = round_manifest.get("sampling", {})
    for name, expected in sampling.items():
        if name not in actual_sampling or actual_sampling[name] != expected:
            raise ValueError(
                f"F0 cache sampling mismatch: {name}={actual_sampling.get(name)!r}, expected {expected!r}"
            )
    if actual_sampling.get("uses_refinement_adapter") is not False:
        raise ValueError("F0 cache round 1 must have uses_refinement_adapter=false")
    refinement = run_manifest.get("refinement") or {}
    if refinement and int(refinement.get("start_round", 1)) <= 1:
        raise ValueError("Cannot reuse a refiner-produced round as F0 y_1")
    for name in ("iqa_pairing", "suggestion_pairing"):
        if round_manifest.get(name) not in (None, "matched"):
            raise ValueError(f"F0 reuse requires matched conditions; source {name} is shuffled")

    expected_hash = adapter_content_sha256(checkpoint)
    recorded_hash = run_manifest.get("producer_adapter_sha256")
    if recorded_hash:
        hash_basis = "saved_adapter_content_hash"
    else:
        # Legacy runs saved a checkpoint path, not a generation-time fingerprint.
        # Verify that checkpoint's current bytes; never guess from its basename.
        old_checkpoint = round_manifest.get("checkpoint_path") or run_manifest.get("checkpoint_path")
        if not old_checkpoint or not Path(old_checkpoint).exists():
            raise FileNotFoundError(
                "Legacy F0 cache has no saved adapter hash and its checkpoint is unavailable: "
                f"{old_checkpoint}. Restore that checkpoint path before reusing the cache."
            )
        recorded_hash = adapter_content_sha256(old_checkpoint)
        hash_basis = "legacy_checkpoint_current_content"
    if recorded_hash != expected_hash:
        raise ValueError("F0 cache adapter content does not match --checkpoint / F0_CHECKPOINT")

    wanted = {}
    output_keys = set()
    for row in lineage:
        source_key = str(Path(row["source_path"]).expanduser().resolve())
        output_key = (row["dataset"], row["sample_id"])
        if source_key in wanted or output_key in output_keys:
            raise ValueError(f"Duplicate selected input or output name: {source_key}")
        if not Path(source_key).is_file():
            raise FileNotFoundError(f"Selected original LR does not exist: {source_key}")
        wanted[source_key] = row
        output_keys.add(output_key)
    if not wanted:
        raise ValueError("No selected training inputs to reuse")

    metadata = {row["name"]: row for row in round_manifest.get("datasets", [])}
    selected = {}
    seen = set()
    with (source_dir / "sample_lineage.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_row = json.loads(line)
            source_key = str(Path(source_row["source_path"]).expanduser().resolve())
            if source_key not in wanted:
                continue
            if source_key in seen:
                raise ValueError(f"Duplicate original LR in F0 cache lineage: {source_key}")
            seen.add(source_key)
            dataset_metadata = metadata.get(source_row["dataset"], {})
            for name, expected in conditions.items():
                if name not in dataset_metadata or dataset_metadata[name] != expected:
                    raise ValueError(
                        f"F0 cache condition mismatch: {name}={dataset_metadata.get(name)!r}, expected {expected!r}"
                    )
            first_round = [r for r in source_row.get("rounds", []) if r.get("round") == 1]
            if len(first_round) != 1 or not first_round[0].get("path"):
                raise ValueError(f"Missing or ambiguous F0 round-1 output: {source_key}")
            image_path = Path(first_round[0]["path"])
            # Support moving an output tree while retaining its relative layout.
            # Original LR paths must still resolve to the current selected inputs.
            old_root = run_manifest.get("output_dir")
            if old_root:
                try:
                    relative = image_path.resolve().relative_to(Path(old_root).resolve())
                except ValueError:
                    pass
                else:
                    image_path = source_dir / relative
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing cached y_1 for {source_key}: {image_path}")
            with Image.open(image_path) as image:
                if image.format != "PNG":
                    raise ValueError(f"Cached y_1 must be a PNG: {image_path}")
                image.verify()
            target = wanted[source_key]
            selected[(target["dataset"], target["sample_id"])] = {
                "path": image_path.resolve(),
                "metadata": dataset_metadata,
            }
    missing = set(wanted) - seen
    if missing:
        raise ValueError(f"F0 cache lacks {len(missing)} selected inputs; first: {sorted(missing)[0]}")
    return {
        "selected": selected,
        "round_manifest": round_manifest,
        "provenance": {
            "source_dir": str(source_dir),
            "source_status": run_manifest.get("status"),
            "sample_count": len(selected),
            "producer_adapter_sha256": expected_hash,
            "adapter_verification": hash_basis,
            "sampling": actual_sampling,
            "copy_mode": "copy",
            "metrics_reused": False,
        },
    }


def copy_cached_round(cache, lineage, output_dirs):
    """Materialize just the selected y_1 images into the usual round layout."""
    destinations = [output_dirs[row["dataset"]] / f"{row['sample_id']}.png" for row in lineage]
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite cached round destination: {path}")
    counts = Counter(row["dataset"] for row in lineage)
    dataset_metadata = {}
    for row, destination in zip(lineage, destinations):
        entry = cache["selected"][(row["dataset"], row["sample_id"])]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["path"], destination)
        # Preserve condition semantics, but do not copy old full-dataset counts
        # or pairing CSV paths into a subset run.
        dataset_metadata[row["dataset"]] = {
            key: entry["metadata"][key]
            for key in ("prompt_variant", "include_caption", "pre_cropped_input", "restore_input_size")
            if key in entry["metadata"]
        }
        dataset_metadata[row["dataset"]].update({
            "valid_image_count": counts[row["dataset"]],
            "skipped_image_count": 0,
            "reused_f0": True,
            "reuse_source_dir": cache["provenance"]["source_dir"],
        })
    return dataset_metadata


def reuse_f0_outputs(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"F0 reuse destination must be empty: {output_dir}")
    inputs = [Path(line.strip()).expanduser().resolve()
              for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    lineage = [{"dataset": "default", "sample_id": path.stem, "source_path": str(path),
                "source_input_root": str(Path(args.input).resolve()), "rounds": []} for path in inputs]
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    flow = config.get("flow_matching", {})
    condition = config.get("condition", {})
    cache = validate_f0_cache(args.source_dir, lineage, args.checkpoint, {
        "num_inference_steps": 25,
        "schedule": flow.get("inference_schedule", "linear"),
        "init_mode": flow.get("inference_init_mode", "pure_noise"),
        "sigma_start": float(flow.get("inference_sigma_start", 1.0)),
        "seed": args.seed, "input_upscale": 1, "uses_refinement_adapter": False,
    }, {
        "prompt_variant": condition.get("prompt_variant"),
        "include_caption": bool(condition.get("include_caption", False)),
        "pre_cropped_input": bool(config.get("data", {}).get("pre_cropped", True)),
        "restore_input_size": False,
    })
    round_dir = output_dir / "round_01"
    dirs = {"default": round_dir / "default"}
    metadata = copy_cached_round(cache, lineage, dirs)
    for row in lineage:
        row["rounds"] = [{"round": 1, "path": str(dirs["default"] / f"{row['sample_id']}.png"), "exists": True}]
    (output_dir / "sample_lineage.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lineage), encoding="utf-8"
    )
    round_manifest = copy.deepcopy(cache["round_manifest"])
    round_manifest.update({
        "run_dir": None, "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "output_dir": str(round_dir),
        "datasets": [{"name": "default", "input_path": str(Path(args.input).resolve()),
                      "output_dir": str(dirs["default"]), **metadata["default"]}],
    })
    _write_json(round_dir / "inference_manifest.json", round_manifest)
    _write_json(output_dir / "f0_reuse_manifest.json", cache["provenance"])
    _write_json(output_dir / "iterative_manifest.json", {
        "status": "completed", "output_dir": str(output_dir), "iterations": 1,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "producer_adapter_sha256": cache["provenance"]["producer_adapter_sha256"],
        "sample_lineage": str(output_dir / "sample_lineage.jsonl"),
        "base_seed": args.seed, "seed_policy": "fixed",
        "metrics": [], "metrics_status": "skipped_for_f0_reuse",
        "f0_reuse": cache["provenance"],
        "rounds": [{"round": 1, "output_dirs": {"default": str(dirs["default"])},
                    "inference_manifest": str(round_dir / "inference_manifest.json"),
                    "missing_output_count": 0, "reused_f0": True}],
    })
    return cache["provenance"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True, help="Old 01 directory or its parent run directory")
    parser.add_argument("--input", required=True, help="Selected LQ paths TXT from create_sr_input_manifest.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    print(json.dumps(reuse_f0_outputs(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
