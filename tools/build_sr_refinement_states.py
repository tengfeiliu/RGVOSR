"""Build portable (x, y_k) refinement states from iterative-inference lineage.

Example (the optional roots only supply human-readable portable keys; they do
not enter hashes).  When the roots are omitted, content-addressed relative keys
are generated automatically, so a run can move between servers unchanged:

python tools/build_sr_refinement_states.py \
  --lineage_jsonl runs/iterative/sample_lineage.jsonl \
  --source_jsonl datasets/LSDIR_precrop512/train.iqa_caption_suggestion.jsonl \
  --dataset_id lsdir_precrop512_v1 --artifact_root runs/iterative \
  --producer_adapter artifacts/f0/rg_flux_adapters --output_jsonl artifacts/states.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.prompt_builder import build_sr_prompt
from models.router_condition import ROUTER_CONDITION_VERSION, extract_router_condition
from rl_sr.artifact_store import read_jsonl, write_jsonl_atomic
from rl_sr.schema import (
    RuntimePaths,
    StateRecord,
    build_sample_key,
    canonical_json,
    content_sha256,
    adapter_content_sha256,
    identity_sha256,
    normalize_relative_key,
)


def _absolute(path):
    return Path(path).expanduser().resolve()


def _portable_key(path, roots, namespace, content_hash=None):
    """Return a portable key without ever placing an absolute path in identity.

    A configured dataset/artifact root gives the most legible key.  The one-click
    runner intentionally has no dataset-root argument; in that case, use the
    file content hash as a synthetic *relative* key.  It remains stable after a
    dataset is mounted at a different server path.
    """
    path = _absolute(path)
    for root_namespace, root in roots:
        try:
            return normalize_relative_key(Path(root_namespace) / path.relative_to(root))
        except ValueError:
            continue
    digest = content_hash or content_sha256(path)
    return normalize_relative_key(Path(namespace) / "sha256" / digest)


def _geometry_sha256(path):
    with Image.open(path) as image:
        image.load()
        payload = {"mode": image.mode, "width": image.width, "height": image.height}
    return identity_sha256(payload)


def _index_source_rows(source_jsonl):
    index = {}
    for row in read_jsonl(source_jsonl):
        lq_path = row.get("lq_path")
        if not lq_path:
            continue
        resolved = str(_absolute(lq_path))
        if resolved in index:
            raise ValueError(f"Duplicate lq_path in source JSONL: {lq_path}")
        index[resolved] = row
    return index


def _prompt_for_row(row, args):
    raw = row.get("unipercept_raw") if isinstance(row.get("unipercept_raw"), dict) else {}
    profile = raw.get("profile") if isinstance(raw.get("profile"), dict) else {}
    return build_sr_prompt(
        profile,
        use_prompt=not args.no_prompt,
        use_suggestions=not args.no_suggestions,
        prompt_variant=args.prompt_variant,
        include_caption=args.include_caption,
    )


def _round_producer_adapters(values):
    mapping = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(
                "--round_producer_adapter must use GENERATED_ROUND=ADAPTER_PATH, for example 1=/models/f0"
            )
        raw_round, raw_path = item.split("=", 1)
        try:
            round_number = int(raw_round)
        except ValueError as exc:
            raise ValueError(f"Invalid generated round in --round_producer_adapter: {item}") from exc
        if round_number <= 0 or not raw_path.strip():
            raise ValueError(f"Invalid --round_producer_adapter: {item}")
        if round_number in mapping:
            raise ValueError(f"Duplicate producer adapter for generated round {round_number}")
        mapping[round_number] = raw_path.strip()
    return mapping


def build_states(args):
    source_index = _index_source_rows(args.source_jsonl)
    lr_root = _absolute(args.lr_root) if args.lr_root else None
    hr_root = _absolute(args.hr_root) if args.hr_root else None
    artifact_root = _absolute(args.artifact_root)
    roots = tuple(
        (namespace, root)
        for namespace, root in (("lr", lr_root), ("hr", hr_root), ("artifact", artifact_root))
        if root is not None
    )
    producer_paths = _round_producer_adapters(args.round_producer_adapter)
    producer_hashes = {"default": adapter_content_sha256(args.producer_adapter)}
    for generated_round, adapter_path in producer_paths.items():
        producer_hashes[generated_round] = adapter_content_sha256(adapter_path)
    sampler_payload = json.loads(args.sampler_json) if args.sampler_json else {
        "num_steps": args.num_inference_steps,
        "schedule": args.inference_schedule,
        "init_mode": args.inference_init_mode,
        "sigma_start": args.inference_sigma_start,
    }
    sampler_hash = identity_sha256(sampler_payload)
    records = []
    for lineage in read_jsonl(args.lineage_jsonl):
        source_path = lineage.get("source_path")
        if not source_path:
            raise ValueError(f"Lineage row lacks source_path: {lineage}")
        source_path = _absolute(source_path)
        source = source_index.get(str(source_path))
        if source is None:
            raise ValueError(
                f"No paired source JSONL record matches lineage source {source_path}. "
                "Use the same pre-cropped manifest that created the iterative run."
            )
        hr_path = source.get("hq_path")
        if not hr_path or not Path(hr_path).is_file():
            raise FileNotFoundError(f"Missing paired HR image for {source_path}: {hr_path}")
        rounds = lineage.get("rounds") or []
        original_hash = content_sha256(source_path)
        original_lr_key = _portable_key(source_path, roots, "original_lr", original_hash)
        sample_key = build_sample_key(args.dataset_id, original_lr_key, original_hash)
        raw_profile = source.get("unipercept_raw") if isinstance(source.get("unipercept_raw"), dict) else {}
        profile = raw_profile.get("profile") if isinstance(raw_profile.get("profile"), dict) else {}
        router_condition = extract_router_condition(profile, version=args.router_condition_version)
        for position, round_output in enumerate(rounds):
            target_round = position + 2
            if target_round > args.max_round:
                break
            current_path = round_output.get("path")
            if not round_output.get("exists") or not current_path or not Path(current_path).is_file():
                continue
            current_path = _absolute(current_path)
            generated_round = int(round_output.get("round", position + 1))
            current_producer_hash = producer_hashes.get(
                generated_round, producer_hashes["default"]
            )
            if target_round == 2:
                parent_state_id = None
            else:
                previous_path = rounds[position - 1].get("path")
                previous_id = next(
                    (item.state_id for item in records if item.runtime.current_sr == str(_absolute(previous_path))),
                    None,
                )
                if previous_id is None:
                    raise ValueError(
                        f"Cannot build round-{target_round} state for {source_path}: "
                        "the previous generated round is absent from the lineage."
                    )
                parent_state_id = previous_id
            current_hash = content_sha256(current_path)
            hr_hash = content_sha256(hr_path)
            records.append(
                StateRecord(
                    dataset_id=args.dataset_id,
                    sample_key=sample_key,
                    original_lr_key=original_lr_key,
                    original_lr_sha256=original_hash,
                    current_output_key=_portable_key(current_path, roots, "generated_sr", current_hash),
                    current_output_sha256=current_hash,
                    round_index=target_round,
                    parent_state_id=parent_state_id,
                    geometry_sha256=_geometry_sha256(current_path),
                    producer_adapter_sha256=current_producer_hash,
                    sampler_sha256=sampler_hash,
                    hr_key=_portable_key(hr_path, roots, "paired_hr", hr_hash),
                    hr_sha256=hr_hash,
                    prompt=_prompt_for_row(source, args),
                    router_condition_sha256=router_condition.source_hash,
                    metadata={
                        "lineage_dataset": str(lineage.get("dataset") or ""),
                        "source_sample_id": str(source.get("sample_id") or lineage.get("sample_id") or ""),
                        "router_condition": router_condition.as_dict(),
                    },
                    runtime=RuntimePaths(
                        original_lr=str(source_path), current_sr=str(current_path), hr=str(_absolute(hr_path))
                    ),
                )
            )
    if not records:
        raise RuntimeError("No usable refinement states were built; check lineage output files and --max_round.")
    write_jsonl_atomic(args.output_jsonl, records)
    manifest = {
        "schema_version": "rl_sr_v1",
        "dataset_id": args.dataset_id,
        "state_count": len(records),
        "round_counts": {str(r): sum(item.round_index == r for item in records) for r in range(2, args.max_round + 1)},
        "producer_adapter_sha256": producer_hashes["default"],
        "producer_adapter_sha256_by_generated_round": {
            str(round_number): value
            for round_number, value in producer_hashes.items()
            if round_number != "default"
        },
        "sampler_sha256": sampler_hash,
        "sampler": sampler_payload,
        # This is documentation only. It is intentionally excluded from all IDs.
        "runtime_roots": {
            "lr_root": str(lr_root) if lr_root else None,
            "hr_root": str(hr_root) if hr_root else None,
            "artifact_root": str(artifact_root),
        },
    }
    Path(args.output_jsonl).with_suffix(".manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage_jsonl", required=True)
    parser.add_argument("--source_jsonl", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument(
        "--lr_root",
        default=None,
        help="Optional dataset mount used only for readable relative keys; omit for portable content-addressed keys.",
    )
    parser.add_argument(
        "--hr_root",
        default=None,
        help="Optional dataset mount used only for readable relative keys; omit for portable content-addressed keys.",
    )
    parser.add_argument("--artifact_root", required=True)
    parser.add_argument("--producer_adapter", required=True)
    parser.add_argument(
        "--round_producer_adapter",
        action="append",
        default=[],
        help="Override the producer of a generated round: ROUND=ADAPTER_PATH. Can be repeated.",
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_round", type=int, default=4)
    parser.add_argument("--sampler_json", default=None, help="Canonical sampler JSON; overrides individual sampler flags.")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--inference_schedule", default="linear")
    parser.add_argument("--inference_init_mode", default="pure_noise")
    parser.add_argument("--inference_sigma_start", type=float, default=1.0)
    parser.add_argument("--prompt_variant", default="iqa_suggestion")
    parser.add_argument("--include_caption", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_prompt", action="store_true")
    parser.add_argument("--no_suggestions", action="store_true")
    parser.add_argument("--router_condition_version", default=ROUTER_CONDITION_VERSION)
    return parser.parse_args()


if __name__ == "__main__":
    summary = build_states(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
