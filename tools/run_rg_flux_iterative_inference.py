"""Run RG-FLUX-SR repeatedly while preserving the existing one-pass CLI.

Round 1 uses the requested SR upscale factor. Every later round feeds the
previous SR image back into the model at its current resolution (upscale=1).
The model is loaded once, each round is written to a separate directory, and
the configured no-reference metrics are evaluated after every round.
"""

import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_rg_flux_sr import (
    build_arg_parser as build_single_pass_arg_parser,
    cfg,
    condition_for_image,
    image_lookup_aliases,
    list_images,
    load_config,
    load_jsonl_conditions,
    normalize_iqa_pairing,
    normalize_suggestion_pairing,
    parse_dataset_dirs,
    resolve_inference_dtype,
    resolve_inference_run,
    run_inference_dataset,
    write_inference_manifest,
)
from metrics.rg_sr_metrics import DEFAULT_OMGSR_METRICS, evaluate_dataset_dirs
from models.rg_flux_artist_factory import build_rg_flux_artist
from models.text_embedding_cache import get_text_embedding_cache


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def _write_csv_atomic(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _resolve_iterative_run(args):
    resolved = resolve_inference_run(args)
    if args.run_dir and not args.output_dir and not args.output_root:
        resolved["output_dir"] = (
            Path(args.run_dir)
            / "iterative_inference"
            / str(resolved["checkpoint_step"])
        )
    return resolved


def _resolve_source_datasets(args):
    if args.dataset_dirs:
        datasets = parse_dataset_dirs(args.dataset_dirs)
    elif args.input:
        datasets = [("default", Path(args.input))]
    else:
        raise ValueError("Either --input or --dataset_dirs is required.")

    names = [name for name, _ in datasets]
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must be unique for iterative inference.")
    return datasets


def _build_lineage(source_datasets):
    lineage = []
    for dataset_name, input_path in source_datasets:
        image_paths = list_images(input_path)
        if not image_paths:
            raise FileNotFoundError(
                f"No input images found for dataset '{dataset_name}': {input_path}"
            )
        stems = [path.stem for path in image_paths]
        duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
        if duplicate_stems:
            preview = ", ".join(duplicate_stems[:3])
            raise ValueError(
                f"Dataset '{dataset_name}' contains duplicate output stems ({preview}); "
                "PNG round outputs would overwrite each other."
            )
        for image_path in image_paths:
            lineage.append(
                {
                    "dataset": dataset_name,
                    "sample_id": image_path.stem,
                    "source_path": str(image_path),
                    "source_input_root": str(input_path),
                    "rounds": [],
                }
            )
    return lineage


def _write_lineage(path, lineage):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in lineage:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def _read_lineage(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _lineage_identity(row):
    return (
        str(row.get("dataset") or ""),
        str(row.get("sample_id") or ""),
        str(Path(row.get("source_path") or "").expanduser().resolve()),
    )


def _completed_generated_rounds(lineage, iterations):
    """Return consecutive rounds whose images exist for every lineage row."""
    completed = []
    for round_number in range(1, iterations + 1):
        complete = True
        for row in lineage:
            matches = [item for item in row.get("rounds", []) if int(item.get("round", -1)) == round_number]
            if len(matches) != 1 or not matches[0].get("exists") or not Path(matches[0].get("path", "")).is_file():
                complete = False
                break
        if not complete:
            break
        completed.append(round_number)
    return completed


def _record_round_outputs(lineage, round_number, output_dirs):
    for row in lineage:
        output_path = output_dirs[row["dataset"]] / f"{row['sample_id']}.png"
        row["rounds"].append(
            {
                "round": round_number,
                "path": str(output_path),
                "exists": output_path.is_file(),
            }
        )


def _build_round_condition_index(base_index, lineage, current_inputs):
    """Map generated B/C/... paths back to A's original JSONL condition."""
    if not base_index:
        return {}
    inherited_index = dict(base_index)
    for row in lineage:
        dataset_name = row["dataset"]
        source_path = Path(row["source_path"])
        condition = condition_for_image(
            base_index,
            source_path,
            dataset_name=dataset_name,
            input_root=Path(row["source_input_root"]),
        )
        if condition is None:
            continue
        current_root = Path(current_inputs[dataset_name])
        if row["rounds"]:
            previous = row["rounds"][-1]
            if not previous["exists"]:
                continue
            current_path = Path(previous["path"])
        else:
            current_path = source_path
        for alias in image_lookup_aliases(
            current_path,
            dataset_name=dataset_name,
            input_root=current_root,
        ):
            inherited_index[alias] = condition
    return inherited_index


def _round_seed(base_seed, round_number, mode):
    if mode == "fixed":
        return int(base_seed)
    if mode == "increment":
        return int(base_seed) + int(round_number) - 1
    raise ValueError(f"Unsupported round seed mode: {mode}")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_runtime_config(args, config):
    config.setdefault("data", {})
    config.setdefault("condition", {})
    config.setdefault("text_encoding", {})

    args.inference_schedule = args.inference_schedule or cfg(
        config, "flow_matching.inference_schedule", "linear"
    )
    args.inference_init_mode = args.inference_init_mode or cfg(
        config, "flow_matching.inference_init_mode", "pure_noise"
    )
    if args.inference_sigma_start is None:
        args.inference_sigma_start = float(
            cfg(config, "flow_matching.inference_sigma_start", 1.0)
        )
    if args.inference_init_mode == "pure_noise" and args.inference_sigma_start != 1.0:
        raise ValueError("--inference_init_mode pure_noise requires --inference_sigma_start 1.0.")

    if args.full_frame_inference:
        config["data"]["pre_cropped"] = False
    config["condition"]["lr_cond_mode"] = args.lr_cond_mode or cfg(
        config, "condition.lr_cond_mode", "latent_adapter"
    )
    config["condition"]["use_prompt"] = args.use_prompt
    if args.use_degradation_vector is None:
        args.use_degradation_vector = bool(
            cfg(config, "condition.use_degradation_vector", True)
        )
    config["condition"]["use_degradation_vector"] = args.use_degradation_vector
    config["condition"]["use_suggestions"] = args.use_suggestions

    router_input_mode = str(cfg(config, "model.lora_moe.router_input_mode", "prompt_lr"))
    if router_input_mode in {"condition8", "condition8_timestep"} and args.use_degradation_vector:
        raise ValueError(
            "condition8 inference requires --no-use_degradation_vector; the legacy "
            "degradation_vector is invalid and is not used as a fallback."
        )

    if args.prompt_variant is None:
        args.prompt_variant = cfg(config, "condition.prompt_variant", None)
    else:
        config["condition"]["prompt_variant"] = args.prompt_variant
    if args.include_caption is None:
        args.include_caption = bool(cfg(config, "condition.include_caption", False))
    else:
        config["condition"]["include_caption"] = bool(args.include_caption)
    if args.prompt_variant == "fixed" and args.include_caption:
        raise ValueError("prompt_variant=fixed cannot be combined with include_caption=true")
    if args.include_caption and not args.jsonl_path:
        raise ValueError(
            "Caption-conditioned inference requires --jsonl_path with a crop-local "
            "caption for every input LQ image."
        )

    if args.text_encoding_mode is not None:
        config["text_encoding"]["mode"] = args.text_encoding_mode
    if args.text_embedding_cache is not None:
        config["text_encoding"]["cache_dir"] = args.text_embedding_cache

    args.suggestion_pairing = normalize_suggestion_pairing(args.suggestion_pairing)
    if args.iqa_pairing is not None:
        args.iqa_pairing = normalize_iqa_pairing(args.iqa_pairing)
    if args.suggestion_pairing == "shuffled" or args.iqa_pairing == "shuffled":
        text_mode = str(cfg(config, "text_encoding.mode", "online") or "online").strip().lower()
        if text_mode != "online":
            raise ValueError(
                "Shuffled prompt conditions require online text encoding. "
                "Pass --text_encoding_mode online."
            )
    return config


def _metric_trend_rows(round_number, summary, round_output_dir):
    directions = summary.get("metric_directions", {})
    rows = []
    for row in summary.get("summary", []):
        rows.append(
            {
                "round": round_number,
                "dataset": row["dataset"],
                "metric": row["metric"],
                "direction": directions.get(row["metric"], ""),
                "mean": row["mean"],
                "std": row["std"],
                "count": row["count"],
                "output_dir": str(round_output_dir / row["dataset"]),
            }
        )
    return rows


def _validate_output_root(output_root):
    output_root = Path(output_root)
    existing_entries = list(output_root.iterdir()) if output_root.exists() else []
    if existing_entries:
        raise FileExistsError(
            f"Iterative output directory is not empty: {output_root}. "
            "Choose a new --output_dir/--output_root to avoid overwriting prior results."
        )


def run_iterative_inference(args):
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero.")
    if args.upscale <= 0:
        raise ValueError("--upscale must be greater than zero.")
    if args.metric_sample_count < -1:
        raise ValueError("--metric_sample_count must be -1 (all), 0 (skip), or a positive count.")

    resolved_run = _resolve_iterative_run(args)
    output_root = Path(resolved_run["output_dir"])
    if args.resume:
        if not output_root.is_dir():
            raise FileNotFoundError(f"Cannot resume missing iterative output directory: {output_root}")
    else:
        _validate_output_root(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    config = _prepare_runtime_config(
        args,
        load_config(resolved_run["checkpoint"], args.config),
    )
    if bool(cfg(config, "data.pre_cropped", True)) and args.upscale != 1:
        raise ValueError(
            "The loaded config uses data.pre_cropped=true, so the existing inference "
            "semantics ignore --upscale. Pass --full_frame_inference for round-1 SR, "
            "or use --upscale 1 when inputs are already at the model target resolution."
        )
    metrics = list(
        args.metrics
        or cfg(config, "evaluation.metrics", DEFAULT_OMGSR_METRICS)
        or DEFAULT_OMGSR_METRICS
    )
    if not metrics:
        raise ValueError("At least one metric is required for iterative inference.")

    dtype, dtype_name = resolve_inference_dtype(config, args.dtype)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source_datasets = _resolve_source_datasets(args)
    requested_lineage = _build_lineage(source_datasets)
    lineage = requested_lineage
    existing_manifest = None
    start_round = 1
    if args.resume:
        existing_manifest = json.loads(
            (output_root / "iterative_manifest.json").read_text(encoding="utf-8")
        )
        lineage = _read_lineage(output_root / "sample_lineage.jsonl")
        if {_lineage_identity(row) for row in lineage} != {
            _lineage_identity(row) for row in requested_lineage
        }:
            raise ValueError("Resume input set does not match the existing iterative lineage.")
        if int(existing_manifest.get("iterations", -1)) != int(args.iterations):
            raise ValueError(
                f"Resume iterations mismatch: existing={existing_manifest.get('iterations')}, requested={args.iterations}"
            )
        if int(existing_manifest.get("base_seed", -1)) != int(args.seed):
            raise ValueError(
                f"Resume seed mismatch: existing={existing_manifest.get('base_seed')}, requested={args.seed}"
            )
        completed_generated = _completed_generated_rounds(lineage, args.iterations)
        start_round = len(completed_generated) + 1
        for row in lineage:
            unexpected = [
                int(item.get("round", -1))
                for item in row.get("rounds", [])
                if int(item.get("round", -1)) >= start_round
            ]
            if unexpected:
                raise ValueError(
                    "Resume found a non-consecutive or incomplete round in sample_lineage.jsonl; "
                    f"first pending round is {start_round}, unexpected entries={unexpected[:3]}."
                )
        for round_number in range(start_round, args.iterations + 1):
            pending_dir = output_root / f"round_{round_number:02d}"
            if pending_dir.exists() and any(pending_dir.iterdir()):
                raise FileExistsError(
                    "Resume refuses to overwrite a partially generated round directory that is not "
                    f"complete in lineage: {pending_dir}. Preserve it for inspection and use a clean recovery copy."
                )
    f0_cache = None
    if args.reuse_f0_dir and not args.resume:
        from tools.reuse_sr_f0_outputs import validate_f0_cache

        if args.refiner_checkpoint and args.refiner_start_round <= 1:
            raise ValueError("--reuse_f0_dir requires F0, not the refiner, in round 1")
        if args.suggestion_pairing != "matched" or args.iqa_pairing not in (None, "matched"):
            raise ValueError("--reuse_f0_dir requires matched prompt conditions")
        f0_cache = validate_f0_cache(
            args.reuse_f0_dir, lineage, resolved_run["checkpoint"],
            {
                "num_inference_steps": args.num_inference_steps,
                "schedule": args.inference_schedule,
                "init_mode": args.inference_init_mode,
                "sigma_start": args.inference_sigma_start,
                "seed": args.seed,
                "input_upscale": args.upscale,
                "uses_refinement_adapter": False,
            },
            {
                "prompt_variant": args.prompt_variant,
                "include_caption": args.include_caption,
                "pre_cropped_input": bool(cfg(config, "data.pre_cropped", True)),
                "restore_input_size": bool(args.restore_input_size),
            },
        )
    lineage_path = output_root / "sample_lineage.jsonl"
    _write_lineage(lineage_path, lineage)

    manifest_path = output_root / "iterative_manifest.json"
    trend_path = output_root / "metric_trends.csv"
    manifest = existing_manifest or {
        "status": "running",
        "run_dir": str(resolved_run["run_dir"]) if resolved_run["run_dir"] else None,
        "checkpoint_step": resolved_run["checkpoint_step"],
        "checkpoint_path": str(resolved_run["checkpoint"]),
        "output_dir": str(output_root),
        "iterations": int(args.iterations),
        "upscale_policy": {
            "round_1": int(args.upscale),
            "later_rounds": 1,
            "description": "first-round SR followed by fixed-resolution iterative enhancement",
        },
        "seed_policy": args.round_seed_mode,
        "base_seed": int(args.seed),
        "metrics": metrics,
        "metric_device": str(args.metric_device),
        "sample_lineage": str(lineage_path),
        "metric_trends": str(trend_path),
        "rounds": [],
    }
    manifest["status"] = "running"
    manifest.pop("error", None)
    manifest["metrics"] = metrics
    manifest["metric_device"] = str(args.metric_device)
    manifest["metric_sample_count"] = int(args.metric_sample_count)
    manifest["metric_sample_seed"] = int(args.metric_sample_seed)
    if args.resume:
        manifest["resume"] = {
            "first_round_to_generate": int(start_round),
            "completed_generated_rounds": list(range(1, start_round)),
            "skip_existing_images": True,
        }
        recorded_rounds = {int(item["round"]) for item in manifest.get("rounds", [])}
        for round_number in range(1, start_round):
            if round_number in recorded_rounds:
                continue
            round_output_dir = output_root / f"round_{round_number:02d}"
            summary_path = round_output_dir / "metrics" / "summary_scores.json"
            manifest.setdefault("rounds", []).append(
                {
                    "round": round_number,
                    "input_upscale": int(args.upscale) if round_number == 1 else 1,
                    "uses_refinement_adapter": bool(
                        args.refiner_checkpoint and round_number >= int(args.refiner_start_round)
                    ),
                    "seed": _round_seed(args.seed, round_number, args.round_seed_mode),
                    "input_dirs": {},
                    "output_dirs": {
                        name: str(round_output_dir / name) for name, _ in source_datasets
                    },
                    "inference_manifest": str(round_output_dir / "inference_manifest.json"),
                    "metrics_dir": str(round_output_dir / "metrics") if summary_path.is_file() else None,
                    "metric_summary": str(summary_path) if summary_path.is_file() else None,
                    "metric_status": "completed_before_resume" if summary_path.is_file() else "skipped_on_resume",
                    "missing_output_count": 0,
                    "recovered_from_lineage": True,
                }
            )
        manifest["rounds"] = sorted(manifest.get("rounds", []), key=lambda item: int(item["round"]))
    if f0_cache is not None:
        manifest["f0_reuse"] = f0_cache["provenance"]
        _write_json_atomic(output_root / "f0_reuse_manifest.json", f0_cache["provenance"])
    if args.refiner_checkpoint:
        if not bool(cfg(config, "condition.refinement.enabled", False)):
            raise ValueError(
                "--refiner_checkpoint requires condition.refinement.enabled: true in --config. "
                "Use configs/rl_sr_refinement_flux2_klein_moe.yaml after setting its runtime paths."
            )
        manifest["refinement"] = {
            "checkpoint": str(Path(args.refiner_checkpoint)),
            "start_round": int(args.refiner_start_round),
            "state": "current SR y_k plus original LR anchor x",
        }
    _write_json_atomic(manifest_path, manifest)

    if start_round > args.iterations:
        manifest["status"] = "completed"
        manifest["resume"]["all_requested_rounds_already_generated"] = True
        _write_json_atomic(manifest_path, manifest)
        return manifest

    artist = None
    trend_rows = []
    if trend_path.is_file():
        with trend_path.open("r", encoding="utf-8", newline="") as handle:
            trend_rows = list(csv.DictReader(handle))
    try:
        artist = build_rg_flux_artist(config).to(device=device)
        artist.load_trainable(resolved_run["checkpoint"], is_trainable=False)
        if hasattr(artist, "align_inference_dtype"):
            artist.align_inference_dtype(dtype=dtype)
        artist.eval()
        text_embedding_cache = get_text_embedding_cache(
            config,
            dtype=cfg(config, "text_encoding.dtype", dtype_name),
        )
        moe_inference_schedule = None
        if hasattr(artist, "set_moe_inference_schedule"):
            moe_inference_schedule = artist.set_moe_inference_schedule()

        base_condition_index = load_jsonl_conditions(args.jsonl_path)
        current_inputs = {name: Path(path) for name, path in source_datasets}
        if start_round > 1:
            current_inputs = {
                dataset_name: output_root / f"round_{start_round - 1:02d}" / dataset_name
                for dataset_name, _ in source_datasets
            }
        lr_cond_mode = config["condition"]["lr_cond_mode"]
        refiner_loaded = False

        def original_anchor_for(image_path, dataset_name, _condition):
            matches = [
                row for row in lineage
                if row["dataset"] == dataset_name and row["sample_id"] == Path(image_path).stem
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot resolve unique original LR anchor for dataset={dataset_name}, image={image_path}"
                )
            return Path(matches[0]["source_path"])

        for round_number in range(start_round, args.iterations + 1):
            round_name = f"round_{round_number:02d}"
            round_output_dir = output_root / round_name
            output_dirs = {
                dataset_name: round_output_dir / dataset_name
                for dataset_name in current_inputs
            }
            round_seed = _round_seed(args.seed, round_number, args.round_seed_mode)
            _set_seed(round_seed)
            round_args = copy.copy(args)
            round_args.upscale = int(args.upscale) if round_number == 1 else 1
            use_refinement = bool(
                args.refiner_checkpoint and round_number >= int(args.refiner_start_round)
            )
            if use_refinement and not refiner_loaded:
                artist.load_trainable(args.refiner_checkpoint, is_trainable=False)
                if hasattr(artist, "align_inference_dtype"):
                    artist.align_inference_dtype(dtype=dtype)
                artist.eval()
                refiner_loaded = True
            condition_index = _build_round_condition_index(
                base_condition_index,
                lineage,
                current_inputs,
            )

            dataset_metadata = {}
            round_datasets = []
            reuse_round = round_number == 1 and f0_cache is not None
            if reuse_round:
                from tools.reuse_sr_f0_outputs import copy_cached_round

                dataset_metadata = copy_cached_round(f0_cache, lineage, output_dirs)
            for dataset_name, input_path in current_inputs.items():
                dataset_output_dir = output_dirs[dataset_name]
                if not reuse_round:
                    dataset_metadata[dataset_name] = run_inference_dataset(
                        dataset_name=dataset_name,
                        input_path=input_path,
                        output_dir=dataset_output_dir,
                        artist=artist,
                        config=config,
                        args=round_args,
                        condition_index=condition_index,
                        text_embedding_cache=text_embedding_cache,
                        device=device,
                        dtype=dtype,
                        lr_cond_mode=lr_cond_mode,
                        anchor_path_resolver=original_anchor_for if use_refinement else None,
                        refinement_round=round_number if use_refinement else None,
                    )
                round_datasets.append((dataset_name, input_path, dataset_output_dir))

            _record_round_outputs(lineage, round_number, output_dirs)
            _write_lineage(lineage_path, lineage)
            missing_outputs = sum(
                not row["rounds"][-1]["exists"] for row in lineage
            )

            round_manifest_path = round_output_dir / "inference_manifest.json"
            write_inference_manifest(
                manifest_path=round_manifest_path,
                run_dir=resolved_run["run_dir"],
                checkpoint_step=resolved_run["checkpoint_step"],
                checkpoint_path=resolved_run["checkpoint"],
                output_dir=round_output_dir,
                datasets=round_datasets,
                suggestion_pairing=(
                    None if round_args.iqa_pairing is not None else round_args.suggestion_pairing
                ),
                suggestion_shuffle_seed=(
                    round_args.suggestion_shuffle_seed
                    if round_args.suggestion_pairing == "shuffled"
                    else None
                ),
                iqa_pairing=round_args.iqa_pairing,
                iqa_shuffle_seed=(
                    round_args.iqa_shuffle_seed
                    if round_args.iqa_pairing == "shuffled"
                    else None
                ),
                dataset_metadata=dataset_metadata,
                moe_routing=moe_inference_schedule,
                sampling={
                    "num_inference_steps": round_args.num_inference_steps,
                    "schedule": round_args.inference_schedule,
                    "init_mode": round_args.inference_init_mode,
                    "sigma_start": round_args.inference_sigma_start,
                    "seed": round_seed,
                    "input_upscale": round_args.upscale,
                    "uses_refinement_adapter": use_refinement,
                },
            )

            metrics_dir = round_output_dir / "metrics"
            metric_summary_path = None
            metric_status = "skipped" if args.metric_sample_count == 0 else "completed"
            if args.metric_sample_count != 0:
                metric_kwargs = {}
                if args.metric_sample_count > 0:
                    metric_kwargs = {
                        "max_samples_per_dataset": args.metric_sample_count,
                        "sample_seed": args.metric_sample_seed,
                    }
                metric_summary = evaluate_dataset_dirs(
                    dataset_dirs=output_dirs,
                    output_dir=metrics_dir,
                    metrics=metrics,
                    device=args.metric_device,
                    **metric_kwargs,
                )
                trend_rows.extend(
                    _metric_trend_rows(round_number, metric_summary, round_output_dir)
                )
                _write_csv_atomic(
                    trend_path,
                    trend_rows,
                    [
                        "round",
                        "dataset",
                        "metric",
                        "direction",
                        "mean",
                        "std",
                        "count",
                        "output_dir",
                    ],
                )
                metric_summary_path = str(metrics_dir / "summary_scores.json")

            manifest["rounds"].append(
                {
                    "round": round_number,
                    "input_upscale": round_args.upscale,
                    "uses_refinement_adapter": use_refinement,
                    "seed": round_seed,
                    "input_dirs": {
                        name: str(path) for name, path in current_inputs.items()
                    },
                    "output_dirs": {
                        name: str(path) for name, path in output_dirs.items()
                    },
                    "inference_manifest": str(round_manifest_path),
                    "metrics_dir": str(metrics_dir) if args.metric_sample_count != 0 else None,
                    "metric_summary": metric_summary_path,
                    "metric_status": metric_status,
                    "metric_sample_count": int(args.metric_sample_count),
                    "missing_output_count": missing_outputs,
                    "reused_f0": reuse_round,
                }
            )
            _write_json_atomic(manifest_path, manifest)
            current_inputs = output_dirs

        manifest["status"] = "completed"
        _write_json_atomic(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(manifest_path, manifest)
        raise


def build_arg_parser():
    parser = build_single_pass_arg_parser()
    parser.description = (
        "Run iterative RG-FLUX-SR: round 1 performs SR, later rounds enhance "
        "the previous output at a fixed resolution, with metrics after every round."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Total number of complete inference rounds (default: 3).",
    )
    parser.add_argument(
        "--refiner_checkpoint",
        default=None,
        help=(
            "Optional shared G_phi refinement adapter. Round 1 uses --checkpoint/F0; "
            "later rounds use this adapter with current SR plus the original-LR anchor."
        ),
    )
    parser.add_argument(
        "--reuse_f0_dir",
        default=None,
        help=(
            "Reuse validated round-1 F0 images from an existing iterative directory. "
            "Only selected inputs are copied; round-1 IQA and later rounds still run."
        ),
    )
    parser.add_argument(
        "--refiner_start_round",
        type=int,
        default=2,
        help="First round that loads --refiner_checkpoint (default: 2).",
    )
    parser.add_argument(
        "--round_seed_mode",
        choices=["fixed", "increment"],
        default="fixed",
        help="Reuse the base seed every round or increment it by round number.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted iterative directory after validating its input set, iteration count, "
            "seed and complete generated rounds. Existing images are never regenerated."
        ),
    )
    parser.add_argument(
        "--metric_sample_count",
        type=int,
        default=-1,
        help="Images evaluated per dataset per round: -1=all (default), 0=skip, positive=fixed subset.",
    )
    parser.add_argument(
        "--metric_sample_seed",
        type=int,
        default=42,
        help="Portable deterministic subset seed used when --metric_sample_count is positive.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="PyIQA metrics. Defaults to evaluation.metrics in the config.",
    )
    parser.add_argument(
        "--metric_device",
        default="cpu",
        help=(
            "Device for per-round PyIQA evaluation. CPU is the safe default while "
            "the SR model remains resident on GPU."
        ),
    )
    return parser


def main():
    run_iterative_inference(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
