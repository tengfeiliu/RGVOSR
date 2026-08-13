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

    resolved_run = _resolve_iterative_run(args)
    output_root = Path(resolved_run["output_dir"])
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
    lineage = _build_lineage(source_datasets)
    lineage_path = output_root / "sample_lineage.jsonl"
    _write_lineage(lineage_path, lineage)

    manifest_path = output_root / "iterative_manifest.json"
    trend_path = output_root / "metric_trends.csv"
    manifest = {
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
    _write_json_atomic(manifest_path, manifest)

    artist = None
    trend_rows = []
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
        lr_cond_mode = config["condition"]["lr_cond_mode"]

        for round_number in range(1, args.iterations + 1):
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
            condition_index = _build_round_condition_index(
                base_condition_index,
                lineage,
                current_inputs,
            )

            dataset_metadata = {}
            round_datasets = []
            for dataset_name, input_path in current_inputs.items():
                dataset_output_dir = output_dirs[dataset_name]
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
                },
            )

            metrics_dir = round_output_dir / "metrics"
            metric_summary = evaluate_dataset_dirs(
                dataset_dirs=output_dirs,
                output_dir=metrics_dir,
                metrics=metrics,
                device=args.metric_device,
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

            manifest["rounds"].append(
                {
                    "round": round_number,
                    "input_upscale": round_args.upscale,
                    "seed": round_seed,
                    "input_dirs": {
                        name: str(path) for name, path in current_inputs.items()
                    },
                    "output_dirs": {
                        name: str(path) for name, path in output_dirs.items()
                    },
                    "inference_manifest": str(round_manifest_path),
                    "metrics_dir": str(metrics_dir),
                    "metric_summary": str(metrics_dir / "summary_scores.json"),
                    "missing_output_count": missing_outputs,
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
        "--round_seed_mode",
        choices=["fixed", "increment"],
        default="fixed",
        help="Reuse the base seed every round or increment it by round number.",
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
