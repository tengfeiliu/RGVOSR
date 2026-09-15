"""Resolve the portable one-click RL-SR runtime settings from one YAML config.

It centralizes the only dataset name used by the launcher: ``data.jsonl_path``.
The generated run folder contains the checkpoint tag, key model settings, and a
timestamp, and gets a suffix when a same-second collision occurs.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import yaml


def _get(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _slug(value) -> str:
    text = str(value or "na").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "na"


def _resolve_path(value, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _checkpoint_tag(checkpoint: str) -> str:
    path = Path(checkpoint).expanduser()
    if path.name == "rg_flux_adapters":
        path = path.parent
    match = re.search(r"checkpoint-(\d+)", path.name)
    if match:
        return f"f0-{int(match.group(1)):06d}"
    return _slug(path.name)


def _nonnegative_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_context(
    config_path: Path, f0_checkpoint: str, repo_root: Path, output_root_override: str | None,
    train_max_samples_override: int | None = None,
    train_subset_seed_override: int | None = None,
    existing_run_dir: Path | None = None,
):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    data_jsonl_value = _get(config, "data", "jsonl_path")
    if not data_jsonl_value:
        raise ValueError("Config is missing data.jsonl_path")
    data_jsonl_path = _resolve_path(data_jsonl_value, repo_root)
    eval_jsonl_value = _get(config, "rl_sr", "evaluation", "jsonl_path", default=None)
    evaluation_jsonl_path = _resolve_path(eval_jsonl_value, repo_root) if eval_jsonl_value else data_jsonl_path

    output_root_value = output_root_override or _get(config, "rl_sr", "output_root", default="exp_rg_flux_rl")
    output_root = _resolve_path(output_root_value, repo_root)
    iterations = int(_get(config, "rl_sr", "iterations", default=4))
    candidates = int(_get(config, "rl_sr", "num_candidates", default=4))
    subset_defaults = {
        "train_max_samples": _get(config, "rl_sr", "train_max_samples", default=0),
        "train_subset_seed": _get(config, "rl_sr", "train_subset_seed", default=42),
    }
    if existing_run_dir is not None:
        if not existing_run_dir.is_dir():
            raise FileNotFoundError(f"Existing RL-SR run does not exist: {existing_run_dir}")
        saved_path = existing_run_dir / "run_context.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.is_file() else {}
        # Old runs predate subset selection and therefore used the whole dataset.
        subset_defaults = {
            "train_max_samples": saved.get("train_max_samples", 0),
            "train_subset_seed": saved.get("train_subset_seed", 42),
        }
    train_max_samples = _nonnegative_int(
        subset_defaults["train_max_samples"] if train_max_samples_override is None else train_max_samples_override,
        "rl_sr.train_max_samples / TRAIN_MAX_SAMPLES",
    )
    train_subset_seed = _nonnegative_int(
        subset_defaults["train_subset_seed"] if train_subset_seed_override is None else train_subset_seed_override,
        "rl_sr.train_subset_seed / TRAIN_SUBSET_SEED",
    )
    if existing_run_dir is not None and (
        train_max_samples != subset_defaults["train_max_samples"]
        or (train_max_samples and train_subset_seed != subset_defaults["train_subset_seed"])
    ):
        raise ValueError(
            "Cannot change the training subset of an existing RL_SR_RUN_DIR. "
            "Unset RL_SR_RUN_DIR to create a separate run with the new sample limit."
        )
    metrics = _get(config, "rl_sr", "evaluation", "metrics", default=["clipiqa", "clipiqa+", "nima", "niqe", "liqe", "musiq"])
    if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
        raise ValueError("rl_sr.evaluation.metrics must be a list of metric names")
    raw_evaluation_datasets = _get(config, "rl_sr", "evaluation", "datasets", default=[])
    if raw_evaluation_datasets is None:
        raw_evaluation_datasets = []
    if not isinstance(raw_evaluation_datasets, list):
        raise ValueError("rl_sr.evaluation.datasets must be a list when provided")
    evaluation_dataset_names = [
        str(item.get("name") or "").strip()
        for item in raw_evaluation_datasets
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    evaluation_tag = "-".join(_slug(name) for name in evaluation_dataset_names) or "evaluation"

    name_parts = [
        _slug(_get(config, "rl_sr", "run_prefix", default="rlsr")),
        _checkpoint_tag(f0_checkpoint),
        _slug(_get(config, "model", "flux_backend", default="flux")),
        _slug(_get(config, "condition", "lr_cond_mode", default="lrcond")),
        _slug(_get(config, "condition", "prompt_variant", default="prompt")),
        f"c{_slug(_get(config, 'model', 'lora_moe', 'router_input_mode', default='na'))}",
        f"s{int(_get(config, 'data', 'crop_size', default=0))}",
        *([f"n{train_max_samples}-ss{train_subset_seed}"] if train_max_samples else []),
        f"r{iterations}",
        f"k{candidates}",
        f"sft{int(_get(config, 'rl_sr', 'sft_max_steps', default=0))}",
        f"nft{int(_get(config, 'rl_sr', 'rl_max_steps', default=0))}",
        f"eval-{evaluation_tag}",
        datetime.now().strftime("%y%m%d-%H%M%S"),
    ]
    run_name = "_".join(name_parts)
    return {
        "config_path": str(config_path),
        "data_jsonl_path": str(data_jsonl_path),
        "evaluation_jsonl_path": str(evaluation_jsonl_path),
        "output_root": str(output_root),
        "run_name": run_name,
        "iterations": iterations,
        "num_candidates": candidates,
        "train_max_samples": train_max_samples,
        "train_subset_seed": train_subset_seed,
        "sft_max_steps": int(_get(config, "rl_sr", "sft_max_steps", default=1000)),
        "rl_max_steps": int(_get(config, "rl_sr", "rl_max_steps", default=250)),
        "seed": int(_get(config, "training", "seed", default=42)),
        "metric_device": str(_get(config, "rl_sr", "evaluation", "metric_device", default="cpu")),
        "reward_device": str(_get(config, "rl_sr", "reward_device", default="cuda")),
        "dataset_id": str(_get(config, "rl_sr", "dataset_id", default="paired_sr_v1")),
        "evaluation_datasets": evaluation_dataset_names,
        "metrics": metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--f0_checkpoint", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--output_root", default=None, help="Optional runtime override; not part of artifact identity.")
    parser.add_argument("--train_max_samples", type=int, default=None, help="Override rl_sr.train_max_samples; 0 means all.")
    parser.add_argument("--train_subset_seed", type=int, default=None, help="Override rl_sr.train_subset_seed.")
    parser.add_argument("--existing_run_dir", type=Path, default=None, help="Restore and validate an existing run's training subset.")
    parser.add_argument("--field", default=None, help="Print exactly one resolved context field.")
    parser.add_argument("--create", action="store_true", help="Atomically choose and create an unused run folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    context = load_context(
        config_path, args.f0_checkpoint, repo_root, args.output_root,
        train_max_samples_override=args.train_max_samples,
        train_subset_seed_override=args.train_subset_seed,
        existing_run_dir=args.existing_run_dir.expanduser().resolve() if args.existing_run_dir else None,
    )
    if args.create:
        output_root = Path(context["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        candidate = output_root / context["run_name"]
        suffix = 1
        while True:
            try:
                candidate.mkdir()
                break
            except FileExistsError:
                suffix += 1
                candidate = output_root / f"{context['run_name']}_r{suffix:02d}"
        context["run_dir"] = str(candidate)
        (candidate / "run_context.json").write_text(
            json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(context["run_dir"])
        return
    if args.field:
        if args.field not in context:
            raise ValueError(f"Unknown field: {args.field}")
        value = context[args.field]
        print(" ".join(value) if isinstance(value, list) else value)
        return
    print(json.dumps(context, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
