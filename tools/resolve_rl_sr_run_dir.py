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


def load_context(config_path: Path, f0_checkpoint: str, repo_root: Path, output_root_override: str | None):
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
    parser.add_argument("--field", default=None, help="Print exactly one resolved context field.")
    parser.add_argument("--create", action="store_true", help="Atomically choose and create an unused run folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    context = load_context(config_path, args.f0_checkpoint, repo_root, args.output_root)
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
