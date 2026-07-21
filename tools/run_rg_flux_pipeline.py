import argparse
import copy
import datetime
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_METRICS = ["clipiqa", "clipiqa+", "nima", "niqe", "liqe", "musiq", "maniqa"]
PROMPT_VARIANTS = ("fixed", "suggestion", "iqa", "iqa_suggestion")
SUGGESTION_PAIRINGS = ("matched", "shuffled")


def cfg(config, path, default=None):
    current = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def cfg_bool(config, path, default=False):
    value = cfg(config, path, default)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off", "none", "null", ""}:
            return False
    return bool(value)


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def format_run_id(now=None):
    now = now or datetime.datetime.now()
    return now.strftime("%y%m%d%H")


def make_experiment_name(config):
    suffix = cfg(config, "training.suffix", "")
    lr_mode = cfg(config, "condition.lr_cond_mode", "latent_adapter")
    stage = cfg(config, "training.stage", "A")
    crop = cfg(config, "data.crop_size", 512)
    backend = str(cfg(config, "model.flux_backend", "flux1") or "flux1").lower()
    prompt_variant = cfg(config, "condition.prompt_variant", None)
    prompt_suffix = f"_prompt_{prompt_variant}" if prompt_variant else ""
    if backend in {"flux2_klein", "flux2-klein", "flux_2_klein"}:
        return f"rg_flux2_klein_sr_ms_stage{stage}_{lr_mode}_size{crop}{suffix}{prompt_suffix}"
    return f"rg_flux_sr_ms_stage{stage}_{lr_mode}_size{crop}{suffix}{prompt_suffix}"


def resolve_experiment_name(config, output_root, now=None):
    explicit_name = cfg(config, "training.exp_name", None)
    if explicit_name:
        return str(explicit_name), None
    base_name = make_experiment_name(config)
    if not cfg_bool(config, "training.add_datetime_suffix", True):
        return base_name, None
    run_id = str(cfg(config, "training.run_id", None) or format_run_id(now))
    exp_name = f"{base_name}_{run_id}"
    candidate = exp_name
    retry_index = 2
    output_root = Path(output_root)
    while (output_root / candidate).exists():
        candidate = f"{exp_name}_r{retry_index:02d}"
        retry_index += 1
    return candidate, run_id


def apply_prompt_variant(config, prompt_variant):
    if prompt_variant is None:
        return config
    prompt_variant = str(prompt_variant).strip().lower().replace("-", "_")
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unsupported prompt_variant '{prompt_variant}'. Expected one of: {', '.join(PROMPT_VARIANTS)}"
        )
    condition = config.setdefault("condition", {})
    condition["prompt_variant"] = prompt_variant
    condition["use_prompt"] = prompt_variant != "fixed"
    condition["use_suggestions"] = prompt_variant in {"suggestion", "iqa_suggestion"}
    prompt_schedule = condition.get("prompt_schedule")
    if isinstance(prompt_schedule, dict) and cfg_bool(config, "condition.prompt_schedule.enabled", False):
        prompt_schedule["after_variant"] = prompt_variant
    return config


def create_runtime_config(train_config_path, now=None, prompt_variant=None):
    source_config = load_yaml(train_config_path)
    runtime_config = copy.deepcopy(source_config)
    apply_prompt_variant(runtime_config, prompt_variant)
    runtime_config.setdefault("training", {})
    output_root = Path(cfg(runtime_config, "training.output_dir", "exp_rg_flux_sr"))
    exp_name, run_id = resolve_experiment_name(runtime_config, output_root=output_root, now=now)
    runtime_config["training"]["exp_name"] = exp_name
    runtime_config["training"]["resolved_exp_name"] = exp_name
    if run_id is not None:
        runtime_config["training"]["run_id"] = run_id
        runtime_config["training"]["resolved_run_id"] = run_id
    else:
        runtime_config["training"].pop("resolved_run_id", None)
    run_dir = output_root / exp_name
    runtime_path = run_dir / "pipeline_runtime_config.yaml"
    write_yaml(runtime_path, runtime_config)
    write_yaml(run_dir / "configs" / "source_config.yaml", source_config)
    write_yaml(run_dir / "configs" / "runtime_config.yaml", runtime_config)
    return runtime_config, run_dir, runtime_path


def parse_dataset_dirs(values):
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--dataset_dirs entry must use name=path format: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"--dataset_dirs entry must use name=path format: {value}")
        if name in parsed:
            raise ValueError(f"Duplicate dataset name: {name}")
        parsed[name] = Path(raw_path)
    return parsed


def format_checkpoint_step(step):
    value = str(step or "").strip()
    if not value:
        raise ValueError("checkpoint step must be non-empty")
    if value.lower() == "latest":
        return "latest"
    if value.startswith("checkpoint-"):
        value = value[len("checkpoint-") :]
    try:
        step_int = int(value)
    except ValueError as exc:
        raise ValueError(f"checkpoint step must be integer, checkpoint-XXXXXXXX, or latest: {step}") from exc
    if step_int < 0:
        raise ValueError(f"checkpoint step must be non-negative: {step}")
    return f"checkpoint-{step_int:08d}"


def find_latest_checkpoint_dir(run_dir):
    checkpoint_root = Path(run_dir) / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Run checkpoint directory does not exist: {checkpoint_root}")
    candidates = sorted(path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under: {checkpoint_root}")
    return candidates[-1]


def resolve_checkpoint_dir(run_dir, step):
    run_dir = Path(run_dir)
    formatted = format_checkpoint_step(step)
    checkpoint_dir = find_latest_checkpoint_dir(run_dir) if formatted == "latest" else run_dir / "checkpoints" / formatted
    adapter_dir = checkpoint_dir / "rg_flux_adapters"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Checkpoint adapter directory does not exist: {adapter_dir}")
    return checkpoint_dir


def planned_checkpoint_dir(run_dir, step):
    formatted = format_checkpoint_step(step)
    if formatted == "latest":
        try:
            return find_latest_checkpoint_dir(run_dir)
        except FileNotFoundError:
            return Path(run_dir) / "checkpoints" / "latest"
    return Path(run_dir) / "checkpoints" / formatted


def checkpoint_artifact_paths(run_dir, checkpoint_name):
    run_dir = Path(run_dir)
    checkpoint_name = Path(checkpoint_name).name
    return {
        "checkpoint_step": checkpoint_name,
        "checkpoint_path": run_dir / "checkpoints" / checkpoint_name / "rg_flux_adapters",
        "inference_dir": run_dir / "inference" / checkpoint_name,
        "inference_manifest": run_dir / "inference" / checkpoint_name / "inference_manifest.json",
        "metrics_dir": run_dir / "metrics" / checkpoint_name,
        "bad_cases_dir": run_dir / "bad_cases" / checkpoint_name,
    }


def suggestion_pairing_label(pairing, shuffle_seed=3407):
    pairing = str(pairing).strip().lower()
    if pairing not in SUGGESTION_PAIRINGS:
        raise ValueError(f"Unsupported suggestion pairing: {pairing}")
    if pairing == "matched":
        return "pairing_matched"
    return f"pairing_shuffled_seed{int(shuffle_seed)}"


def suggestion_pairing_artifact_paths(
    run_dir,
    checkpoint_name,
    pairing,
    shuffle_seed=3407,
    inference_output_root=None,
):
    run_dir = Path(run_dir)
    checkpoint_name = Path(checkpoint_name).name
    label = suggestion_pairing_label(pairing, shuffle_seed)
    if inference_output_root:
        inference_dir = Path(inference_output_root) / run_dir.name / checkpoint_name / label
        metrics_dir = inference_dir / "metrics"
        bad_cases_dir = inference_dir / "bad_cases"
        comparison_dir = Path(inference_output_root) / run_dir.name / checkpoint_name / "pairing_comparison"
    else:
        inference_dir = run_dir / "inference" / checkpoint_name / label
        metrics_dir = run_dir / "metrics" / checkpoint_name / label
        bad_cases_dir = run_dir / "bad_cases" / checkpoint_name / label
        comparison_dir = run_dir / "metrics" / checkpoint_name / "pairing_comparison"
    return {
        "label": label,
        "inference_dir": inference_dir,
        "inference_manifest": inference_dir / "inference_manifest.json",
        "metrics_dir": metrics_dir,
        "bad_cases_dir": bad_cases_dir,
        "comparison_dir": comparison_dir,
    }


def write_run_summary(run_dir, runtime_config_path, records, pipeline_manifest_path=None, pipeline_type="single_lora", extra=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for record in records or []:
        step = record.get("checkpoint_step")
        if not step:
            continue
        checkpoints[step] = {
            "checkpoint_path": record.get("checkpoint_path"),
            "inference_manifest": record.get("inference_manifest"),
            "inference_output_dir": record.get("inference_output_dir"),
            "metrics_output_dir": record.get("metrics_output_dir"),
            "bad_cases_output_dir": record.get("bad_cases_output_dir"),
        }
        if record.get("suggestion_pairing_runs") is not None:
            checkpoints[step]["suggestion_pairing_runs"] = record["suggestion_pairing_runs"]
            checkpoints[step]["pairing_comparison_output_dir"] = record.get("pairing_comparison_output_dir")
    payload = {
        "pipeline_type": pipeline_type,
        "run_dir": str(run_dir),
        "runtime_config_path": str(runtime_config_path) if runtime_config_path else None,
        "pipeline_manifest": str(pipeline_manifest_path) if pipeline_manifest_path else None,
        "checkpoints": checkpoints,
    }
    if extra:
        payload.update(extra)
    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return summary_path


def _append_optional(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def _append_optional_bool(cmd, positive_flag, negative_flag, value):
    if value is True:
        cmd.append(positive_flag)
    elif value is False:
        cmd.append(negative_flag)


def build_train_command(args, runtime_config_path):
    cmd = ["accelerate", "launch"]
    _append_optional(cmd, "--config_file", args.accelerate_config)
    _append_optional(cmd, "--num_processes", args.num_processes)
    cmd.extend(["train_rg_flux_sr.py", "--config", str(runtime_config_path)])
    if getattr(args, "dry_run_train", False):
        cmd.append("--dry_run")
    return cmd


def build_inference_command(
    args,
    run_dir,
    checkpoint_dir,
    config_path=None,
    output_dir=None,
    suggestion_pairing=None,
    suggestion_shuffle_seed=None,
    text_encoding_mode=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    artifact_paths = checkpoint_artifact_paths(run_dir, checkpoint_dir.name)
    cmd = [
        sys.executable,
        "inference_rg_flux_sr.py",
        "--dataset_dirs",
        *list(args.dataset_dirs),
        "--run_dir",
        str(run_dir),
        "--checkpoint_step",
        checkpoint_dir.name,
    ]
    if output_dir is not None:
        output_dir = Path(output_dir)
        manifest_path = output_dir / "inference_manifest.json"
        cmd.extend(["--output_dir", str(output_dir)])
    elif args.inference_output_root:
        output_root = Path(args.inference_output_root)
        manifest_path = output_root / Path(run_dir).name / checkpoint_dir.name / "inference_manifest.json"
        cmd.extend(["--output_root", str(output_root)])
    else:
        manifest_path = artifact_paths["inference_manifest"]
        cmd.extend(["--output_dir", str(artifact_paths["inference_dir"])])
    _append_optional(cmd, "--config", config_path)
    _append_optional(cmd, "--text_encoding_mode", text_encoding_mode or args.text_encoding_mode)
    _append_optional(cmd, "--text_embedding_cache", args.text_embedding_cache)
    _append_optional(cmd, "--jsonl_path", args.jsonl_path)
    _append_optional(cmd, "--num_inference_steps", args.num_inference_steps)
    _append_optional(cmd, "--upscale", args.upscale)
    _append_optional(cmd, "--dtype", args.dtype)
    _append_optional(cmd, "--device", args.device)
    _append_optional(cmd, "--lr_cond_mode", args.lr_cond_mode)
    _append_optional(cmd, "--min_size", args.min_size)
    _append_optional(cmd, "--prompt_variant", getattr(args, "prompt_variant", None))
    _append_optional(cmd, "--seed", getattr(args, "seed", None))
    _append_optional(cmd, "--suggestion_pairing", suggestion_pairing)
    _append_optional(cmd, "--suggestion_shuffle_seed", suggestion_shuffle_seed)
    _append_optional_bool(cmd, "--use_prompt", "--no-use_prompt", args.use_prompt)
    _append_optional_bool(cmd, "--use_suggestions", "--no-use_suggestions", args.use_suggestions)
    _append_optional_bool(cmd, "--use_degradation_vector", "--no-use_degradation_vector", args.use_degradation_vector)
    if args.restore_input_size:
        cmd.append("--restore_input_size")
    return cmd, manifest_path


def build_pairing_comparison_command(matched_metrics_dir, shuffled_metrics_dir, output_dir):
    return [
        sys.executable,
        "tools/compare_rg_flux_pairing_metrics.py",
        "--matched_metrics_dir",
        str(matched_metrics_dir),
        "--shuffled_metrics_dir",
        str(shuffled_metrics_dir),
        "--output_dir",
        str(output_dir),
    ]


def build_eval_command(args, inference_manifest, metrics_dir=None):
    inference_manifest = Path(inference_manifest)
    metrics_dir = Path(metrics_dir) if metrics_dir is not None else inference_manifest.parent / "metrics"
    cmd = [
        sys.executable,
        "eval_rg_flux_sr_metrics.py",
        "--inference_manifest",
        str(inference_manifest),
        "--output_dir",
        str(metrics_dir),
        "--device",
        str(args.metric_device),
    ]
    if args.metrics:
        cmd.extend(["--metrics", *list(args.metrics)])
    return cmd, metrics_dir


def build_bad_case_command(args, metrics_dir, bad_cases_dir, inference_manifest=None):
    metrics = list(args.bad_case_metrics or [])
    if not metrics:
        raise ValueError("--bad_case_metrics must include at least one metric when --run_bad_cases is enabled.")
    cmd = [
        sys.executable,
        "tools/analyze_rg_flux_bad_cases.py",
        "--metrics_csv",
        str(Path(metrics_dir) / "per_image_scores.csv"),
        "--summary_json",
        str(Path(metrics_dir) / "summary_scores.json"),
        "--metrics",
        *metrics,
        "--mode",
        str(args.bad_case_mode),
        "--worst_k",
        str(args.bad_case_worst_k),
        "--lq_dirs",
        *list(args.dataset_dirs),
        "--output_dir",
        str(bad_cases_dir),
    ]
    _append_optional(cmd, "--inference_manifest", inference_manifest)
    _append_optional(cmd, "--jsonl_path", getattr(args, "jsonl_path", None))
    _append_optional(cmd, "--prompt_variant", getattr(args, "prompt_variant", None))
    _append_optional(cmd, "--font_size", getattr(args, "bad_case_font_size", 40))
    _append_optional_bool(
        cmd,
        "--use_prompt",
        "--no-use_prompt",
        getattr(args, "use_prompt", None),
    )
    _append_optional_bool(
        cmd,
        "--use_suggestions",
        "--no-use_suggestions",
        getattr(args, "use_suggestions", None),
    )
    return cmd


def find_run_config_path(run_dir):
    for name in ("pipeline_runtime_config.yaml", "args.json"):
        path = Path(run_dir) / name
        if path.exists():
            return path
    return None


def load_run_config(run_dir):
    path = find_run_config_path(run_dir)
    if path is None:
        return None
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return load_yaml(path)


def resolve_skip_train_config_path(run_dir, train_config_path=None):
    run_config_path = find_run_config_path(run_dir)
    if run_config_path is not None:
        return run_config_path
    if train_config_path:
        train_config_path = Path(train_config_path)
        if train_config_path.exists():
            return train_config_path
        raise FileNotFoundError(f"--train_config does not exist: {train_config_path}")
    raise FileNotFoundError(
        f"No pipeline_runtime_config.yaml or args.json found under {run_dir}. "
        "Pass --train_config when using --skip_train with an older run directory."
    )


def apply_config_prompt_defaults(args, config):
    if not isinstance(config, dict):
        return
    condition = config.get("condition", {}) if isinstance(config.get("condition"), dict) else {}
    if getattr(args, "prompt_variant", None) is None and "prompt_variant" in condition:
        args.prompt_variant = condition["prompt_variant"]
    if args.use_prompt is None and "use_prompt" in condition:
        args.use_prompt = bool(condition["use_prompt"])
    if args.use_suggestions is None and "use_suggestions" in condition:
        args.use_suggestions = bool(condition["use_suggestions"])
    if args.use_degradation_vector is None and "use_degradation_vector" in condition:
        args.use_degradation_vector = bool(condition["use_degradation_vector"])


def run_command(cmd):
    print("[pipeline] running:", " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def write_pipeline_manifest(run_dir, runtime_config_path, checkpoint_steps, records, train_returncode=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "pipeline_manifest.json"
    payload = {
        "runtime_config_path": str(runtime_config_path) if runtime_config_path else None,
        "run_dir": str(run_dir),
        "checkpoint_steps": list(checkpoint_steps),
        "train_returncode": train_returncode,
        "records": records,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_run_summary(
        run_dir=run_dir,
        runtime_config_path=runtime_config_path,
        records=records,
        pipeline_manifest_path=manifest_path,
        pipeline_type="single_lora",
        extra={"train_returncode": train_returncode},
    )
    return manifest_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run RG-FLUX-SR train -> inference -> metrics as one pipeline.")
    parser.add_argument("--train_config", default=None, help="Training YAML config. Required unless --skip_train.")
    parser.add_argument("--accelerate_config", default=None, help="Accelerate config used for training.")
    parser.add_argument("--num_processes", type=int, default=None, help="Optional accelerate --num_processes.")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and use --run_dir checkpoints.")
    parser.add_argument("--run_dir", default=None, help="Existing run directory, required with --skip_train.")
    parser.add_argument("--checkpoint_steps", nargs="+", required=True, help="Checkpoint steps, e.g. 20000 40000 latest.")
    parser.add_argument("--dataset_dirs", nargs="+", required=True, help="Shared inference/eval datasets as name=folder.")
    parser.add_argument(
        "--inference_output_root",
        default=None,
        help=(
            "Optional root for legacy inference outputs. If omitted, outputs are written under "
            "run_dir/inference/checkpoint-XXXXXXXX."
        ),
    )
    parser.add_argument("--text_encoding_mode", choices=["online", "cached", "auto"], default=None)
    parser.add_argument("--text_embedding_cache", default=None)
    parser.add_argument("--jsonl_path", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--lr_cond_mode", choices=["latent_adapter", "latent_concat", "flux2_image_concat"], default=None)
    parser.add_argument("--min_size", type=int, default=None)
    parser.add_argument("--restore_input_size", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Inference seed shared by all comparison runs.")
    parser.add_argument("--use_prompt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_suggestions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--prompt_variant",
        choices=PROMPT_VARIANTS,
        default=None,
        help="Prompt ablation variant. Explicit variants override the legacy prompt booleans.",
    )
    parser.add_argument("--use_degradation_vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--compare_suggestion_pairing",
        action="store_true",
        help=(
            "Run matched and deterministic cross-image shuffled suggestions with the same checkpoint, "
            "then produce paired metric deltas."
        ),
    )
    parser.add_argument(
        "--suggestion_shuffle_seed",
        type=int,
        default=3407,
        help="Base seed for the deterministic no-self-match suggestion permutation.",
    )
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--metric_device", default="cpu")
    parser.add_argument("--run_bad_cases", action="store_true", help="Run bad case analysis after metrics.")
    parser.add_argument("--bad_case_metrics", nargs="+", default=["clipiqa", "maniqa", "musiq"])
    parser.add_argument("--bad_case_mode", choices=["separate", "joint_mean"], default="separate")
    parser.add_argument("--bad_case_worst_k", type=int, default=50)
    parser.add_argument("--bad_case_font_size", type=int, default=40)
    parser.add_argument("--dry_run_train", action="store_true", help="Pass --dry_run to train_rg_flux_sr.py.")
    parser.add_argument("--dry_run_pipeline", action="store_true", help="Print/write command plan without running commands.")
    return parser


def parse_args(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.skip_train and not args.run_dir:
        raise ValueError("--run_dir is required with --skip_train.")
    if not args.skip_train and not args.train_config:
        raise ValueError("--train_config is required unless --skip_train.")
    parse_dataset_dirs(args.dataset_dirs)
    return args


def main(argv=None):
    args = parse_args(argv)
    records = []
    train_returncode = None

    if args.skip_train:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        runtime_config_path = resolve_skip_train_config_path(run_dir, args.train_config)
        runtime_config = load_yaml(runtime_config_path) if runtime_config_path.suffix != ".json" else json.loads(runtime_config_path.read_text(encoding="utf-8"))
    else:
        runtime_config, run_dir, runtime_config_path = create_runtime_config(
            args.train_config,
            prompt_variant=args.prompt_variant,
        )
        train_cmd = build_train_command(args, runtime_config_path)
        if args.dry_run_pipeline:
            train_returncode = None
        else:
            train_returncode = run_command(train_cmd)
            if train_returncode != 0:
                write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
                raise SystemExit(train_returncode)

    apply_config_prompt_defaults(args, runtime_config)

    if args.compare_suggestion_pairing:
        if not args.jsonl_path:
            raise ValueError("--compare_suggestion_pairing requires --jsonl_path.")
        if args.prompt_variant not in {"suggestion", "iqa_suggestion"}:
            raise ValueError(
                "--compare_suggestion_pairing requires --prompt_variant suggestion or iqa_suggestion "
                "so that changing the suggestion is the intended intervention."
            )
        if args.text_encoding_mode != "online":
            print(
                "[pipeline] suggestion pairing comparison uses online text encoding for both runs.",
                flush=True,
            )

    for step in args.checkpoint_steps:
        checkpoint_dir = planned_checkpoint_dir(run_dir, step) if args.dry_run_pipeline else resolve_checkpoint_dir(run_dir, step)
        if args.compare_suggestion_pairing:
            pairing_runs = {}
            record = {
                "checkpoint_step": checkpoint_dir.name,
                "checkpoint_path": str(checkpoint_dir / "rg_flux_adapters"),
                "suggestion_pairing_comparison": True,
                "suggestion_shuffle_seed": args.suggestion_shuffle_seed,
                "inference_seed": args.seed,
                "text_encoding_mode": "online",
                "suggestion_pairing_runs": pairing_runs,
                "pairing_comparison_output_dir": None,
                "pairing_comparison_command": None,
                "pairing_comparison_returncode": None,
            }

            for pairing in SUGGESTION_PAIRINGS:
                paths = suggestion_pairing_artifact_paths(
                    run_dir=run_dir,
                    checkpoint_name=checkpoint_dir.name,
                    pairing=pairing,
                    shuffle_seed=args.suggestion_shuffle_seed,
                    inference_output_root=args.inference_output_root,
                )
                inference_cmd, inference_manifest = build_inference_command(
                    args,
                    run_dir,
                    checkpoint_dir,
                    runtime_config_path,
                    output_dir=paths["inference_dir"],
                    suggestion_pairing=pairing,
                    suggestion_shuffle_seed=args.suggestion_shuffle_seed,
                    text_encoding_mode="online",
                )
                eval_cmd, metrics_dir = build_eval_command(
                    args,
                    inference_manifest,
                    paths["metrics_dir"],
                )
                bad_case_cmd = (
                    build_bad_case_command(
                        args,
                        metrics_dir,
                        paths["bad_cases_dir"],
                        inference_manifest=inference_manifest,
                    )
                    if args.run_bad_cases
                    else None
                )
                pairing_record = {
                    "pairing": pairing,
                    "inference_manifest": str(inference_manifest),
                    "inference_output_dir": str(paths["inference_dir"]),
                    "metrics_output_dir": str(metrics_dir),
                    "bad_cases_output_dir": str(paths["bad_cases_dir"]),
                    "inference_command": inference_cmd,
                    "eval_command": eval_cmd,
                    "bad_case_command": bad_case_cmd,
                    "inference_returncode": None,
                    "eval_returncode": None,
                    "bad_case_returncode": None,
                }
                pairing_runs[pairing] = pairing_record

                if not args.dry_run_pipeline:
                    pairing_record["inference_returncode"] = run_command(inference_cmd)
                    if pairing_record["inference_returncode"] != 0:
                        records.append(record)
                        write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
                        raise SystemExit(pairing_record["inference_returncode"])
                    pairing_record["eval_returncode"] = run_command(eval_cmd)
                    if pairing_record["eval_returncode"] != 0:
                        records.append(record)
                        write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
                        raise SystemExit(pairing_record["eval_returncode"])
                    if args.run_bad_cases:
                        pairing_record["bad_case_returncode"] = run_command(bad_case_cmd)
                        if pairing_record["bad_case_returncode"] != 0:
                            records.append(record)
                            write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
                            raise SystemExit(pairing_record["bad_case_returncode"])

            comparison_dir = suggestion_pairing_artifact_paths(
                run_dir=run_dir,
                checkpoint_name=checkpoint_dir.name,
                pairing="matched",
                shuffle_seed=args.suggestion_shuffle_seed,
                inference_output_root=args.inference_output_root,
            )["comparison_dir"]
            comparison_cmd = build_pairing_comparison_command(
                pairing_runs["matched"]["metrics_output_dir"],
                pairing_runs["shuffled"]["metrics_output_dir"],
                comparison_dir,
            )
            record["pairing_comparison_output_dir"] = str(comparison_dir)
            record["pairing_comparison_command"] = comparison_cmd
            if not args.dry_run_pipeline:
                record["pairing_comparison_returncode"] = run_command(comparison_cmd)
                if record["pairing_comparison_returncode"] != 0:
                    records.append(record)
                    write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
                    raise SystemExit(record["pairing_comparison_returncode"])
            records.append(record)
            continue

        inference_cmd, inference_manifest = build_inference_command(args, run_dir, checkpoint_dir, runtime_config_path)
        artifact_paths = checkpoint_artifact_paths(run_dir, checkpoint_dir.name)
        metrics_target = None if args.inference_output_root else artifact_paths["metrics_dir"]
        eval_cmd, metrics_dir = build_eval_command(args, inference_manifest, metrics_target)
        bad_cases_dir = artifact_paths["bad_cases_dir"]
        bad_case_cmd = (
            build_bad_case_command(
                args,
                metrics_dir,
                bad_cases_dir,
                inference_manifest=inference_manifest,
            )
            if args.run_bad_cases
            else None
        )
        record = {
            "checkpoint_step": checkpoint_dir.name,
            "checkpoint_path": str(checkpoint_dir / "rg_flux_adapters"),
            "inference_manifest": str(inference_manifest),
            "inference_output_dir": str(Path(inference_manifest).parent),
            "metrics_output_dir": str(metrics_dir),
            "bad_cases_output_dir": str(bad_cases_dir),
            "inference_command": inference_cmd,
            "eval_command": eval_cmd,
            "bad_case_command": bad_case_cmd,
            "inference_returncode": None,
            "eval_returncode": None,
            "bad_case_returncode": None,
        }
        if not args.dry_run_pipeline:
            record["inference_returncode"] = run_command(inference_cmd)
            if record["inference_returncode"] != 0:
                records.append(record)
                write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
                raise SystemExit(record["inference_returncode"])
            record["eval_returncode"] = run_command(eval_cmd)
            if record["eval_returncode"] != 0:
                records.append(record)
                write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
                raise SystemExit(record["eval_returncode"])
            if args.run_bad_cases:
                record["bad_case_returncode"] = run_command(bad_case_cmd)
                if record["bad_case_returncode"] != 0:
                    records.append(record)
                    write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
                    raise SystemExit(record["bad_case_returncode"])
        records.append(record)

    manifest_path = write_pipeline_manifest(run_dir, runtime_config_path, args.checkpoint_steps, records, train_returncode)
    print(f"[pipeline] saved manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
