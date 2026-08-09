import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.rg_sr_metrics import DEFAULT_OMGSR_METRICS  # noqa: E402
from models.prompt_builder import PROMPT_VARIANTS  # noqa: E402
from tools.moe_router_ablation import (  # noqa: E402
    ROUTER_ABLATION_MODES,
    checkpoint_fingerprint,
    mean_router_weights_from_history,
)
from tools.run_rg_flux_pipeline import build_bad_case_command, build_eval_command  # noqa: E402


PUBLIC_MODES = (*ROUTER_ABLATION_MODES[:-1], "onehot")


def _load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def find_run_config(run_dir, explicit_config=None):
    if explicit_config:
        path = Path(explicit_config)
        if not path.exists():
            raise FileNotFoundError(f"Config does not exist: {path}")
        return path
    for name in ("pipeline_runtime_config.yaml", "args.json"):
        path = Path(run_dir) / name
        if path.exists():
            return path
    return None


def resolve_num_experts(run_dir, config_path=None, explicit_num_experts=None):
    if explicit_num_experts is not None:
        value = int(explicit_num_experts)
    else:
        config_path = find_run_config(run_dir, config_path)
        if config_path is None:
            raise FileNotFoundError(
                "Could not infer num_routed_experts because the run has no config. "
                "Pass --config or --num_experts."
            )
        config = _load_config(config_path)
        value = int(config.get("model", {}).get("lora_moe", {}).get("num_routed_experts", 0))
    if value <= 0:
        raise ValueError("num_routed_experts must be positive.")
    return value


def resolve_checkpoint(run_dir, step):
    run_dir = Path(run_dir)
    text = str(step).strip()
    if text.lower() == "latest":
        candidates = sorted((run_dir / "checkpoints").glob("checkpoint-*"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found under {run_dir / 'checkpoints'}")
        checkpoint_dir = candidates[-1]
    else:
        if text.startswith("checkpoint-"):
            text = text[len("checkpoint-") :]
        checkpoint_dir = run_dir / "checkpoints" / f"checkpoint-{int(text):08d}"
    adapter_dir = checkpoint_dir / "rg_flux_adapters"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Checkpoint adapter does not exist: {adapter_dir}")
    return checkpoint_dir, adapter_dir


def expand_modes(modes, num_experts):
    modes = list(modes)
    # Every metric comparison uses A as its paired baseline. Add it for any subset.
    if "learned_top2" not in modes:
        modes.insert(0, "learned_top2")
    ordered = []
    preferred = ["learned_top2", "fixed_mean", "shuffle_condition8", "uniform", "dense_soft", "onehot"]
    for mode in preferred:
        if mode not in modes:
            continue
        if mode == "onehot":
            ordered.extend(("onehot", index) for index in range(num_experts))
        else:
            ordered.append((mode, None))
    return ordered


def mode_label(mode, expert_index=None):
    return f"onehot_e{expert_index}" if mode == "onehot" else mode


def _append_optional(command, flag, value):
    if value is not None:
        command.extend([flag, str(value)])


def _append_optional_bool(command, positive, negative, value):
    if value is True:
        command.append(positive)
    elif value is False:
        command.append(negative)


def build_ablation_inference_command(
    args,
    checkpoint_dir,
    output_dir,
    mode,
    expert_index=None,
    fixed_weights=None,
    learned_trace=None,
    config_path=None,
):
    command = [
        sys.executable,
        "tools/inference_rg_flux_moe_router_ablation.py",
        "--dataset_dirs",
        *list(args.dataset_dirs),
        "--run_dir",
        str(args.run_dir),
        "--checkpoint_step",
        checkpoint_dir.name,
        "--output_dir",
        str(output_dir),
        "--router_ablation_mode",
        mode,
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--upscale",
        str(args.upscale),
        "--dtype",
        str(args.dtype),
        "--seed",
        str(args.seed),
    ]
    _append_optional(command, "--config", config_path)
    _append_optional(command, "--jsonl_path", args.jsonl_path)
    _append_optional(command, "--text_encoding_mode", args.text_encoding_mode)
    _append_optional(command, "--text_embedding_cache", args.text_embedding_cache)
    _append_optional(command, "--device", args.device)
    _append_optional(command, "--lr_cond_mode", args.lr_cond_mode)
    _append_optional(command, "--min_size", args.min_size)
    _append_optional(command, "--prompt_variant", args.prompt_variant)
    _append_optional_bool(command, "--use_prompt", "--no-use_prompt", args.use_prompt)
    _append_optional_bool(command, "--use_suggestions", "--no-use_suggestions", args.use_suggestions)
    _append_optional_bool(command, "--include_caption", "--no-include_caption", args.include_caption)
    _append_optional_bool(
        command,
        "--use_degradation_vector",
        "--no-use_degradation_vector",
        args.use_degradation_vector,
    )
    if args.full_frame_inference:
        command.append("--full_frame_inference")
    if args.restore_input_size:
        command.append("--restore_input_size")
    if mode == "fixed_mean":
        command.extend(["--fixed_weights", *[f"{value:.10g}" for value in fixed_weights]])
    if mode == "onehot":
        command.extend(["--onehot_expert", str(expert_index)])
    if mode == "shuffle_condition8":
        command.extend(
            [
                "--shuffle_reference_trace",
                str(learned_trace),
                "--condition_shuffle_seed",
                str(args.condition_shuffle_seed),
            ]
        )
    return command


def run_command(command, dry_run=False):
    print("[router_ablation] running:", " ".join(map(str, command)), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run read-only A-F LoRA-MoE Router ablations over existing checkpoints."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint_steps", nargs="+", required=True)
    parser.add_argument("--ablation_modes", nargs="+", choices=PUBLIC_MODES, default=list(PUBLIC_MODES))
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--dataset_dirs", nargs="+", required=True)
    parser.add_argument("--jsonl_path", default=None)
    parser.add_argument("--ablation_output_root", default="router_ablation_outputs")
    parser.add_argument("--fixed_weight_history", default=None)
    parser.add_argument("--fixed_weight_last_n", type=int, default=1000)
    parser.add_argument("--fixed_weight_field", choices=("usage", "used"), default="usage")
    parser.add_argument("--condition_shuffle_seed", type=int, default=3407)
    parser.add_argument("--text_encoding_mode", choices=("online", "cached", "auto"), default=None)
    parser.add_argument("--text_embedding_cache", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--lr_cond_mode",
        choices=("latent_adapter", "latent_concat", "flux2_image_concat"),
        default=None,
    )
    parser.add_argument("--min_size", type=int, default=None)
    parser.add_argument("--full_frame_inference", action="store_true")
    parser.add_argument("--restore_input_size", action="store_true")
    parser.add_argument("--use_prompt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_suggestions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include_caption", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prompt_variant", choices=PROMPT_VARIANTS, default=None)
    parser.add_argument("--use_degradation_vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_OMGSR_METRICS)
    parser.add_argument("--metric_device", default="cpu")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--run_bad_cases", action="store_true")
    parser.add_argument("--bad_case_metrics", nargs="+", default=["clipiqa", "maniqa", "musiq"])
    parser.add_argument("--bad_case_mode", choices=("separate", "joint_mean"), default="separate")
    parser.add_argument("--bad_case_worst_k", type=int, default=50)
    parser.add_argument("--bad_case_font_size", type=int, default=40)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--no_verify_checkpoint", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.run_bad_cases and args.skip_eval:
        raise ValueError("--run_bad_cases cannot be combined with --skip_eval.")
    run_dir = Path(args.run_dir)
    config_path = find_run_config(run_dir, args.config)
    num_experts = resolve_num_experts(run_dir, config_path, args.num_experts)
    expanded_modes = expand_modes(args.ablation_modes, num_experts)
    output_root = Path(args.ablation_output_root) / run_dir.name
    output_root.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.fixed_weight_history or run_dir / "logs" / "loss_history.csv")
    manifest = {
        "run_dir": str(run_dir),
        "config": str(config_path) if config_path else None,
        "num_experts": num_experts,
        "checkpoint_steps": list(args.checkpoint_steps),
        "ablation_modes": [mode_label(mode, expert) for mode, expert in expanded_modes],
        "records": [],
    }

    try:
        for requested_step in args.checkpoint_steps:
            checkpoint_dir, adapter_dir = resolve_checkpoint(run_dir, requested_step)
            checkpoint_root = output_root / checkpoint_dir.name
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            fingerprint_before = None if args.no_verify_checkpoint or args.dry_run else checkpoint_fingerprint(adapter_dir)
            fixed_weights = None
            fixed_weight_metadata = None
            if any(mode == "fixed_mean" for mode, _ in expanded_modes):
                fixed_weights, fixed_weight_metadata = mean_router_weights_from_history(
                    history_path,
                    checkpoint_step=checkpoint_dir.name,
                    last_n=args.fixed_weight_last_n,
                    weight_field=args.fixed_weight_field,
                    return_metadata=True,
                )
            learned_trace = checkpoint_root / "learned_top2" / "router_trace.jsonl"

            for mode, expert_index in expanded_modes:
                label = mode_label(mode, expert_index)
                mode_dir = checkpoint_root / label
                inference_manifest = mode_dir / "inference_manifest.json"
                inference_command = build_ablation_inference_command(
                    args,
                    checkpoint_dir=checkpoint_dir,
                    output_dir=mode_dir,
                    mode=mode,
                    expert_index=expert_index,
                    fixed_weights=fixed_weights,
                    learned_trace=learned_trace,
                    config_path=config_path,
                )
                record = {
                    "checkpoint": checkpoint_dir.name,
                    "mode": label,
                    "inference_output_dir": str(mode_dir),
                    "inference_manifest": str(inference_manifest),
                    "inference_command": inference_command,
                    "fixed_weights": fixed_weights if mode == "fixed_mean" else None,
                    "fixed_weight_field": args.fixed_weight_field if mode == "fixed_mean" else None,
                    "fixed_weight_last_n": args.fixed_weight_last_n if mode == "fixed_mean" else None,
                    "fixed_weight_metadata": fixed_weight_metadata if mode == "fixed_mean" else None,
                    "inference_returncode": None,
                    "eval_returncode": None,
                    "bad_case_returncode": None,
                }
                manifest["records"].append(record)
                returncode = run_command(inference_command, dry_run=args.dry_run)
                record["inference_returncode"] = returncode
                if returncode != 0:
                    raise SystemExit(returncode)

                if not args.skip_eval:
                    eval_command, metrics_dir = build_eval_command(
                        args,
                        inference_manifest,
                        metrics_dir=mode_dir / "metrics",
                    )
                    record["eval_command"] = eval_command
                    record["metrics_output_dir"] = str(metrics_dir)
                    returncode = run_command(eval_command, dry_run=args.dry_run)
                    record["eval_returncode"] = returncode
                    if returncode != 0:
                        raise SystemExit(returncode)
                    if args.run_bad_cases:
                        bad_case_command = build_bad_case_command(
                            args,
                            metrics_dir,
                            mode_dir / "bad_cases",
                            inference_manifest=inference_manifest,
                        )
                        record["bad_case_command"] = bad_case_command
                        returncode = run_command(bad_case_command, dry_run=args.dry_run)
                        record["bad_case_returncode"] = returncode
                        if returncode != 0:
                            raise SystemExit(returncode)

            if not args.skip_eval:
                analysis_command = [
                    sys.executable,
                    "tools/analyze_rg_flux_moe_router_ablation.py",
                    "--checkpoint_root",
                    str(checkpoint_root),
                    "--bootstrap_samples",
                    str(args.bootstrap_samples),
                    "--seed",
                    str(args.seed),
                ]
                returncode = run_command(analysis_command, dry_run=args.dry_run)
                if returncode != 0:
                    raise SystemExit(returncode)

            if fingerprint_before is not None:
                fingerprint_after = checkpoint_fingerprint(adapter_dir)
                if fingerprint_before != fingerprint_after:
                    raise RuntimeError(
                        f"Checkpoint changed during Router ablation: {adapter_dir}"
                    )
                manifest.setdefault("checkpoint_fingerprints", {})[checkpoint_dir.name] = fingerprint_after
    finally:
        manifest_path = output_root / "router_ablation_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        print(f"[router_ablation] saved manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
