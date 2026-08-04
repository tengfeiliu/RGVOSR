import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_rg_flux_pipeline import (
    DEFAULT_METRICS,
    PROMPT_VARIANTS,
    apply_config_prompt_defaults,
    build_bad_case_command,
    build_eval_command,
    build_inference_command,
    build_train_command,
    cfg,
    cfg_bool,
    checkpoint_artifact_paths,
    load_yaml,
    parse_dataset_dirs,
    planned_checkpoint_dir as planned_single_checkpoint_dir,
    resolve_checkpoint_dir,
    resolve_experiment_name,
    resolve_skip_train_config_path,
    run_command,
    write_run_summary,
    write_yaml,
)


def resolve_single_lora_checkpoint(
    single_lora_checkpoint=None,
    single_lora_run_dir=None,
    single_lora_checkpoint_step=None,
):
    if single_lora_checkpoint and single_lora_run_dir:
        raise ValueError("--single_lora_checkpoint cannot be combined with --single_lora_run_dir.")
    if single_lora_checkpoint:
        checkpoint = Path(single_lora_checkpoint)
        if (checkpoint / "rg_flux_adapters").exists():
            checkpoint = checkpoint / "rg_flux_adapters"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Single-LoRA checkpoint does not exist: {checkpoint}")
        return checkpoint
    if not single_lora_run_dir:
        raise ValueError("--single_lora_checkpoint or --single_lora_run_dir is required unless --skip_stage1 is used.")
    step = single_lora_checkpoint_step or "latest"
    checkpoint_dir = resolve_checkpoint_dir(single_lora_run_dir, step)
    return checkpoint_dir / "rg_flux_adapters"


def _load_config_file(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return load_yaml(path)


def _sync_text_and_condition_overrides(config, args):
    if args.text_encoding_mode is not None:
        config.setdefault("text_encoding", {})["mode"] = args.text_encoding_mode
    if args.text_embedding_cache is not None:
        config.setdefault("text_encoding", {})["cache_dir"] = args.text_embedding_cache
    condition = config.setdefault("condition", {})
    prompt_variant = getattr(args, "prompt_variant", None)
    if prompt_variant is not None:
        condition["prompt_variant"] = prompt_variant
        condition["use_prompt"] = prompt_variant != "fixed"
        condition["use_suggestions"] = prompt_variant in {"suggestion", "iqa_suggestion"}
        prompt_schedule = condition.get("prompt_schedule")
        if isinstance(prompt_schedule, dict) and cfg_bool(config, "condition.prompt_schedule.enabled", False):
            prompt_schedule["after_variant"] = prompt_variant
    if args.use_prompt is not None:
        condition["use_prompt"] = bool(args.use_prompt)
    if args.use_suggestions is not None:
        condition["use_suggestions"] = bool(args.use_suggestions)
    if args.use_degradation_vector is not None:
        condition["use_degradation_vector"] = bool(args.use_degradation_vector)
    moe = config.setdefault("model", {}).setdefault("lora_moe", {})
    router_input_mode = getattr(args, "router_input_mode", None)
    teacher_routing_enabled = getattr(args, "teacher_routing_enabled", None)
    router_teacher_weight = getattr(args, "router_teacher_weight", None)
    stage_label = getattr(args, "stage_label", None)
    if router_input_mode is not None:
        moe["router_input_mode"] = str(router_input_mode)
    if teacher_routing_enabled is not None:
        moe["teacher_routing_enabled"] = bool(teacher_routing_enabled)
    semantic_prototype_init = getattr(args, "semantic_prototype_init", None)
    if semantic_prototype_init is not None:
        moe["semantic_prototype_init"] = bool(semantic_prototype_init)
    if router_teacher_weight is not None:
        config.setdefault("loss", {})["router_teacher_weight"] = float(
            router_teacher_weight
        )
    if stage_label:
        config.setdefault("training", {})["suffix"] = str(stage_label)


def create_moe_runtime_config(moe_config_path, args, now=None):
    runtime_config = load_yaml(moe_config_path)
    runtime_config.setdefault("model", {})
    runtime_config["model"]["flux_backend"] = "flux2_klein"
    runtime_config["model"]["lora_backend"] = "moe"
    _sync_text_and_condition_overrides(runtime_config, args)
    init_seed = getattr(args, "init_seed", None)
    if init_seed is None:
        init_seed = int(cfg(runtime_config, "training.seed", 42))
    runtime_config.setdefault("_runtime", {})["moe_init_seed"] = int(init_seed)
    expert_init_seed = getattr(args, "expert_init_seed", None)
    if expert_init_seed is None:
        expert_init_seed = init_seed
    runtime_config["_runtime"]["moe_expert_init_seed"] = int(expert_init_seed)

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
    stage1_output = run_dir / "stage1_init" / "rg_flux_adapters"
    runtime_config["training"]["resume_ckpt"] = str(stage1_output)
    runtime_config["training"]["resume_training_state"] = False
    write_yaml(runtime_path, runtime_config)
    write_yaml(run_dir / "configs" / "source_config.yaml", load_yaml(moe_config_path))
    write_yaml(run_dir / "configs" / "runtime_config.yaml", runtime_config)
    return runtime_config, run_dir, runtime_path, stage1_output


def build_stage1_command(args, runtime_config_path, single_lora_checkpoint, stage1_output):
    runtime_config = _load_config_file(runtime_config_path)
    init_seed = getattr(args, "init_seed", None)
    if init_seed is None:
        init_seed = int(cfg(runtime_config, "_runtime.moe_init_seed", 42))
    expert_init_seed = getattr(args, "expert_init_seed", None)
    if expert_init_seed is None:
        expert_init_seed = int(cfg(runtime_config, "_runtime.moe_expert_init_seed", init_seed))
    cmd = [
        sys.executable,
        "tools/init_flux2_lora_moe.py",
        "--config",
        str(runtime_config_path),
        "--single_lora_checkpoint",
        str(single_lora_checkpoint),
        "--output",
        str(stage1_output),
        "--prototype_num_samples",
        str(args.prototype_num_samples),
        "--device",
        str(args.init_device),
        "--dtype",
        str(args.dtype),
        "--perturb_scale",
        str(args.perturb_scale),
        "--seed",
        str(init_seed),
        "--expert_init_seed",
        str(expert_init_seed),
    ]
    return cmd


def validate_stage1_output(stage1_output):
    stage1_output = Path(stage1_output)
    if not stage1_output.exists():
        raise FileNotFoundError(f"Stage1 MoE adapter output does not exist: {stage1_output}")
    moe_state = stage1_output / "flux2_klein_lora_moe_state.pt"
    if not moe_state.exists():
        raise FileNotFoundError(f"Stage1 MoE adapter output is missing {moe_state.name}: {stage1_output}")


def planned_checkpoint_dir(run_dir, checkpoint_step):
    return planned_single_checkpoint_dir(run_dir, checkpoint_step)


def write_moe_pipeline_manifest(
    run_dir,
    runtime_config_path,
    single_lora_checkpoint,
    stage1_output,
    checkpoint_steps,
    records,
    stage1_returncode=None,
    train_returncode=None,
    stage1_command=None,
    train_command=None,
    stage1_seed=None,
    expert_init_seed=None,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "moe_pipeline_manifest.json"
    payload = {
        "runtime_config_path": str(runtime_config_path) if runtime_config_path else None,
        "run_dir": str(run_dir),
        "single_lora_checkpoint": str(single_lora_checkpoint) if single_lora_checkpoint is not None else None,
        "stage1_output": str(stage1_output) if stage1_output is not None else None,
        "checkpoint_steps": list(checkpoint_steps),
        "stage1_command": stage1_command,
        "stage1_returncode": stage1_returncode,
        "stage1_seed": stage1_seed,
        "expert_init_seed": expert_init_seed,
        "train_command": train_command,
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
        pipeline_type="moe_lora",
        extra={
            "single_lora_checkpoint": payload["single_lora_checkpoint"],
            "stage1_output": payload["stage1_output"],
            "stage1_returncode": stage1_returncode,
            "train_returncode": train_returncode,
        },
    )
    return manifest_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run FLUX.2-klein LoRA-MoE Stage1 -> train -> inference -> metrics.")
    parser.add_argument("--moe_config", default=None, help="MoE training YAML config. Required unless --skip_train.")
    parser.add_argument("--single_lora_checkpoint", default=None, help="Stage0 Single-LoRA adapter or checkpoint directory.")
    parser.add_argument("--single_lora_run_dir", default=None, help="Stage0 Single-LoRA run directory.")
    parser.add_argument("--single_lora_checkpoint_step", default="latest", help="Stage0 checkpoint step, e.g. 32000 or latest.")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage1 init. Intended with --skip_train.")
    parser.add_argument("--skip_train", action="store_true", help="Skip MoE training and evaluate an existing --moe_run_dir.")
    parser.add_argument("--moe_run_dir", default=None, help="Existing MoE run directory, required with --skip_train.")
    parser.add_argument("--accelerate_config", default=None, help="Accelerate config used for MoE training.")
    parser.add_argument("--num_processes", type=int, default=None, help="Optional accelerate --num_processes.")
    parser.add_argument("--checkpoint_steps", nargs="+", required=True, help="MoE checkpoint steps, e.g. 20000 40000 latest.")
    parser.add_argument("--dataset_dirs", nargs="+", required=True, help="Shared inference/eval datasets as name=folder.")
    parser.add_argument(
        "--inference_output_root",
        default=None,
        help=(
            "Optional root for legacy inference outputs. If omitted, outputs are written under "
            "moe_run_dir/inference/checkpoint-XXXXXXXX."
        ),
    )
    parser.add_argument("--prototype_num_samples", type=int, default=128)
    parser.add_argument("--perturb_scale", type=float, default=0.3)
    parser.add_argument(
        "--router_input_mode",
        choices=["prompt_lr", "prompt_only", "condition8", "condition8_timestep"],
        default=None,
        help="Override model.lora_moe.router_input_mode for staged ablations.",
    )
    parser.add_argument(
        "--teacher_routing_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable structured IQA/Suggestion teacher routing.",
    )
    parser.add_argument("--router_teacher_weight", type=float, default=None)
    parser.add_argument(
        "--semantic_prototype_init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use confidence-weighted expert semantics instead of unsupervised KMeans prototypes.",
    )
    parser.add_argument(
        "--stage_label",
        default=None,
        help="Override training suffix so staged runs are easy to distinguish.",
    )
    parser.add_argument("--init_device", default="cuda")
    parser.add_argument(
        "--init_seed",
        type=int,
        default=None,
        help="Stage1 initialization seed; defaults to training.seed from the runtime config.",
    )
    parser.add_argument(
        "--expert_init_seed",
        type=int,
        default=None,
        help="Independent routed-A perturbation seed; defaults to init_seed.",
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
    parser.add_argument(
        "--full_frame_inference",
        action="store_true",
        help="Use arbitrary-size full-frame inputs for inference without changing the pre-cropped training config.",
    )
    parser.add_argument("--use_prompt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_suggestions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prompt_variant", choices=PROMPT_VARIANTS, default=None)
    parser.add_argument("--use_degradation_vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--metric_device", default="cpu")
    parser.add_argument("--run_bad_cases", action="store_true", help="Run bad case analysis after metrics.")
    parser.add_argument("--bad_case_metrics", nargs="+", default=["clipiqa", "maniqa", "musiq"])
    parser.add_argument("--bad_case_mode", choices=["separate", "joint_mean"], default="separate")
    parser.add_argument("--bad_case_worst_k", type=int, default=50)
    parser.add_argument("--bad_case_font_size", type=int, default=40)
    parser.add_argument("--dry_run_train", action="store_true", help="Pass --dry_run to train_rg_flux_sr.py.")
    parser.add_argument("--dry_run_pipeline", action="store_true", help="Write command plan without running child commands.")
    return parser


def parse_args(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.single_lora_checkpoint and args.single_lora_run_dir:
        raise ValueError("--single_lora_checkpoint cannot be combined with --single_lora_run_dir.")
    if args.skip_train and not args.moe_run_dir:
        raise ValueError("--moe_run_dir is required with --skip_train.")
    if not args.skip_train and not args.moe_config:
        raise ValueError("--moe_config is required unless --skip_train.")
    if not args.skip_stage1 and not (args.single_lora_checkpoint or args.single_lora_run_dir):
        raise ValueError("--single_lora_checkpoint or --single_lora_run_dir is required unless --skip_stage1.")
    if args.skip_stage1 and not args.skip_train:
        raise ValueError("--skip_stage1 is only supported with --skip_train in this pipeline.")
    parse_dataset_dirs(args.dataset_dirs)
    return args


def _run_inference_and_eval_for_steps(args, run_dir, runtime_config_path, checkpoint_steps, records, strict_checkpoints=True):
    for step in checkpoint_steps:
        checkpoint_dir = (
            resolve_checkpoint_dir(run_dir, step)
            if strict_checkpoints
            else planned_checkpoint_dir(run_dir, step)
        )
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
                raise SystemExit(record["inference_returncode"])
            record["eval_returncode"] = run_command(eval_cmd)
            if record["eval_returncode"] != 0:
                records.append(record)
                raise SystemExit(record["eval_returncode"])
            if args.run_bad_cases:
                record["bad_case_returncode"] = run_command(bad_case_cmd)
                if record["bad_case_returncode"] != 0:
                    records.append(record)
                    raise SystemExit(record["bad_case_returncode"])
        records.append(record)


def main(argv=None):
    args = parse_args(argv)
    records = []
    stage1_returncode = None
    train_returncode = None
    stage1_command = None
    train_command = None
    single_lora_checkpoint = None
    stage1_output = None
    stage1_seed = None
    expert_init_seed = None

    if args.skip_train:
        run_dir = Path(args.moe_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"MoE run directory does not exist: {run_dir}")
        runtime_config_path = resolve_skip_train_config_path(run_dir, args.moe_config)
        runtime_config = _load_config_file(runtime_config_path)
        stage1_output = run_dir / "stage1_init" / "rg_flux_adapters"
        stage1_seed = cfg(runtime_config, "_runtime.moe_init_seed", None)
        expert_init_seed = cfg(runtime_config, "_runtime.moe_expert_init_seed", None)
    else:
        runtime_config, run_dir, runtime_config_path, stage1_output = create_moe_runtime_config(args.moe_config, args)
        stage1_seed = int(cfg(runtime_config, "_runtime.moe_init_seed", 42))
        expert_init_seed = int(cfg(runtime_config, "_runtime.moe_expert_init_seed", stage1_seed))
        single_lora_checkpoint = resolve_single_lora_checkpoint(
            single_lora_checkpoint=args.single_lora_checkpoint,
            single_lora_run_dir=args.single_lora_run_dir,
            single_lora_checkpoint_step=args.single_lora_checkpoint_step,
        )
        stage1_command = build_stage1_command(args, runtime_config_path, single_lora_checkpoint, stage1_output)
        if not args.dry_run_pipeline:
            stage1_returncode = run_command(stage1_command)
            if stage1_returncode != 0:
                write_moe_pipeline_manifest(
                    run_dir,
                    runtime_config_path,
                    single_lora_checkpoint,
                    stage1_output,
                    args.checkpoint_steps,
                    records,
                    stage1_returncode=stage1_returncode,
                    train_returncode=train_returncode,
                    stage1_command=stage1_command,
                    train_command=train_command,
                    stage1_seed=stage1_seed,
                    expert_init_seed=expert_init_seed,
                )
                raise SystemExit(stage1_returncode)
            validate_stage1_output(stage1_output)

        train_command = build_train_command(args, runtime_config_path)
        if not args.dry_run_pipeline:
            train_returncode = run_command(train_command)
            if train_returncode != 0:
                write_moe_pipeline_manifest(
                    run_dir,
                    runtime_config_path,
                    single_lora_checkpoint,
                    stage1_output,
                    args.checkpoint_steps,
                    records,
                    stage1_returncode=stage1_returncode,
                    train_returncode=train_returncode,
                    stage1_command=stage1_command,
                    train_command=train_command,
                    stage1_seed=stage1_seed,
                    expert_init_seed=expert_init_seed,
                )
                raise SystemExit(train_returncode)

    _sync_text_and_condition_overrides(runtime_config, args)
    apply_config_prompt_defaults(args, runtime_config)

    try:
        _run_inference_and_eval_for_steps(
            args,
            run_dir,
            runtime_config_path,
            args.checkpoint_steps,
            records,
            strict_checkpoints=not args.dry_run_pipeline,
        )
    finally:
        manifest_path = write_moe_pipeline_manifest(
            run_dir,
            runtime_config_path,
            single_lora_checkpoint,
            stage1_output,
            args.checkpoint_steps,
            records,
            stage1_returncode=stage1_returncode,
            train_returncode=train_returncode,
            stage1_command=stage1_command,
            train_command=train_command,
            stage1_seed=stage1_seed,
            expert_init_seed=expert_init_seed,
        )
        print(f"[moe_pipeline] saved manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
