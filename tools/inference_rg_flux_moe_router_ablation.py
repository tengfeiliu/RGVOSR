import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference_rg_flux_sr as base_inference  # noqa: E402
from tools.moe_router_ablation import (  # noqa: E402
    ROUTER_ABLATION_MODES,
    RouterAblationController,
)


def build_arg_parser():
    parser = base_inference.build_arg_parser()
    parser.description = (
        "Read-only LoRA-MoE Router ablation inference. Existing inference entrypoints "
        "and checkpoint files are not modified."
    )
    parser.add_argument(
        "--router_ablation_mode",
        choices=ROUTER_ABLATION_MODES,
        required=True,
    )
    parser.add_argument("--fixed_weights", nargs="+", type=float, default=None)
    parser.add_argument("--onehot_expert", type=int, default=None)
    parser.add_argument("--shuffle_reference_trace", default=None)
    parser.add_argument("--condition_shuffle_seed", type=int, default=3407)
    parser.add_argument(
        "--router_trace_path",
        default=None,
        help="Defaults to <output_dir>/router_trace.jsonl.",
    )
    parser.add_argument(
        "--allow_nonempty_output",
        action="store_true",
        help="Allow writing into a non-empty output directory. Disabled by default for paired reproducibility.",
    )
    return parser


def _validate_mode_args(args):
    if args.router_ablation_mode == "fixed_mean" and not args.fixed_weights:
        raise ValueError("fixed_mean requires --fixed_weights.")
    if args.router_ablation_mode != "fixed_mean" and args.fixed_weights:
        raise ValueError("--fixed_weights is only valid with fixed_mean.")
    if args.router_ablation_mode == "onehot" and args.onehot_expert is None:
        raise ValueError("onehot requires --onehot_expert.")
    if args.router_ablation_mode != "onehot" and args.onehot_expert is not None:
        raise ValueError("--onehot_expert is only valid with onehot.")
    if args.router_ablation_mode == "shuffle_condition8" and not args.shuffle_reference_trace:
        raise ValueError("shuffle_condition8 requires --shuffle_reference_trace.")


def _write_ablation_manifest(inference_manifest_path, metadata):
    inference_manifest_path = Path(inference_manifest_path)
    with inference_manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["router_ablation"] = metadata
    with inference_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    _validate_mode_args(args)
    resolved = base_inference.resolve_inference_run(args)
    output_dir = Path(resolved["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_nonempty_output:
        raise FileExistsError(
            f"Router ablation output directory is not empty: {output_dir}. "
            "Use a fresh directory or pass --allow_nonempty_output explicitly."
        )
    trace_path = Path(args.router_trace_path or output_dir / "router_trace.jsonl")
    inference_manifest_path = output_dir / "inference_manifest.json"

    controller = RouterAblationController(
        mode=args.router_ablation_mode,
        num_steps=args.num_inference_steps,
        fixed_weights=args.fixed_weights,
        onehot_expert=args.onehot_expert,
        shuffle_reference_trace=args.shuffle_reference_trace,
        shuffle_seed=args.condition_shuffle_seed,
    )
    original_builder = base_inference.build_rg_flux_artist

    def build_instrumented_artist(config):
        artist = original_builder(config)
        controller.install(artist)
        return artist

    base_inference.build_rg_flux_artist = build_instrumented_artist
    succeeded = False
    try:
        base_inference.main(args)
        succeeded = True
    finally:
        base_inference.build_rg_flux_artist = original_builder
        controller.uninstall()

    if succeeded:
        metadata = controller.finalize(
            trace_path=trace_path,
            inference_manifest_path=inference_manifest_path,
        )
        metadata.update(
            {
                "condition_shuffle_seed": (
                    args.condition_shuffle_seed
                    if args.router_ablation_mode == "shuffle_condition8"
                    else None
                ),
                "shuffle_reference_trace": (
                    str(args.shuffle_reference_trace)
                    if args.router_ablation_mode == "shuffle_condition8"
                    else None
                ),
                "compatibility": (
                    "learned_top2 records the unmodified Router output; all other modes are "
                    "isolated to this diagnostic process"
                ),
            }
        )
        _write_ablation_manifest(inference_manifest_path, metadata)
        print(f"[router_ablation] saved trace: {trace_path}", flush=True)
        print(f"[router_ablation] updated manifest: {inference_manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
