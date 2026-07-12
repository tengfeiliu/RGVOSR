import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS = ("fixed", "suggestion", "iqa", "iqa_suggestion")


def normalize_variant(variant):
    normalized = str(variant).strip().lower().replace("-", "_")
    if normalized not in DEFAULT_VARIANTS:
        raise ValueError(
            f"Unsupported ablation variant '{variant}'. Expected one of: {', '.join(DEFAULT_VARIANTS)}"
        )
    return normalized


def build_variant_command(variant, pipeline_args):
    variant = normalize_variant(variant)
    pipeline_args = list(pipeline_args)
    if "--prompt_variant" in pipeline_args:
        raise ValueError("Pass prompt variants through --variants, not through the nested pipeline arguments.")
    return [
        sys.executable,
        "tools/run_rg_flux_pipeline.py",
        "--prompt_variant",
        variant,
        *pipeline_args,
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run fixed, suggestion, IQA, and IQA+suggestion RG-FLUX-SR ablations sequentially."
    )
    parser.add_argument("--variants", nargs="+", choices=DEFAULT_VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to tools/run_rg_flux_pipeline.py. Prefix them with --.",
    )
    args = parser.parse_args(argv)
    if args.pipeline_args and args.pipeline_args[0] == "--":
        args.pipeline_args = args.pipeline_args[1:]
    if not args.pipeline_args:
        raise ValueError("Pipeline arguments are required after '--'.")
    return args


def main(argv=None):
    args = parse_args(argv)
    for variant in args.variants:
        command = build_variant_command(variant, args.pipeline_args)
        print(f"[prompt-ablation] starting variant={variant}", flush=True)
        print("[prompt-ablation] running:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            print(
                f"[prompt-ablation] variant={variant} failed with return code {completed.returncode}; stopping.",
                flush=True,
            )
            return completed.returncode
        print(f"[prompt-ablation] completed variant={variant}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
