import argparse
import json
from pathlib import Path

from metrics.rg_sr_metrics import DEFAULT_OMGSR_METRICS, evaluate_dataset_dirs, parse_expected_counts, parse_name_path


def load_inference_manifest(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Inference manifest must be a JSON object: {path}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Inference manifest is missing non-empty datasets list: {path}")
    dataset_dirs = {}
    for record in datasets:
        if not isinstance(record, dict):
            raise ValueError(f"Inference manifest dataset entry must be an object: {record}")
        name = str(record.get("name") or "").strip()
        output_dir = record.get("output_dir")
        if not name or not output_dir:
            raise ValueError(f"Inference manifest dataset entry requires name and output_dir: {record}")
        if name in dataset_dirs:
            raise ValueError(f"Duplicate dataset name in inference manifest: {name}")
        dataset_dirs[name] = Path(output_dir)
    return manifest, dataset_dirs


def resolve_evaluation_inputs(args):
    if args.inference_manifest:
        manifest, dataset_dirs = load_inference_manifest(args.inference_manifest)
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(manifest["output_dir"]) / "metrics"
        return dataset_dirs, Path(output_dir)

    if not args.dataset_dirs:
        raise ValueError("--dataset_dirs is required unless --inference_manifest is used.")
    if args.output_dir is None:
        raise ValueError("--output_dir is required unless --inference_manifest is used.")
    return parse_name_path(args.dataset_dirs, "--dataset_dirs"), Path(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RG-FLUX-SR output images with OMGSR PyIQA no-reference metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        default=None,
        help="Dataset result directories in name=path format, e.g. smoke=outputs/smoke",
    )
    parser.add_argument(
        "--inference_manifest",
        type=Path,
        default=None,
        help="Optional inference_manifest.json written by inference_rg_flux_sr.py.",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for CSV/JSON metric outputs.")
    parser.add_argument("--device", default="cpu", help="Torch device used by PyIQA metrics.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_OMGSR_METRICS, help="PyIQA metric names.")
    parser.add_argument("--expected_counts", nargs="+", default=None, help="Optional checks in name=count format.")
    args = parser.parse_args()
    if args.inference_manifest is None and not args.dataset_dirs:
        parser.error("--dataset_dirs is required unless --inference_manifest is used.")
    if args.inference_manifest is None and args.output_dir is None:
        parser.error("--output_dir is required unless --inference_manifest is used.")
    return args


def main():
    args = parse_args()
    dataset_dirs, output_dir = resolve_evaluation_inputs(args)
    summary = evaluate_dataset_dirs(
        dataset_dirs=dataset_dirs,
        output_dir=output_dir,
        metrics=args.metrics,
        device=args.device,
        expected_counts=parse_expected_counts(args.expected_counts),
    )
    print("dataset,metric,direction,mean,std,count")
    for row in summary["summary"]:
        metric = row["metric"]
        print(
            f"{row['dataset']},{metric},{summary['metric_directions'][metric]},"
            f"{row['mean']:.6f},{row['std']:.6f},{row['count']}"
        )
    print(f"Saved metric outputs to: {output_dir}")


if __name__ == "__main__":
    main()
