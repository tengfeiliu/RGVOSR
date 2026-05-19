import argparse
from pathlib import Path

from metrics.rg_sr_metrics import DEFAULT_OMGSR_METRICS, evaluate_dataset_dirs, parse_expected_counts, parse_name_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RG-FLUX-SR output images with OMGSR PyIQA no-reference metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        required=True,
        help="Dataset result directories in name=path format, e.g. smoke=outputs/smoke",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for CSV/JSON metric outputs.")
    parser.add_argument("--device", default="cpu", help="Torch device used by PyIQA metrics.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_OMGSR_METRICS, help="PyIQA metric names.")
    parser.add_argument("--expected_counts", nargs="+", default=None, help="Optional checks in name=count format.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = evaluate_dataset_dirs(
        dataset_dirs=parse_name_path(args.dataset_dirs, "--dataset_dirs"),
        output_dir=args.output_dir,
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
    print(f"Saved metric outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
