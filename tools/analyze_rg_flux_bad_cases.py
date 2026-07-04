import argparse
import csv
import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.rg_sr_metrics import IMAGE_EXTENSIONS, LOWER_BETTER_FALLBACKS, parse_name_path


def load_metric_rows(metrics_csv):
    metrics_csv = Path(metrics_csv)
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in metrics CSV: {metrics_csv}")
    required = {"dataset", "filename", "path"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Metrics CSV {metrics_csv} is missing required columns: {sorted(missing)}")
    return rows


def load_metric_directions(summary_json=None):
    if not summary_json:
        return {}
    summary_json = Path(summary_json)
    with summary_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    directions = payload.get("metric_directions", {})
    if not isinstance(directions, dict):
        raise ValueError(f"summary_scores.json metric_directions must be an object: {summary_json}")
    return {str(key): str(value) for key, value in directions.items()}


def metric_direction(metric, directions):
    direction = directions.get(metric)
    if direction in {"higher_better", "lower_better"}:
        return direction
    return "lower_better" if metric.lower() in LOWER_BETTER_FALLBACKS else "higher_better"


def metric_value(row, metric):
    if metric not in row:
        raise ValueError(f"Metric '{metric}' is not present in per-image CSV")
    try:
        return float(row[metric])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for metric '{metric}' in row {row}") from exc


def normalized_values(rows, metric):
    values = [metric_value(row, metric) for row in rows]
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def attach_joint_badness(rows, metrics, directions):
    rows = [dict(row) for row in rows]
    metric_badness = {}
    for metric in metrics:
        direction = metric_direction(metric, directions)
        norm = normalized_values(rows, metric)
        if direction == "higher_better":
            badness_values = [1.0 - value for value in norm]
        else:
            badness_values = norm
        metric_badness[metric] = badness_values

    for index, row in enumerate(rows):
        badness_values = []
        for metric in metrics:
            badness = metric_badness[metric][index]
            row[f"{metric}_badness"] = f"{badness:.6f}"
            badness_values.append(badness)
        row["joint_badness"] = f"{(sum(badness_values) / len(badness_values)):.6f}"
    return rows


def rank_single_metric(rows, metric, directions, worst_k):
    direction = metric_direction(metric, directions)
    reverse = direction == "lower_better"
    ranked = sorted(rows, key=lambda row: metric_value(row, metric), reverse=reverse)
    return [dict(row) for row in ranked[:worst_k]]


def rank_joint_mean(rows, metrics, directions, worst_k):
    rows_with_badness = attach_joint_badness(rows, metrics, directions)
    ranked = sorted(rows_with_badness, key=lambda row: float(row["joint_badness"]), reverse=True)
    return ranked[:worst_k]


def build_lq_indexes(lq_dirs):
    indexes = {}
    for dataset, directory in (lq_dirs or {}).items():
        directory = Path(directory)
        index = {}
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name not in index:
                    index[path.name] = path
        indexes[dataset] = index
    return indexes


def find_lq_path(row, lq_indexes):
    dataset = row.get("dataset", "")
    filename = row.get("filename", "")
    return lq_indexes.get(dataset, {}).get(filename)


def placeholder_image(size, text):
    image = Image.new("RGB", size, color=(210, 210, 210))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = max((size[0] - (bbox[2] - bbox[0])) // 2, 0)
    y = max((size[1] - (bbox[3] - bbox[1])) // 2, 0)
    draw.text((x, y), text, fill=(40, 40, 40), font=font)
    return image


def draw_text_lines(draw, xy, lines, fill=(0, 0, 0)):
    font = ImageFont.load_default()
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += 14


def compose_comparison_image(row, lq_path, output_path, metrics, title_metric=None):
    sr_path = Path(row["path"])
    with Image.open(sr_path) as sr_image:
        sr = sr_image.convert("RGB")
    if lq_path is not None and Path(lq_path).exists():
        with Image.open(lq_path) as lq_image:
            lq = lq_image.convert("RGB").resize(sr.size, Image.Resampling.BICUBIC)
        lq_found = True
    else:
        lq = placeholder_image(sr.size, "LQ unavailable")
        lq_found = False

    label_h = 24
    caption_h = 58
    width = sr.width * 2
    height = label_h + sr.height + caption_h
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(lq, (0, label_h))
    canvas.paste(sr, (sr.width, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, label_h), fill=(245, 245, 245))
    draw.text((8, 6), "LQ upscaled", fill=(0, 0, 0), font=ImageFont.load_default())
    draw.text((sr.width + 8, 6), "SR output", fill=(0, 0, 0), font=ImageFont.load_default())

    score_parts = []
    if "joint_badness" in row:
        score_parts.append(f"joint_badness={row['joint_badness']}")
    for metric in metrics:
        if metric in row:
            score_parts.append(f"{metric}={float(row[metric]):.6f}")
    caption = [
        f"rank={row.get('rank', '')} dataset={row.get('dataset', '')} filename={row.get('filename', '')}",
        " ".join(score_parts),
        f"lq_found={str(lq_found).lower()} metric={title_metric or ''}",
    ]
    draw_text_lines(draw, (8, label_h + sr.height + 6), caption)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return lq_found


def csv_fieldnames(rows, metrics):
    base = ["rank", "dataset", "filename", "sr_path", "lq_path", "lq_found"]
    extra = []
    if rows and "joint_badness" in rows[0]:
        extra.append("joint_badness")
    for metric in metrics:
        extra.append(metric)
        badness_key = f"{metric}_badness"
        if rows and badness_key in rows[0]:
            extra.append(badness_key)
    return base + extra + ["comparison_path"]


def write_worst_cases_csv(path, rows, metrics):
    fieldnames = csv_fieldnames(rows, metrics)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_html_report(path, rows, metrics, title):
    path = Path(path)
    rel_images = [
        Path(row["comparison_path"]).relative_to(path.parent).as_posix()
        if Path(row["comparison_path"]).is_absolute()
        else Path(row["comparison_path"]).as_posix()
        for row in rows
    ]
    blocks = []
    for row, rel_image in zip(rows, rel_images):
        score_text = []
        if "joint_badness" in row:
            score_text.append(f"joint_badness={row['joint_badness']}")
        for metric in metrics:
            if metric in row:
                score_text.append(f"{metric}={float(row[metric]):.6f}")
        blocks.append(
            "<section>"
            f"<h3>#{html.escape(str(row['rank']))} {html.escape(row.get('dataset', ''))} / {html.escape(row.get('filename', ''))}</h3>"
            f"<p>{html.escape(' | '.join(score_text))}</p>"
            f"<img src=\"{html.escape(rel_image)}\" style=\"max-width:100%;height:auto;\" />"
            "</section>"
        )
    content = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;}section{margin-bottom:28px;}img{border:1px solid #ddd;}</style>"
        "</head><body>"
        f"<h1>{html.escape(title)}</h1>"
        + "\n".join(blocks)
        + "</body></html>"
    )
    path.write_text(content, encoding="utf-8")


def image_output_name(rank, filename):
    stem = Path(filename).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"worst_{rank:04d}_{safe_stem}.png"


def write_case_outputs(rows, metrics, output_dir, lq_indexes, title):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    output_rows = []
    for rank, row in enumerate(rows, start=1):
        row = dict(row)
        row["rank"] = rank
        lq_path = find_lq_path(row, lq_indexes)
        comparison_path = images_dir / image_output_name(rank, row["filename"])
        lq_found = compose_comparison_image(row, lq_path, comparison_path, metrics, title_metric=title)
        row["sr_path"] = row["path"]
        row["lq_path"] = str(lq_path) if lq_path else ""
        row["lq_found"] = "true" if lq_found else "false"
        row["comparison_path"] = str(comparison_path)
        output_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_worst_cases_csv(output_dir / "worst_cases.csv", output_rows, metrics)
    write_html_report(output_dir / "report.html", output_rows, metrics, title)
    return output_rows


def joint_output_name(metrics):
    return "joint_mean_" + "_".join(metrics)


def run_analysis(metrics_csv, summary_json, metrics, mode, worst_k, lq_dirs, output_dir):
    if not metrics:
        raise ValueError("At least one metric is required")
    if worst_k <= 0:
        raise ValueError("worst_k must be positive")
    if mode not in {"separate", "joint_mean"}:
        raise ValueError(f"Unsupported mode: {mode}")

    rows = load_metric_rows(metrics_csv)
    directions = load_metric_directions(summary_json)
    lq_indexes = build_lq_indexes(lq_dirs)
    output_dir = Path(output_dir)
    written = []

    if mode == "joint_mean":
        ranked = rank_joint_mean(rows, metrics, directions, worst_k)
        target_dir = output_dir / joint_output_name(metrics)
        written.append(write_case_outputs(ranked, metrics, target_dir, lq_indexes, joint_output_name(metrics)))
        return written

    multi_metric = len(metrics) > 1
    for metric in metrics:
        ranked = rank_single_metric(rows, metric, directions, worst_k)
        target_dir = output_dir / metric if multi_metric else output_dir
        written.append(write_case_outputs(ranked, [metric], target_dir, lq_indexes, metric))
    return written


def parse_metrics(args):
    if args.metric and args.metrics:
        raise ValueError("--metric cannot be combined with --metrics")
    if args.metric:
        return [args.metric]
    if args.metrics:
        return list(args.metrics)
    raise ValueError("--metric or --metrics is required")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Mine low-scoring RG-FLUX-SR cases and render LQ/SR comparison images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics_csv", type=Path, required=True, help="metrics/per_image_scores.csv")
    parser.add_argument("--summary_json", type=Path, default=None, help="metrics/summary_scores.json")
    parser.add_argument("--metric", default=None, help="Single metric to analyze, e.g. maniqa")
    parser.add_argument("--metrics", nargs="+", default=None, help="Multiple metrics to analyze")
    parser.add_argument("--mode", choices=["separate", "joint_mean"], default="separate")
    parser.add_argument("--worst_k", type=int, default=50)
    parser.add_argument("--lq_dirs", nargs="*", default=[], help="Dataset LQ dirs in name=path format")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    metrics = parse_metrics(args)
    lq_dirs = parse_name_path(args.lq_dirs, "--lq_dirs") if args.lq_dirs else {}
    run_analysis(
        metrics_csv=args.metrics_csv,
        summary_json=args.summary_json,
        metrics=metrics,
        mode=args.mode,
        worst_k=args.worst_k,
        lq_dirs=lq_dirs,
        output_dir=args.output_dir,
    )
    print(f"Saved bad case analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
