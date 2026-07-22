import argparse
import copy
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
from models.prompt_builder import DEFAULT_SR_PROMPT, build_sr_prompt


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


def _normalized_path(value):
    return str(value or "").replace("\\", "/")


def _profile_lookup_keys(value, dataset=None):
    if not value:
        return []
    normalized = _normalized_path(value)
    name = Path(normalized).name
    keys = [f"path:{normalized}"]
    if dataset:
        keys.append(f"dataset:{dataset}/{name}")
    keys.append(f"name:{name}")
    return keys


def load_profile_index(jsonl_path=None, target_paths=None, target_names=None):
    if not jsonl_path:
        return {}
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Prompt JSONL does not exist: {jsonl_path}")

    target_paths = {_normalized_path(path) for path in (target_paths or []) if path}
    target_names = {str(name) for name in (target_names or []) if name}
    index = {}
    ambiguous_names = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_paths = [
                record.get(field)
                for field in ("lq_path", "image_path", "path", "hq_path")
                if record.get(field)
            ]
            if target_paths or target_names:
                matches_path = any(_normalized_path(value) in target_paths for value in record_paths)
                matches_name = any(Path(_normalized_path(value)).name in target_names for value in record_paths)
                if not matches_path and not matches_name:
                    continue
            unipercept_raw = record.get("unipercept_raw")
            profile = unipercept_raw.get("profile") if isinstance(unipercept_raw, dict) else None
            if not isinstance(profile, dict):
                continue
            dataset = record.get("dataset_name") or record.get("dataset")
            for field in ("lq_path", "image_path", "path", "hq_path"):
                value = record.get(field)
                for key in _profile_lookup_keys(value, dataset=dataset):
                    if key.startswith("name:") and key in index and index[key] != profile:
                        ambiguous_names.add(key)
                    else:
                        index[key] = profile
    for key in ambiguous_names:
        index.pop(key, None)
    return index


def _resolve_manifest_path(value, manifest_path):
    if not value:
        return None
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    candidate = Path(manifest_path).parent / path
    return candidate if candidate.exists() else path


def load_inference_prompt_index(inference_manifest=None):
    prompts = {}
    pairing_rows = {}
    if not inference_manifest:
        return prompts, pairing_rows

    inference_manifest = Path(inference_manifest)
    if not inference_manifest.exists():
        raise FileNotFoundError(f"Inference manifest does not exist: {inference_manifest}")
    with inference_manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    for dataset_entry in payload.get("datasets", []):
        if not isinstance(dataset_entry, dict):
            continue
        dataset_name = str(dataset_entry.get("name") or "")
        pairing_path = _resolve_manifest_path(
            dataset_entry.get("suggestion_pairing_manifest")
            or dataset_entry.get("iqa_pairing_manifest"),
            inference_manifest,
        )
        if pairing_path is None or not pairing_path.exists():
            continue
        with pairing_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    pairing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_dataset = str(pairing.get("dataset") or dataset_name)
                output_filename = pairing.get("output_filename")
                if not output_filename:
                    source_path = pairing.get("source_image_path") or pairing.get("source_lq_path")
                    if source_path:
                        output_filename = f"{Path(source_path).stem}.png"
                if not output_filename:
                    continue
                key = (row_dataset, str(output_filename))
                pairing_rows[key] = pairing
                prompt = pairing.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    prompts[key] = prompt.strip()
    return prompts, pairing_rows


def build_prompt_context(
    *,
    inference_manifest=None,
    jsonl_path=None,
    use_prompt=True,
    use_suggestions=True,
    prompt_variant=None,
    case_rows=None,
    lq_indexes=None,
):
    prompts, pairing_rows = load_inference_prompt_index(inference_manifest)
    unresolved_rows = [
        row
        for row in (case_rows or [])
        if (str(row.get("dataset") or ""), str(row.get("filename") or "")) not in prompts
    ]
    target_names = {str(row.get("filename") or "") for row in unresolved_rows}
    target_paths = {
        str(lq_path)
        for row in unresolved_rows
        for lq_path in [find_lq_path(row, lq_indexes or {})]
        if lq_path is not None
    }
    return {
        "prompts": prompts,
        "pairing_rows": pairing_rows,
        "profiles": load_profile_index(
            jsonl_path,
            target_paths=target_paths,
            target_names=target_names,
        ) if unresolved_rows else {},
        "use_prompt": bool(use_prompt),
        "use_suggestions": bool(use_suggestions),
        "prompt_variant": prompt_variant,
    }


def resolve_case_prompt(row, lq_path, prompt_context):
    if not prompt_context:
        return ""
    dataset = str(row.get("dataset") or "")
    filename = str(row.get("filename") or "")
    case_key = (dataset, filename)
    exact_prompt = prompt_context["prompts"].get(case_key)
    if exact_prompt:
        return exact_prompt

    pairing = prompt_context["pairing_rows"].get(case_key, {})
    profile = None
    lookup_values = [
        lq_path,
        pairing.get("source_lq_path"),
        pairing.get("source_image_path"),
        filename,
    ]
    for value in lookup_values:
        for key in _profile_lookup_keys(value, dataset=dataset):
            profile = prompt_context["profiles"].get(key)
            if profile is not None:
                break
        if profile is not None:
            break

    prompt_variant = prompt_context["prompt_variant"]
    use_prompt = prompt_context["use_prompt"]
    use_suggestions = prompt_context["use_suggestions"]
    if profile is None:
        if prompt_variant == "fixed" or not use_prompt:
            return DEFAULT_SR_PROMPT
        return ""

    profile = copy.deepcopy(profile)
    donor_suggestion = pairing.get("donor_suggestion")
    if donor_suggestion is not None:
        profile["suggestion"] = donor_suggestion
    return build_sr_prompt(
        profile,
        use_prompt=use_prompt,
        use_suggestions=use_suggestions,
        prompt_variant=prompt_variant,
    )


def load_report_font(font_size=40):
    font_size = int(font_size)
    if font_size <= 0:
        raise ValueError(f"font_size must be positive, got {font_size}")
    return ImageFont.load_default(size=font_size)


def font_layout(font, font_size):
    bbox = font.getbbox("Ag")
    glyph_height = max(bbox[3] - bbox[1], int(font_size))
    line_height = glyph_height + max(4, int(font_size) // 5)
    padding = max(8, int(font_size) // 3)
    return line_height, padding


def placeholder_image(size, text, font=None):
    image = Image.new("RGB", size, color=(210, 210, 210))
    draw = ImageDraw.Draw(image)
    font = font or ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = max((size[0] - (bbox[2] - bbox[0])) // 2, 0)
    y = max((size[1] - (bbox[3] - bbox[1])) // 2, 0)
    draw.text((x, y), text, fill=(40, 40, 40), font=font)
    return image


def draw_text_lines(draw, xy, lines, fill=(0, 0, 0), font=None, line_height=14):
    font = font or ImageFont.load_default()
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def _split_long_word(draw, word, max_width, font):
    chunks = []
    current = ""
    for character in word:
        candidate = current + character
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if current and bbox[2] - bbox[0] > max_width:
            chunks.append(current)
            current = character
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [""]


def wrap_text_pixels(draw, text, max_width, font):
    lines = []
    for paragraph in str(text or "").splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            chunks = _split_long_word(draw, word, max_width, font)
            lines.extend(chunks[:-1])
            current = chunks[-1]
        if current:
            lines.append(current)
    return lines


def compose_comparison_image(
    row,
    lq_path,
    output_path,
    metrics,
    title_metric=None,
    prompt="",
    font_size=40,
):
    sr_path = Path(row["path"])
    with Image.open(sr_path) as sr_image:
        sr = sr_image.convert("RGB")
    font = load_report_font(font_size)
    line_height, padding = font_layout(font, font_size)
    if lq_path is not None and Path(lq_path).exists():
        with Image.open(lq_path) as lq_image:
            lq = lq_image.convert("RGB").resize(sr.size, Image.Resampling.BICUBIC)
        lq_found = True
    else:
        lq = placeholder_image(sr.size, "LQ unavailable", font=font)
        lq_found = False

    label_h = line_height + padding * 2
    width = max(sr.width * 2, int(font_size) * 20)
    layout_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    prompt = str(prompt or "").strip()
    prompt_lines = wrap_text_pixels(
        layout_draw,
        prompt,
        max_width=width - padding * 2,
        font=font,
    ) if prompt else []

    score_parts = []
    if "joint_badness" in row:
        score_parts.append(f"joint_badness={row['joint_badness']}")
    for metric in metrics:
        if metric in row:
            score_parts.append(f"{metric}={float(row[metric]):.6f}")
    caption_fields = [
        f"rank={row.get('rank', '')} dataset={row.get('dataset', '')} filename={row.get('filename', '')}",
        " ".join(score_parts),
        f"lq_found={str(lq_found).lower()} metric={title_metric or ''}",
    ]
    caption = []
    for field in caption_fields:
        caption.extend(wrap_text_pixels(layout_draw, field, width - padding * 2, font))
    caption_h = padding * 2 + line_height * len(caption)
    prompt_h = 0 if not prompt_lines else padding * 2 + line_height * (len(prompt_lines) + 1)
    height = label_h + sr.height + caption_h + prompt_h
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(lq, (0, label_h))
    canvas.paste(sr, (sr.width, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, label_h), fill=(245, 245, 245))
    draw.text((padding, padding), "LQ upscaled", fill=(0, 0, 0), font=font)
    draw.text((sr.width + padding, padding), "SR output", fill=(0, 0, 0), font=font)

    caption_top = label_h + sr.height
    draw_text_lines(
        draw,
        (padding, caption_top + padding),
        caption,
        font=font,
        line_height=line_height,
    )
    if prompt_lines:
        prompt_top = caption_top + caption_h
        draw.rectangle((0, prompt_top, width, height), fill=(245, 245, 245))
        draw_text_lines(
            draw,
            (padding, prompt_top + padding),
            ["Prompt:", *prompt_lines],
            font=font,
            line_height=line_height,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return lq_found


def csv_fieldnames(rows, metrics):
    base = ["rank", "dataset", "filename", "sr_path", "lq_path", "lq_found", "prompt"]
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
            f"<pre style=\"white-space:pre-wrap\"><strong>Prompt:</strong> {html.escape(row.get('prompt', ''))}</pre>"
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


def write_case_outputs(
    rows,
    metrics,
    output_dir,
    lq_indexes,
    title,
    prompt_context=None,
    font_size=40,
):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    output_rows = []
    for rank, row in enumerate(rows, start=1):
        row = dict(row)
        row["rank"] = rank
        lq_path = find_lq_path(row, lq_indexes)
        prompt = resolve_case_prompt(row, lq_path, prompt_context)
        comparison_path = images_dir / image_output_name(rank, row["filename"])
        lq_found = compose_comparison_image(
            row,
            lq_path,
            comparison_path,
            metrics,
            title_metric=title,
            prompt=prompt,
            font_size=font_size,
        )
        row["sr_path"] = row["path"]
        row["lq_path"] = str(lq_path) if lq_path else ""
        row["lq_found"] = "true" if lq_found else "false"
        row["prompt"] = prompt
        row["comparison_path"] = str(comparison_path)
        output_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_worst_cases_csv(output_dir / "worst_cases.csv", output_rows, metrics)
    write_html_report(output_dir / "report.html", output_rows, metrics, title)
    return output_rows


def joint_output_name(metrics):
    return "joint_mean_" + "_".join(metrics)


def run_analysis(
    metrics_csv,
    summary_json,
    metrics,
    mode,
    worst_k,
    lq_dirs,
    output_dir,
    inference_manifest=None,
    jsonl_path=None,
    use_prompt=True,
    use_suggestions=True,
    prompt_variant=None,
    font_size=40,
):
    if not metrics:
        raise ValueError("At least one metric is required")
    if worst_k <= 0:
        raise ValueError("worst_k must be positive")
    if mode not in {"separate", "joint_mean"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if int(font_size) <= 0:
        raise ValueError("font_size must be positive")

    rows = load_metric_rows(metrics_csv)
    directions = load_metric_directions(summary_json)
    lq_indexes = build_lq_indexes(lq_dirs)
    output_dir = Path(output_dir)

    if mode == "joint_mean":
        ranked = rank_joint_mean(rows, metrics, directions, worst_k)
        ranked_groups = [
            (ranked, metrics, output_dir / joint_output_name(metrics), joint_output_name(metrics))
        ]
    else:
        multi_metric = len(metrics) > 1
        ranked_groups = []
        for metric in metrics:
            ranked = rank_single_metric(rows, metric, directions, worst_k)
            target_dir = output_dir / metric if multi_metric else output_dir
            ranked_groups.append((ranked, [metric], target_dir, metric))

    case_rows = [row for ranked, _, _, _ in ranked_groups for row in ranked]
    prompt_context = build_prompt_context(
        inference_manifest=inference_manifest,
        jsonl_path=jsonl_path,
        use_prompt=use_prompt,
        use_suggestions=use_suggestions,
        prompt_variant=prompt_variant,
        case_rows=case_rows,
        lq_indexes=lq_indexes,
    )
    written = []
    for ranked, ranked_metrics, target_dir, title in ranked_groups:
        written.append(
            write_case_outputs(
                ranked,
                ranked_metrics,
                target_dir,
                lq_indexes,
                title,
                prompt_context=prompt_context,
                font_size=font_size,
            )
        )
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
    parser.add_argument(
        "--inference_manifest",
        type=Path,
        default=None,
        help="Inference manifest used to recover the exact per-image prompt when available.",
    )
    parser.add_argument(
        "--jsonl_path",
        type=Path,
        default=None,
        help="Condition JSONL used to reconstruct prompts from older inference outputs.",
    )
    parser.add_argument("--use_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_suggestions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prompt_variant",
        choices=("fixed", "suggestion", "iqa", "iqa_suggestion"),
        default=None,
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=40,
        help="Pixel size for labels, metrics, and prompt text in comparison images.",
    )
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
        inference_manifest=args.inference_manifest,
        jsonl_path=args.jsonl_path,
        use_prompt=args.use_prompt,
        use_suggestions=args.use_suggestions,
        prompt_variant=args.prompt_variant,
        font_size=args.font_size,
    )
    print(f"Saved bad case analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
