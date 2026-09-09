"""Profile Condition8, Condition8 text, IQA, and suggestion distinctiveness.

The script deliberately depends only on the Python standard library and the
repository's production prompt/condition builders.  It therefore measures the
exact representation consumed by training and inference instead of maintaining
a second, analysis-only extractor.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.prompt_builder import (  # noqa: E402
    IQA_FIELDS,
    build_condition8_text,
    normalized_prompt_profile,
)
from models.router_condition import (  # noqa: E402
    ROUTER_CONDITION_KEYS,
    extract_router_condition,
)


LEVEL_LABELS = ("no visible", "subtle", "mild", "moderate", "severe", "extreme")
PHYSICAL_KEY_MAP = {
    "blur": ("blur",),
    "noise": ("noise",),
    "compression": ("jpeg", "compression"),
    "ringing_aliasing": ("ringing", "ringing_aliasing"),
    "texture_loss": ("texture_loss",),
    "photometric": ("color_shift", "photometric"),
    "structure_risk": ("text_region_risk", "structure_risk"),
    "hallucination_risk": ("hallucination_risk",),
}


def percentile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return math.nan
    position = (len(values) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def safe_mean(values):
    return statistics.fmean(values) if values else math.nan


def safe_std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def pearson(x_values, y_values):
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return math.nan
    x_values, y_values = zip(*pairs)
    x_mean = safe_mean(x_values)
    y_mean = safe_mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_norm = sum((x - x_mean) ** 2 for x in x_values)
    y_norm = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_norm * y_norm)
    return numerator / denominator if denominator else math.nan


def tied_ranks(values):
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        start = end
    return ranks


def spearman(x_values, y_values):
    if len(x_values) < 2 or len(y_values) < 2:
        return math.nan
    return pearson(tied_ranks(x_values), tied_ranks(y_values))


def normalized_entropy(counter):
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counter.values()
        if count
    )
    return entropy / math.log(len(counter))


def condition_level(value):
    anchors = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    return LEVEL_LABELS[min(range(len(anchors)), key=lambda index: abs(anchors[index] - value))]


def normalize_text(value):
    return " ".join(str(value or "").casefold().split())


def text_tokens(value):
    return set(normalize_text(value).replace("/", " ").replace("-", " ").split())


def sampled_pairwise_jaccard(texts, pair_count, seed):
    if len(texts) < 2 or pair_count <= 0:
        return {"pairs": 0, "mean": math.nan, "p50": math.nan, "p90": math.nan}
    token_sets = [text_tokens(text) for text in texts]
    rng = random.Random(seed)
    similarities = []
    for _ in range(min(pair_count, len(texts) * 10)):
        left = rng.randrange(len(texts))
        right = rng.randrange(len(texts) - 1)
        if right >= left:
            right += 1
        union = token_sets[left] | token_sets[right]
        similarities.append(
            len(token_sets[left] & token_sets[right]) / len(union) if union else 1.0
        )
    return {
        "pairs": len(similarities),
        "mean": safe_mean(similarities),
        "p50": percentile(similarities, 0.5),
        "p90": percentile(similarities, 0.9),
    }


def source_profile(profile, source):
    profile = profile if isinstance(profile, dict) else {}
    if source == "condition8":
        return profile
    if source == "iqa":
        return {"iqa": profile.get("iqa") or {}, "suggestion": ""}
    if source == "suggestion":
        return {"iqa": {}, "suggestion": profile.get("suggestion") or ""}
    raise ValueError(f"Unknown prompt source: {source}")


def physical_vector(record):
    candidates = []
    result = record.get("result")
    if isinstance(result, dict):
        candidates.append(result.get("degradation_vector"))
    raw_params = record.get("raw_degradation_params")
    if isinstance(raw_params, dict):
        candidates.append(raw_params.get("degradation_vector"))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        values = []
        available = []
        for condition_key in ROUTER_CONDITION_KEYS:
            matched = [
                float(candidate[key])
                for key in PHYSICAL_KEY_MAP[condition_key]
                if key in candidate and candidate[key] is not None
            ]
            values.append(max(matched) if matched else math.nan)
            available.append(bool(matched))
        return values, available
    return [math.nan] * len(ROUTER_CONDITION_KEYS), [False] * len(ROUTER_CONDITION_KEYS)


def iter_records(jsonl_path, limit=None):
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        yielded = 0
        for line_no, line in enumerate(handle, start=1):
            if limit is not None and yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, None, f"invalid JSON: {exc}"
                continue
            yield line_no, record, None
            yielded += 1


def summarize_texts(texts, pair_count, seed):
    normalized = [normalize_text(text) for text in texts]
    nonempty = [text for text in normalized if text]
    counts = Counter(nonempty)
    lengths = [len(text.split()) for text in nonempty]
    top_text, top_count = counts.most_common(1)[0] if counts else ("", 0)
    return {
        "records": len(texts),
        "nonempty_rate": len(nonempty) / len(texts) if texts else 0.0,
        "unique_count": len(counts),
        "unique_rate": len(counts) / len(nonempty) if nonempty else 0.0,
        "normalized_entropy": normalized_entropy(counts),
        "top1_share": top_count / len(nonempty) if nonempty else 0.0,
        "top1_sha256": hashlib.sha256(top_text.encode("utf-8")).hexdigest()[:12] if top_text else "",
        "word_count": {
            "mean": safe_mean(lengths),
            "p10": percentile(lengths, 0.1),
            "p50": percentile(lengths, 0.5),
            "p90": percentile(lengths, 0.9),
        },
        "pairwise_jaccard": sampled_pairwise_jaccard(nonempty, pair_count, seed),
    }


def distinctiveness_label(summary):
    coverage = safe_mean([row["valid_rate"] for row in summary])
    spread = safe_mean([row["std"] for row in summary])
    entropy = safe_mean([row["level_entropy"] for row in summary])
    collapsed = sum(row["top_level_share"] >= 0.8 for row in summary)
    if coverage >= 0.5 and spread >= 0.12 and entropy >= 0.55 and collapsed <= 2:
        return "strong"
    if coverage >= 0.3 and spread >= 0.07 and entropy >= 0.35 and collapsed <= 5:
        return "moderate"
    return "weak"


def analyze(jsonl_path, limit=None, pair_count=10000, seed=42):
    sources = ("condition8", "iqa", "suggestion")
    values = {source: [[] for _ in ROUTER_CONDITION_KEYS] for source in sources}
    masks = {source: [[] for _ in ROUTER_CONDITION_KEYS] for source in sources}
    physical = [[] for _ in ROUTER_CONDITION_KEYS]
    physical_available = [[] for _ in ROUTER_CONDITION_KEYS]
    texts = {"iqa": [], "suggestion": [], "condition8_text": []}
    condition_signatures = Counter()
    errors = []
    field_present = Counter()
    record_count = 0

    for line_no, record, parse_error in iter_records(jsonl_path, limit=limit):
        if parse_error:
            errors.append({"line": line_no, "error": parse_error})
            continue
        unipercept_raw = record.get("unipercept_raw")
        profile = unipercept_raw.get("profile") if isinstance(unipercept_raw, dict) else None
        if not isinstance(profile, dict):
            errors.append({"line": line_no, "error": "missing unipercept_raw.profile"})
            continue
        try:
            normalized = normalized_prompt_profile(profile)
            source_conditions = {
                source: extract_router_condition(source_profile(profile, source))
                for source in sources
            }
            canonical_text = build_condition8_text(profile)
        except (TypeError, ValueError) as exc:
            errors.append({"line": line_no, "error": str(exc)})
            continue

        record_count += 1
        for field in IQA_FIELDS:
            if normalized["iqa"][field]:
                field_present[f"iqa.{field}"] += 1
        if normalized["suggestion"]:
            field_present["suggestion"] += 1
        iqa_text = " ".join(normalized["iqa"][field] for field in IQA_FIELDS)
        texts["iqa"].append(iqa_text)
        texts["suggestion"].append(normalized["suggestion"])
        texts["condition8_text"].append(canonical_text)

        for source, condition in source_conditions.items():
            for index, (value, valid) in enumerate(zip(condition.values, condition.valid_mask)):
                values[source][index].append(float(value))
                masks[source][index].append(bool(valid))
        combined = source_conditions["condition8"]
        condition_signatures[
            tuple(
                condition_level(value) if valid else "unknown"
                for value, valid in zip(combined.values, combined.valid_mask)
            )
        ] += 1
        physical_values, availability = physical_vector(record)
        for index, (value, available) in enumerate(zip(physical_values, availability)):
            physical[index].append(value)
            physical_available[index].append(available)

    if not record_count:
        raise ValueError(f"No usable prompt profiles found in {jsonl_path}")

    dimension_rows = []
    for source in sources:
        for index, key in enumerate(ROUTER_CONDITION_KEYS):
            dimension_values = values[source][index]
            valid_values = [
                value for value, valid in zip(dimension_values, masks[source][index]) if valid
            ]
            level_counts = Counter(condition_level(value) for value in valid_values)
            top_level, top_count = level_counts.most_common(1)[0] if level_counts else ("", 0)
            aligned_pairs = [
                (value, physical_value)
                for value, valid, physical_value, available in zip(
                    dimension_values,
                    masks[source][index],
                    physical[index],
                    physical_available[index],
                )
                if valid and available and math.isfinite(physical_value)
            ]
            extracted_aligned = [pair[0] for pair in aligned_pairs]
            physical_aligned = [pair[1] for pair in aligned_pairs]
            dimension_rows.append(
                {
                    "source": source,
                    "dimension": key,
                    "records": record_count,
                    "valid_rate": len(valid_values) / record_count,
                    "zero_rate_when_valid": (
                        sum(value == 0.0 for value in valid_values) / len(valid_values)
                        if valid_values
                        else math.nan
                    ),
                    "mean": safe_mean(valid_values),
                    "std": safe_std(valid_values),
                    "p10": percentile(valid_values, 0.1),
                    "p50": percentile(valid_values, 0.5),
                    "p90": percentile(valid_values, 0.9),
                    "unique_numeric": len(set(valid_values)),
                    "level_entropy": normalized_entropy(level_counts),
                    "top_level": top_level,
                    "top_level_share": top_count / len(valid_values) if valid_values else math.nan,
                    "physical_pairs": len(aligned_pairs),
                    "physical_pearson": pearson(extracted_aligned, physical_aligned),
                    "physical_spearman": spearman(extracted_aligned, physical_aligned),
                    **{f"level_{label.replace(' ', '_')}_rate": level_counts[label] / len(valid_values) if valid_values else 0.0 for label in LEVEL_LABELS},
                }
            )

    text_summaries = {
        name: summarize_texts(source_texts, pair_count, seed + index)
        for index, (name, source_texts) in enumerate(texts.items())
    }
    source_contribution = []
    for index, key in enumerate(ROUTER_CONDITION_KEYS):
        iqa_mask = masks["iqa"][index]
        suggestion_mask = masks["suggestion"][index]
        combined_values = values["condition8"][index]
        iqa_values = values["iqa"][index]
        both = sum(left and right for left, right in zip(iqa_mask, suggestion_mask))
        iqa_only = sum(left and not right for left, right in zip(iqa_mask, suggestion_mask))
        suggestion_only = sum(not left and right for left, right in zip(iqa_mask, suggestion_mask))
        neither = record_count - both - iqa_only - suggestion_only
        changed = [
            abs(combined - iqa)
            for combined, iqa, has_iqa, has_suggestion in zip(
                combined_values,
                iqa_values,
                iqa_mask,
                suggestion_mask,
            )
            if has_iqa and has_suggestion
        ]
        source_contribution.append(
            {
                "dimension": key,
                "both_rate": both / record_count,
                "iqa_only_rate": iqa_only / record_count,
                "suggestion_only_rate": suggestion_only / record_count,
                "neither_rate": neither / record_count,
                "suggestion_changes_iqa_rate_when_both": (
                    sum(delta > 1e-12 for delta in changed) / len(changed) if changed else 0.0
                ),
                "mean_abs_change_when_both": safe_mean(changed),
            }
        )
    combined_rows = [row for row in dimension_rows if row["source"] == "condition8"]
    top_signature_count = condition_signatures.most_common(1)[0][1]
    summary = {
        "input": str(Path(jsonl_path)),
        "records": record_count,
        "errors": len(errors),
        "field_completeness": {
            field: field_present[field] / record_count
            for field in (*[f"iqa.{field}" for field in IQA_FIELDS], "suggestion")
        },
        "condition8": {
            "distinctiveness": distinctiveness_label(combined_rows),
            "unique_level_combinations": len(condition_signatures),
            "unique_combination_rate": len(condition_signatures) / record_count,
            "top_combination_share": top_signature_count / record_count,
            "combination_entropy": normalized_entropy(condition_signatures),
            "mean_valid_dimensions": safe_mean(
                [sum(row) for row in zip(*masks["condition8"])]
            ),
        },
        "text": text_summaries,
        "source_contribution": source_contribution,
        "dimensions": dimension_rows,
        "error_examples": errors[:20],
    }
    return summary


def format_number(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_markdown(summary):
    condition = summary["condition8"]
    lines = [
        "# Prompt degradation distribution report",
        "",
        f"- Input: `{summary['input']}`",
        f"- Usable records: {summary['records']:,}; errors: {summary['errors']:,}",
        f"- Condition8 distinctiveness (distribution-only heuristic): **{condition['distinctiveness']}**",
        f"- Mean valid dimensions: {condition['mean_valid_dimensions']:.2f}/8",
        f"- Unique canonical combinations: {condition['unique_level_combinations']:,} "
        f"({condition['unique_combination_rate']:.1%}); top combination share: "
        f"{condition['top_combination_share']:.1%}",
        "",
        "## Field completeness",
        "",
    ]
    lines.extend(
        f"- `{field}`: {rate:.1%}"
        for field, rate in summary["field_completeness"].items()
    )
    lines.extend(
        [
            "",
            "## Condition dimensions",
            "",
            "| Source | Dimension | Valid | Mean | Std | P10 | P50 | P90 | Level entropy | Top level | Top share | Spearman vs physical |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in summary["dimensions"]:
        lines.append(
            "| {source} | {dimension} | {valid_rate:.1%} | {mean} | {std} | {p10} | "
            "{p50} | {p90} | {entropy} | {top} | {top_share} | {spearman} |".format(
                source=row["source"],
                dimension=row["dimension"],
                valid_rate=row["valid_rate"],
                mean=format_number(row["mean"]),
                std=format_number(row["std"]),
                p10=format_number(row["p10"]),
                p50=format_number(row["p50"]),
                p90=format_number(row["p90"]),
                entropy=format_number(row["level_entropy"]),
                top=row["top_level"] or "NA",
                top_share=format_number(row["top_level_share"]),
                spearman=format_number(row["physical_spearman"]),
            )
        )
    lines.extend(
        [
            "",
            "## IQA and suggestion contribution",
            "",
            "| Dimension | Both | IQA only | Suggestion only | Neither | Suggestion changes combined when both | Mean absolute change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["source_contribution"]:
        lines.append(
            "| {dimension} | {both_rate:.1%} | {iqa_only_rate:.1%} | "
            "{suggestion_only_rate:.1%} | {neither_rate:.1%} | {change_rate:.1%} | {mean_change} |".format(
                dimension=row["dimension"],
                both_rate=row["both_rate"],
                iqa_only_rate=row["iqa_only_rate"],
                suggestion_only_rate=row["suggestion_only_rate"],
                neither_rate=row["neither_rate"],
                change_rate=row["suggestion_changes_iqa_rate_when_both"],
                mean_change=format_number(row["mean_abs_change_when_both"]),
            )
        )
    lines.extend(["", "## Text diversity", ""])
    for name, text in summary["text"].items():
        lines.append(
            f"- **{name}**: unique {text['unique_rate']:.1%}; top-1 {text['top1_share']:.1%}; "
            f"entropy {text['normalized_entropy']:.3f}; words p50={text['word_count']['p50']:.0f}; "
            f"random-pair Jaccard mean={text['pairwise_jaccard']['mean']:.3f}, "
            f"p90={text['pairwise_jaccard']['p90']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "The heuristic rating only detects distribution collapse. Treat the prompt as genuinely "
            "discriminative only when dimensions also align with physical degradation and matched-vs-shuffled "
            "prompt inference produces a significant quality advantage.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = summary["dimensions"]
    csv_path = output_dir / "dimension_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_dir / "report.md"
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    return summary_path, csv_path, report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Condition8 and source prompt degradation distributions."
    )
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pair_count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = analyze(
        args.jsonl_path,
        limit=args.limit,
        pair_count=args.pair_count,
        seed=args.seed,
    )
    paths = write_outputs(summary, args.output_dir)
    print(
        json.dumps(
            {
                "records": summary["records"],
                "errors": summary["errors"],
                "condition8": summary["condition8"],
                "outputs": [str(path) for path in paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
