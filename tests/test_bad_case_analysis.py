import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def write_image(path, color, size=(16, 16)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def write_scores(path, rows, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "filename", "path", "width", "height", *metrics])
        writer.writeheader()
        writer.writerows(rows)


def read_worst_cases(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class BadCaseAnalysisTests(unittest.TestCase):
    def test_single_higher_better_metric_selects_lowest_scores(self):
        from tools.analyze_rg_flux_bad_cases import run_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr_dir = root / "sr"
            lq_dir = root / "lq"
            for name, color in [("a.png", "red"), ("b.png", "green"), ("c.png", "blue")]:
                write_image(sr_dir / name, color)
                write_image(lq_dir / name, color)
            scores = root / "metrics" / "per_image_scores.csv"
            write_scores(
                scores,
                [
                    {"dataset": "RealLQ250", "filename": "a.png", "path": sr_dir / "a.png", "width": 16, "height": 16, "maniqa": 0.9},
                    {"dataset": "RealLQ250", "filename": "b.png", "path": sr_dir / "b.png", "width": 16, "height": 16, "maniqa": 0.1},
                    {"dataset": "RealLQ250", "filename": "c.png", "path": sr_dir / "c.png", "width": 16, "height": 16, "maniqa": 0.3},
                ],
                ["maniqa"],
            )

            output = root / "bad_maniqa"
            run_analysis(
                metrics_csv=scores,
                summary_json=None,
                metrics=["maniqa"],
                mode="separate",
                worst_k=2,
                lq_dirs={"RealLQ250": lq_dir},
                output_dir=output,
            )

            rows = read_worst_cases(output / "worst_cases.csv")
            self.assertEqual([row["filename"] for row in rows], ["b.png", "c.png"])
            self.assertEqual(rows[0]["lq_found"], "true")
            self.assertTrue((output / "report.html").exists())
            self.assertEqual(len(list((output / "images").glob("*.png"))), 2)

    def test_single_lower_better_metric_selects_highest_scores(self):
        from tools.analyze_rg_flux_bad_cases import run_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr_dir = root / "sr"
            lq_dir = root / "lq"
            for name in ["a.png", "b.png", "c.png"]:
                write_image(sr_dir / name, "white")
                write_image(lq_dir / name, "black")
            scores = root / "metrics" / "per_image_scores.csv"
            write_scores(
                scores,
                [
                    {"dataset": "RealLR200", "filename": "a.png", "path": sr_dir / "a.png", "width": 16, "height": 16, "niqe": 3.0},
                    {"dataset": "RealLR200", "filename": "b.png", "path": sr_dir / "b.png", "width": 16, "height": 16, "niqe": 7.0},
                    {"dataset": "RealLR200", "filename": "c.png", "path": sr_dir / "c.png", "width": 16, "height": 16, "niqe": 5.0},
                ],
                ["niqe"],
            )

            output = root / "bad_niqe"
            run_analysis(
                metrics_csv=scores,
                summary_json=None,
                metrics=["niqe"],
                mode="separate",
                worst_k=2,
                lq_dirs={"RealLR200": lq_dir},
                output_dir=output,
            )

            rows = read_worst_cases(output / "worst_cases.csv")
            self.assertEqual([row["filename"] for row in rows], ["b.png", "c.png"])

    def test_separate_mode_writes_one_directory_per_metric_when_multiple_metrics(self):
        from tools.analyze_rg_flux_bad_cases import run_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr_dir = root / "sr"
            lq_dir = root / "lq"
            for name in ["a.png", "b.png"]:
                write_image(sr_dir / name, "white")
                write_image(lq_dir / name, "black")
            scores = root / "metrics" / "per_image_scores.csv"
            write_scores(
                scores,
                [
                    {"dataset": "RealLQ250", "filename": "a.png", "path": sr_dir / "a.png", "width": 16, "height": 16, "maniqa": 0.1, "niqe": 2.0},
                    {"dataset": "RealLQ250", "filename": "b.png", "path": sr_dir / "b.png", "width": 16, "height": 16, "maniqa": 0.9, "niqe": 8.0},
                ],
                ["maniqa", "niqe"],
            )

            output = root / "bad"
            run_analysis(
                metrics_csv=scores,
                summary_json=None,
                metrics=["maniqa", "niqe"],
                mode="separate",
                worst_k=1,
                lq_dirs={"RealLQ250": lq_dir},
                output_dir=output,
            )

            self.assertEqual(read_worst_cases(output / "maniqa" / "worst_cases.csv")[0]["filename"], "a.png")
            self.assertEqual(read_worst_cases(output / "niqe" / "worst_cases.csv")[0]["filename"], "b.png")

    def test_joint_mean_normalizes_metric_badness_and_averages(self):
        from tools.analyze_rg_flux_bad_cases import run_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr_dir = root / "sr"
            lq_dir = root / "lq"
            for name in ["a.png", "b.png", "c.png"]:
                write_image(sr_dir / name, "white")
                write_image(lq_dir / name, "black")
            scores = root / "metrics" / "per_image_scores.csv"
            summary = root / "metrics" / "summary_scores.json"
            write_scores(
                scores,
                [
                    {"dataset": "RealLQ250", "filename": "a.png", "path": sr_dir / "a.png", "width": 16, "height": 16, "maniqa": 1.0, "niqe": 1.0},
                    {"dataset": "RealLQ250", "filename": "b.png", "path": sr_dir / "b.png", "width": 16, "height": 16, "maniqa": 0.0, "niqe": 10.0},
                    {"dataset": "RealLQ250", "filename": "c.png", "path": sr_dir / "c.png", "width": 16, "height": 16, "maniqa": 0.5, "niqe": 5.5},
                ],
                ["maniqa", "niqe"],
            )
            summary.write_text(
                json.dumps({"metric_directions": {"maniqa": "higher_better", "niqe": "lower_better"}}),
                encoding="utf-8",
            )

            output = root / "joint"
            run_analysis(
                metrics_csv=scores,
                summary_json=summary,
                metrics=["maniqa", "niqe"],
                mode="joint_mean",
                worst_k=2,
                lq_dirs={"RealLQ250": lq_dir},
                output_dir=output,
            )

            rows = read_worst_cases(output / "joint_mean_maniqa_niqe" / "worst_cases.csv")
            self.assertEqual([row["filename"] for row in rows], ["b.png", "c.png"])
            self.assertAlmostEqual(float(rows[0]["joint_badness"]), 1.0)
            self.assertAlmostEqual(float(rows[1]["joint_badness"]), 0.5)

    def test_missing_lq_does_not_stop_visualization(self):
        from tools.analyze_rg_flux_bad_cases import run_analysis

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr_dir = root / "sr"
            lq_dir = root / "lq"
            write_image(sr_dir / "missing.png", "white")
            scores = root / "metrics" / "per_image_scores.csv"
            write_scores(
                scores,
                [
                    {"dataset": "RealLQ250", "filename": "missing.png", "path": sr_dir / "missing.png", "width": 16, "height": 16, "maniqa": 0.1},
                ],
                ["maniqa"],
            )

            output = root / "bad"
            run_analysis(
                metrics_csv=scores,
                summary_json=None,
                metrics=["maniqa"],
                mode="separate",
                worst_k=1,
                lq_dirs={"RealLQ250": lq_dir},
                output_dir=output,
            )

            rows = read_worst_cases(output / "worst_cases.csv")
            self.assertEqual(rows[0]["lq_found"], "false")
            self.assertTrue((output / "images" / "worst_0001_missing.png").exists())


if __name__ == "__main__":
    unittest.main()
