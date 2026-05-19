import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import yaml
from PIL import Image


class RGSrMetricsTests(unittest.TestCase):
    def test_default_metrics_match_omgsr(self):
        from metrics.rg_sr_metrics import DEFAULT_OMGSR_METRICS

        self.assertEqual(
            DEFAULT_OMGSR_METRICS,
            ["clipiqa", "clipiqa+", "nima", "niqe", "liqe", "musiq", "maniqa"],
        )

    def test_metric_direction_uses_lower_better_fallback_for_niqe(self):
        from metrics.rg_sr_metrics import metric_direction

        class MetricWithoutDirection:
            pass

        self.assertEqual(metric_direction("niqe", MetricWithoutDirection()), "lower_better")
        self.assertEqual(metric_direction("musiq", MetricWithoutDirection()), "higher_better")

    def test_evaluate_metrics_writes_each_metric_for_each_image(self):
        from metrics.rg_sr_metrics import build_rows, evaluate_metrics

        calls = []

        class FakeScore:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numel(self):
                return 1

            def item(self):
                return self.value

        class FakeMetric:
            lower_better = False

            def __init__(self, name):
                self.name = name

            def eval(self):
                return self

            def __call__(self, path):
                calls.append((self.name, Path(path).name))
                return FakeScore(float(len(calls)))

        fake_pyiqa = types.SimpleNamespace(create_metric=lambda name, device: FakeMetric(name))
        old_pyiqa = sys.modules.get("pyiqa")
        sys.modules["pyiqa"] = fake_pyiqa
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image_a = root / "a.png"
                image_b = root / "b.png"
                Image.new("RGB", (8, 8), color="red").save(image_a)
                Image.new("RGB", (8, 8), color="blue").save(image_b)
                rows = build_rows({"smoke": [image_a, image_b]})

                directions = evaluate_metrics(rows, ["clipiqa", "niqe"], "cpu")
        finally:
            if old_pyiqa is None:
                sys.modules.pop("pyiqa", None)
            else:
                sys.modules["pyiqa"] = old_pyiqa

        self.assertEqual(
            calls,
            [
                ("clipiqa", "a.png"),
                ("clipiqa", "b.png"),
                ("niqe", "a.png"),
                ("niqe", "b.png"),
            ],
        )
        self.assertEqual(directions["clipiqa"], "higher_better")
        self.assertEqual(rows[0]["clipiqa"], 1.0)
        self.assertEqual(rows[1]["niqe"], 4.0)

    def test_write_outputs_creates_per_image_csv_and_summary_json(self):
        from metrics.rg_sr_metrics import build_summary, write_outputs

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "metrics"
            rows = [
                {
                    "dataset": "smoke",
                    "filename": "a.png",
                    "path": "/tmp/a.png",
                    "width": 8,
                    "height": 8,
                    "clipiqa": 0.25,
                    "niqe": 4.0,
                },
                {
                    "dataset": "smoke",
                    "filename": "b.png",
                    "path": "/tmp/b.png",
                    "width": 8,
                    "height": 8,
                    "clipiqa": 0.75,
                    "niqe": 2.0,
                },
            ]
            metrics = ["clipiqa", "niqe"]
            summary_rows = build_summary(rows, ["smoke"], metrics)
            write_outputs(output_dir, rows, summary_rows, metrics, {"clipiqa": "higher_better", "niqe": "lower_better"})

            with (output_dir / "per_image_scores.csv").open(newline="", encoding="utf-8") as handle:
                per_image_rows = list(csv.DictReader(handle))
            summary_json = json.loads((output_dir / "summary_scores.json").read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "summary_scores.csv").exists())
            self.assertEqual(per_image_rows[0]["filename"], "a.png")
            self.assertEqual(summary_json["metrics"], metrics)
            clip_summary = next(row for row in summary_json["summary"] if row["metric"] == "clipiqa")
            self.assertEqual(clip_summary["mean"], 0.5)
            self.assertEqual(clip_summary["count"], 2)

    def test_training_configs_enable_500_step_evaluation_by_default(self):
        for config_path in ("configs/train_rg_flux_sr_ms.yaml", "configs/train_rg_flux_sr_ms_smoke_256.yaml"):
            config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            evaluation = config["evaluation"]

            self.assertTrue(evaluation["enabled"])
            self.assertEqual(evaluation["eval_every"], 500)
            self.assertEqual(evaluation["num_samples"], 8)
            self.assertEqual(evaluation["output_dir"], "eval")
            self.assertEqual(evaluation["metrics"], ["clipiqa", "clipiqa+", "nima", "niqe", "liqe", "musiq", "maniqa"])

    def test_train_script_wires_periodic_rg_flux_evaluation(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn("run_rg_flux_evaluation", source)
        self.assertIn("evaluation.eval_every", source)
        self.assertIn("eval/<metric>", source)


if __name__ == "__main__":
    unittest.main()
