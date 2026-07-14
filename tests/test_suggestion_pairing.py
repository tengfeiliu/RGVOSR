import ast
import copy
import csv
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import yaml


class SuggestionPairingTests(unittest.TestCase):
    @staticmethod
    def load_inference_helpers():
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {
            "normalize_suggestion_pairing",
            "effective_suggestion_shuffle_seed",
            "build_suggestion_donor_indices",
            "profile_with_donor_suggestion",
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        namespace = {
            "copy": copy,
            "hashlib": hashlib,
            "random": random,
            "SUGGESTION_PAIRINGS": ("matched", "shuffled"),
        }
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "inference_rg_flux_sr.py", "exec"), namespace)
        return namespace

    def test_shuffled_pairing_is_reproducible_one_to_one_and_has_no_self_pairs(self):
        helpers = self.load_inference_helpers()
        build_indices = helpers["build_suggestion_donor_indices"]
        first = build_indices(200, pairing="shuffled", seed=1234)
        second = build_indices(200, pairing="shuffled", seed=1234)

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(200)))
        self.assertTrue(all(source != donor for source, donor in enumerate(first)))
        self.assertNotEqual(first, build_indices(200, pairing="shuffled", seed=1235))
        self.assertEqual(build_indices(4, pairing="matched", seed=1234), [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_indices(1, pairing="shuffled", seed=1234)

    def test_dataset_seed_is_stable_and_profile_swap_only_changes_suggestion(self):
        helpers = self.load_inference_helpers()
        effective_seed = helpers["effective_suggestion_shuffle_seed"]
        swap = helpers["profile_with_donor_suggestion"]

        self.assertEqual(effective_seed(3407, "RealLR200"), effective_seed(3407, "RealLR200"))
        self.assertNotEqual(effective_seed(3407, "RealLR200"), effective_seed(3407, "RealLQ250"))

        source = {
            "iqa": {"distortion": "source diagnosis"},
            "suggestion": "source suggestion",
            "iaa": {"comprehensive": "source aesthetics"},
        }
        donor = {
            "iqa": {"distortion": "donor diagnosis"},
            "suggestion": "donor suggestion",
        }
        paired = swap(source, donor)
        self.assertEqual(paired["suggestion"], "donor suggestion")
        self.assertEqual(paired["iqa"], source["iqa"])
        self.assertEqual(paired["iaa"], source["iaa"])
        self.assertEqual(source["suggestion"], "source suggestion")

    def test_pipeline_builds_same_checkpoint_online_matched_and_shuffled_commands(self):
        from tools.run_rg_flux_pipeline import (
            build_inference_command,
            suggestion_pairing_artifact_paths,
        )

        class Args:
            dataset_dirs = ["RealLR200=/data/RealLR200"]
            inference_output_root = None
            text_encoding_mode = "cached"
            text_embedding_cache = "cache"
            jsonl_path = "datasets/inference_cleaned.jsonl"
            num_inference_steps = 25
            upscale = 4
            dtype = "bf16"
            device = None
            lr_cond_mode = "flux2_image_concat"
            min_size = None
            prompt_variant = "suggestion"
            seed = 42
            use_prompt = True
            use_suggestions = True
            use_degradation_vector = False
            restore_input_size = False

        run_dir = Path("exp_rg_flux_sr/run")
        checkpoint_dir = run_dir / "checkpoints" / "checkpoint-00020000"
        commands = {}
        for pairing in ("matched", "shuffled"):
            paths = suggestion_pairing_artifact_paths(run_dir, checkpoint_dir.name, pairing, 3407)
            command, manifest = build_inference_command(
                Args,
                run_dir,
                checkpoint_dir,
                config_path=run_dir / "pipeline_runtime_config.yaml",
                output_dir=paths["inference_dir"],
                suggestion_pairing=pairing,
                suggestion_shuffle_seed=3407,
                text_encoding_mode="online",
            )
            commands[pairing] = command
            self.assertEqual(manifest, paths["inference_manifest"])
            self.assertEqual(command[command.index("--checkpoint_step") + 1], "checkpoint-00020000")
            self.assertEqual(command[command.index("--text_encoding_mode") + 1], "online")
            self.assertEqual(command[command.index("--suggestion_pairing") + 1], pairing)
            self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertEqual(
            commands["matched"][commands["matched"].index("--run_dir") + 1],
            commands["shuffled"][commands["shuffled"].index("--run_dir") + 1],
        )

    def test_dry_run_pipeline_records_both_pairings_and_metric_comparison(self):
        from tools.run_rg_flux_pipeline import main

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "pipeline_runtime_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "condition": {
                            "prompt_variant": "suggestion",
                            "use_prompt": True,
                            "use_suggestions": True,
                            "use_degradation_vector": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            returncode = main(
                [
                    "--skip_train",
                    "--run_dir",
                    str(run_dir),
                    "--checkpoint_steps",
                    "20000",
                    "--dataset_dirs",
                    "RealLR200=/data/RealLR200",
                    "--jsonl_path",
                    "datasets/inference_cleaned.jsonl",
                    "--compare_suggestion_pairing",
                    "--dry_run_pipeline",
                ]
            )
            self.assertEqual(returncode, 0)
            manifest = json.loads((run_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
            record = manifest["records"][0]
            self.assertEqual(set(record["suggestion_pairing_runs"]), {"matched", "shuffled"})
            self.assertTrue(
                any("compare_rg_flux_pairing_metrics.py" in part for part in record["pairing_comparison_command"])
            )

    def test_paired_metric_comparison_is_direction_aware(self):
        from tools.compare_rg_flux_pairing_metrics import compare_pairing_metrics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matched_dir = root / "matched"
            shuffled_dir = root / "shuffled"
            output_dir = root / "comparison"
            for directory in (matched_dir, shuffled_dir):
                directory.mkdir()
                (directory / "summary_scores.json").write_text(
                    json.dumps(
                        {
                            "metric_directions": {
                                "clipiqa": "higher_better",
                                "niqe": "lower_better",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            fieldnames = ["dataset", "filename", "path", "width", "height", "clipiqa", "niqe"]
            matched_rows = [
                {"dataset": "D", "filename": "1.png", "path": "m1", "width": 1, "height": 1, "clipiqa": 0.8, "niqe": 3.0},
                {"dataset": "D", "filename": "2.png", "path": "m2", "width": 1, "height": 1, "clipiqa": 0.6, "niqe": 4.0},
            ]
            shuffled_rows = [
                {"dataset": "D", "filename": "1.png", "path": "s1", "width": 1, "height": 1, "clipiqa": 0.7, "niqe": 3.5},
                {"dataset": "D", "filename": "2.png", "path": "s2", "width": 1, "height": 1, "clipiqa": 0.5, "niqe": 4.5},
            ]
            for directory, rows in ((matched_dir, matched_rows), (shuffled_dir, shuffled_rows)):
                with (directory / "per_image_scores.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

            _, json_path = compare_pairing_metrics(matched_dir, shuffled_dir, output_dir)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            by_metric = {row["metric"]: row for row in payload["comparisons"]}
            self.assertAlmostEqual(by_metric["clipiqa"]["matched_advantage"], 0.1)
            self.assertAlmostEqual(by_metric["niqe"]["matched_advantage"], 0.5)
            self.assertEqual(by_metric["niqe"]["matched_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
