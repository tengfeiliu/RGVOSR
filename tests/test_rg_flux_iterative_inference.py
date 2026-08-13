import argparse
import csv
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml
from PIL import Image


def import_iterative_module():
    """Import the orchestration module without requiring the 4B model runtime."""
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "fp32"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: None,
    )
    fake_torch.device = lambda value: value
    fake_torch.manual_seed = lambda seed: None
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.random = types.SimpleNamespace(seed=lambda seed: None)

    fake_inference = types.ModuleType("inference_rg_flux_sr")

    def cfg(config, path, default=None):
        current = config
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def build_single_pass_arg_parser():
        parser = argparse.ArgumentParser()
        inputs = parser.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--input", default=None)
        inputs.add_argument("--dataset_dirs", nargs="+", default=None)
        parser.add_argument("--output_dir", default=None)
        parser.add_argument("--output_root", default=None)
        parser.add_argument("--checkpoint", default=None)
        parser.add_argument("--run_dir", default=None)
        parser.add_argument("--checkpoint_step", default=None)
        parser.add_argument("--config", default=None)
        parser.add_argument("--jsonl_path", default=None)
        parser.add_argument("--text_encoding_mode", default=None)
        parser.add_argument("--text_embedding_cache", default=None)
        parser.add_argument("--num_inference_steps", type=int, default=25)
        parser.add_argument("--inference_schedule", default=None)
        parser.add_argument("--inference_init_mode", default=None)
        parser.add_argument("--inference_sigma_start", type=float, default=None)
        parser.add_argument("--lr_cond_mode", default=None)
        parser.add_argument("--use_prompt", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--use_suggestions", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--prompt_variant", default=None)
        parser.add_argument("--include_caption", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--suggestion_pairing", default="matched")
        parser.add_argument("--suggestion_shuffle_seed", type=int, default=3407)
        parser.add_argument("--iqa_pairing", default=None)
        parser.add_argument("--iqa_shuffle_seed", type=int, default=3407)
        parser.add_argument(
            "--use_degradation_vector",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", default=None)
        parser.add_argument("--dtype", default="bf16")
        parser.add_argument("--upscale", type=int, default=4)
        parser.add_argument("--min_size", type=int, default=None)
        parser.add_argument("--restore_input_size", action="store_true")
        parser.add_argument("--full_frame_inference", action="store_true")
        return parser

    def parse_dataset_dirs(values):
        return [(name, Path(path)) for name, path in (value.split("=", 1) for value in values)]

    def list_images(input_path):
        input_path = Path(input_path)
        if input_path.is_file():
            return [input_path]
        return sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))

    def aliases(value, dataset_name=None, input_root=None):
        value = Path(value)
        result = [str(value).replace("\\", "/"), value.name]
        if dataset_name:
            result.append(f"{dataset_name}/{value.name}")
        return result

    def condition_for_image(index, path, dataset_name=None, input_root=None):
        for alias in aliases(path, dataset_name=dataset_name, input_root=input_root):
            if alias in index:
                return index[alias]
        return None

    def resolve_inference_run(args):
        return {
            "run_dir": Path(args.run_dir) if args.run_dir else None,
            "checkpoint": Path(args.checkpoint),
            "checkpoint_step": "checkpoint-test",
            "output_dir": Path(args.output_dir),
        }

    def load_config(checkpoint, explicit_config=None):
        return yaml.safe_load(Path(explicit_config).read_text(encoding="utf-8"))

    def write_inference_manifest(manifest_path, **kwargs):
        Path(manifest_path).write_text(json.dumps(kwargs, default=str), encoding="utf-8")

    fake_inference.build_arg_parser = build_single_pass_arg_parser
    fake_inference.cfg = cfg
    fake_inference.condition_for_image = condition_for_image
    fake_inference.image_lookup_aliases = aliases
    fake_inference.list_images = list_images
    fake_inference.load_config = load_config
    fake_inference.load_jsonl_conditions = lambda path: {}
    fake_inference.normalize_iqa_pairing = lambda value: value
    fake_inference.normalize_suggestion_pairing = lambda value: value
    fake_inference.parse_dataset_dirs = parse_dataset_dirs
    fake_inference.resolve_inference_dtype = lambda config, dtype: ("fp32", "fp32")
    fake_inference.resolve_inference_run = resolve_inference_run
    fake_inference.run_inference_dataset = lambda **kwargs: {}
    fake_inference.write_inference_manifest = write_inference_manifest

    fake_artist_factory = types.ModuleType("models.rg_flux_artist_factory")
    fake_artist_factory.build_rg_flux_artist = lambda config: FakeArtist()
    fake_text_cache = types.ModuleType("models.text_embedding_cache")
    fake_text_cache.get_text_embedding_cache = lambda config, dtype=None: None

    sys.modules.pop("tools.run_rg_flux_iterative_inference", None)
    tools_package = sys.modules.get("tools")
    if tools_package is not None:
        tools_package.__dict__.pop("run_rg_flux_iterative_inference", None)
    with mock.patch.dict(
        "sys.modules",
        {
            "torch": fake_torch,
            "numpy": fake_numpy,
            "inference_rg_flux_sr": fake_inference,
            "models.rg_flux_artist_factory": fake_artist_factory,
            "models.text_embedding_cache": fake_text_cache,
        },
    ):
        return importlib.import_module("tools.run_rg_flux_iterative_inference")


class FakeArtist:
    def __init__(self):
        self.loaded = 0
        self.aligned = 0
        self.eval_calls = 0

    def to(self, device=None):
        self.device = device
        return self

    def load_trainable(self, checkpoint, is_trainable=False):
        self.loaded += 1

    def align_inference_dtype(self, dtype=None):
        self.aligned += 1

    def eval(self):
        self.eval_calls += 1
        return self

    def set_moe_inference_schedule(self):
        return {"enabled": False}


class RGFluxIterativeInferenceTests(unittest.TestCase):
    def test_parser_exposes_iterative_options(self):
        iterative = import_iterative_module()

        args = iterative.build_arg_parser().parse_args(
            [
                "--input",
                "input.png",
                "--checkpoint",
                "checkpoint",
                "--output_dir",
                "output",
                "--iterations",
                "4",
                "--round_seed_mode",
                "increment",
                "--metrics",
                "clipiqa",
                "niqe",
            ]
        )
        self.assertEqual(args.iterations, 4)
        self.assertEqual(args.round_seed_mode, "increment")
        self.assertEqual(args.metrics, ["clipiqa", "niqe"])
        self.assertEqual(args.metric_device, "cpu")

    def test_generated_round_path_inherits_original_jsonl_condition(self):
        iterative = import_iterative_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            current_root = root / "round_01" / "demo"
            source_root.mkdir()
            current_root.mkdir(parents=True)
            source_path = source_root / "sample.jpg"
            current_path = current_root / "sample.png"
            Image.new("RGB", (8, 8)).save(source_path)
            Image.new("RGB", (8, 8)).save(current_path)
            condition = {"profile": {"caption": "original caption"}, "result": {}}
            base_index = {str(source_path).replace("\\", "/"): condition}
            lineage = [
                {
                    "dataset": "demo",
                    "sample_id": "sample",
                    "source_path": str(source_path),
                    "source_input_root": str(source_root),
                    "rounds": [
                        {"round": 1, "path": str(current_path), "exists": True}
                    ],
                }
            ]

            inherited = iterative._build_round_condition_index(
                base_index,
                lineage,
                {"demo": current_root},
            )
            normalized = str(current_path).replace("\\", "/")
            self.assertIs(inherited[normalized], condition)

    def test_runs_first_round_at_requested_scale_then_fixed_resolution(self):
        iterative = import_iterative_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.png"
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(input_path)
            checkpoint = root / "checkpoint" / "rg_flux_adapters"
            checkpoint.mkdir(parents=True)
            output_root = root / "iterative_output"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"dtype": "fp32"},
                        "data": {"pre_cropped": False, "vae_align": 16},
                        "condition": {
                            "lr_cond_mode": "flux2_image_concat",
                            "use_degradation_vector": False,
                            "include_caption": False,
                        },
                        "text_encoding": {"mode": "online", "dtype": "fp32"},
                        "flow_matching": {
                            "inference_schedule": "linear",
                            "inference_init_mode": "pure_noise",
                            "inference_sigma_start": 1.0,
                        },
                        "evaluation": {"metrics": ["niqe"]},
                    }
                ),
                encoding="utf-8",
            )
            args = iterative.build_arg_parser().parse_args(
                [
                    "--input",
                    str(input_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--output_dir",
                    str(output_root),
                    "--config",
                    str(config_path),
                    "--iterations",
                    "3",
                    "--upscale",
                    "4",
                    "--dtype",
                    "fp32",
                    "--device",
                    "cpu",
                    "--round_seed_mode",
                    "increment",
                ]
            )

            fake_artist = FakeArtist()
            observed_upscales = []
            observed_inputs = []

            def fake_inference_dataset(**kwargs):
                observed_upscales.append(kwargs["args"].upscale)
                observed_inputs.append(Path(kwargs["input_path"]))
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                for image_path in iterative.list_images(kwargs["input_path"]):
                    with Image.open(image_path) as image:
                        image.convert("RGB").save(output_dir / f"{image_path.stem}.png")
                return {"valid_image_count": 1, "skipped_image_count": 0}

            metric_round = {"value": 0}

            def fake_metrics(dataset_dirs, output_dir, metrics, device):
                metric_round["value"] += 1
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "metric_directions": {"niqe": "lower_better"},
                    "summary": [
                        {
                            "dataset": "default",
                            "metric": "niqe",
                            "mean": float(metric_round["value"]),
                            "std": 0.0,
                            "count": 1,
                        }
                    ],
                }
                (output_dir / "summary_scores.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                return payload

            with (
                mock.patch.object(iterative, "build_rg_flux_artist", return_value=fake_artist),
                mock.patch.object(iterative, "get_text_embedding_cache", return_value=None),
                mock.patch.object(iterative, "run_inference_dataset", side_effect=fake_inference_dataset),
                mock.patch.object(iterative, "evaluate_dataset_dirs", side_effect=fake_metrics) as eval_mock,
            ):
                manifest = iterative.run_iterative_inference(args)

            self.assertEqual(observed_upscales, [4, 1, 1])
            self.assertEqual(observed_inputs[0], input_path)
            self.assertEqual(observed_inputs[1], output_root / "round_01" / "default")
            self.assertEqual(observed_inputs[2], output_root / "round_02" / "default")
            self.assertEqual(fake_artist.loaded, 1)
            self.assertEqual(eval_mock.call_count, 3)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual([row["seed"] for row in manifest["rounds"]], [42, 43, 44])

            saved_manifest = json.loads(
                (output_root / "iterative_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["status"], "completed")
            self.assertEqual(len(saved_manifest["rounds"]), 3)
            with (output_root / "metric_trends.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                trend_rows = list(csv.DictReader(handle))
            self.assertEqual([row["round"] for row in trend_rows], ["1", "2", "3"])

    def test_refuses_to_overwrite_existing_round_outputs(self):
        iterative = import_iterative_module()

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "output"
            (output_root / "round_01").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                iterative._validate_output_root(output_root)

    def test_rejects_claimed_upscale_when_precropped_semantics_would_ignore_it(self):
        iterative = import_iterative_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.png"
            Image.new("RGB", (8, 8)).save(input_path)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"dtype": "fp32"},
                        "data": {"pre_cropped": True},
                        "condition": {
                            "lr_cond_mode": "flux2_image_concat",
                            "use_degradation_vector": False,
                        },
                        "text_encoding": {"mode": "online"},
                    }
                ),
                encoding="utf-8",
            )
            args = iterative.build_arg_parser().parse_args(
                [
                    "--input",
                    str(input_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--output_dir",
                    str(root / "output"),
                    "--config",
                    str(config_path),
                    "--upscale",
                    "4",
                    "--dtype",
                    "fp32",
                ]
            )
            with self.assertRaisesRegex(ValueError, "full_frame_inference"):
                iterative.run_iterative_inference(args)


if __name__ == "__main__":
    unittest.main()
