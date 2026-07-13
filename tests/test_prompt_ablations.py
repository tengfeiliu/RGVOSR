import argparse
import tempfile
import unittest
from pathlib import Path

import yaml


class PromptAblationTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "iqa": {
                "distortion_location": "Blur is visible around the subject.",
                "distortion_severity": "The blur is moderate.",
                "distortion_type": "Defocus blur.",
                "overall_quality": "Fine details are softened.",
            },
            "iaa": {"comprehensive": "Centered composition with a calm mood."},
            "suggestion": "Moderately restore natural edge sharpness.",
        }

    def test_suggestion_variant_prefixes_fixed_prompt_and_excludes_diagnosis_and_iaa(self):
        from models.prompt_builder import DEFAULT_SR_PROMPT, build_sr_prompt

        prompt = build_sr_prompt(self.profile, prompt_variant="suggestion")

        self.assertTrue(prompt.startswith(DEFAULT_SR_PROMPT))
        self.assertIn("Restoration suggestion:", prompt)
        self.assertIn(self.profile["suggestion"], prompt)
        self.assertNotIn("IQA profile:", prompt)
        self.assertNotIn("IAA comprehensive:", prompt)

    def test_iqa_variant_prefixes_fixed_prompt_and_excludes_suggestion_and_iaa(self):
        from models.prompt_builder import DEFAULT_SR_PROMPT, build_sr_prompt

        prompt = build_sr_prompt(self.profile, prompt_variant="iqa")

        self.assertTrue(prompt.startswith(DEFAULT_SR_PROMPT))
        self.assertIn("IQA profile:", prompt)
        self.assertIn("distortion_location:", prompt)
        self.assertNotIn("Restoration suggestion:", prompt)
        self.assertNotIn("IAA comprehensive:", prompt)

    def test_iqa_suggestion_variant_combines_both_after_fixed_prompt_without_iaa(self):
        from models.prompt_builder import DEFAULT_SR_PROMPT, build_sr_prompt

        prompt = build_sr_prompt(self.profile, prompt_variant="iqa_suggestion")

        self.assertTrue(prompt.startswith(DEFAULT_SR_PROMPT))
        self.assertIn("IQA profile:", prompt)
        self.assertIn("Restoration suggestion:", prompt)
        self.assertNotIn("IAA comprehensive:", prompt)

    def test_invalid_prompt_variant_is_rejected(self):
        from models.prompt_builder import build_sr_prompt

        with self.assertRaisesRegex(ValueError, "Unsupported prompt_variant"):
            build_sr_prompt(self.profile, prompt_variant="unknown")


class PromptAblationPipelineTests(unittest.TestCase):
    def test_ablation_suite_defaults_include_fixed_prompt_baseline(self):
        from tools.run_rg_flux_prompt_ablations import DEFAULT_VARIANTS, parse_args

        self.assertEqual(DEFAULT_VARIANTS, ("fixed", "suggestion", "iqa", "iqa_suggestion"))

        args = parse_args([
            "--",
            "--train_config",
            "configs/ablation.yaml",
            "--checkpoint_steps",
            "20000",
        ])

        self.assertEqual(args.variants, ["fixed", "suggestion", "iqa", "iqa_suggestion"])

    def test_ablation_suite_builds_one_pipeline_command_per_variant(self):
        from tools.run_rg_flux_prompt_ablations import build_variant_command

        command = build_variant_command(
            "fixed",
            [
                "--train_config",
                "configs/ablation.yaml",
                "--checkpoint_steps",
                "20000",
                "--dataset_dirs",
                "eval=/data/eval",
            ],
        )

        self.assertIn("tools/run_rg_flux_pipeline.py", command)
        self.assertIn("--prompt_variant", command)
        self.assertEqual(command[command.index("--prompt_variant") + 1], "fixed")
        self.assertIn("configs/ablation.yaml", command)

    def test_fixed_prompt_runtime_config_disables_dynamic_prompt_flags(self):
        from tools.run_rg_flux_pipeline import create_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "train.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"flux_backend": "flux2_klein"},
                        "text_encoding": {"mode": "online"},
                        "data": {"crop_size": 512},
                        "condition": {
                            "lr_cond_mode": "flux2_image_concat",
                            "use_prompt": True,
                            "use_suggestions": True,
                        },
                        "training": {
                            "stage": "0B",
                            "output_dir": str(root / "exp"),
                            "add_datetime_suffix": False,
                            "suffix": "_ablation",
                        },
                    }
                ),
                encoding="utf-8",
            )

            runtime_config, run_dir, _ = create_runtime_config(config_path, prompt_variant="fixed")

        self.assertEqual(runtime_config["condition"]["prompt_variant"], "fixed")
        self.assertFalse(runtime_config["condition"]["use_prompt"])
        self.assertFalse(runtime_config["condition"]["use_suggestions"])
        self.assertIn("prompt_fixed", run_dir.name)

    def test_runtime_config_and_inference_command_share_prompt_variant(self):
        from tools.run_rg_flux_pipeline import (
            build_inference_command,
            create_runtime_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "train.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"flux_backend": "flux2_klein"},
                        "text_encoding": {"mode": "online"},
                        "data": {"crop_size": 512},
                        "condition": {
                            "lr_cond_mode": "flux2_image_concat",
                            "use_prompt": False,
                            "use_suggestions": False,
                        },
                        "training": {
                            "stage": "0B",
                            "output_dir": str(root / "exp"),
                            "add_datetime_suffix": False,
                            "suffix": "_ablation",
                        },
                    }
                ),
                encoding="utf-8",
            )

            runtime_config, run_dir, runtime_path = create_runtime_config(
                config_path,
                prompt_variant="iqa_suggestion",
            )

            self.assertEqual(runtime_config["condition"]["prompt_variant"], "iqa_suggestion")
            self.assertTrue(runtime_config["condition"]["use_prompt"])
            self.assertTrue(runtime_config["condition"]["use_suggestions"])
            self.assertIn("prompt_iqa_suggestion", run_dir.name)

            args = argparse.Namespace(
                dataset_dirs=["eval=/data/eval"],
                inference_output_root=None,
                text_encoding_mode="online",
                text_embedding_cache=None,
                jsonl_path="datasets/inference_cleaned.jsonl",
                num_inference_steps=25,
                upscale=4,
                dtype="bf16",
                device=None,
                lr_cond_mode=None,
                min_size=None,
                use_prompt=True,
                use_suggestions=True,
                use_degradation_vector=False,
                prompt_variant="iqa_suggestion",
                restore_input_size=False,
            )
            checkpoint_dir = run_dir / "checkpoints" / "checkpoint-00020000"
            cmd, _ = build_inference_command(
                args,
                run_dir,
                checkpoint_dir,
                runtime_path,
            )

            self.assertIn("--prompt_variant", cmd)
            self.assertEqual(cmd[cmd.index("--prompt_variant") + 1], "iqa_suggestion")

    def test_runtime_prompt_variant_updates_curriculum_after_variant(self):
        from tools.run_rg_flux_pipeline import create_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "train.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"flux_backend": "flux2_klein"},
                        "text_encoding": {"mode": "online"},
                        "data": {"crop_size": 512},
                        "condition": {
                            "lr_cond_mode": "flux2_image_concat",
                            "prompt_variant": "suggestion",
                            "prompt_schedule": {
                                "enabled": True,
                                "switch_step": 10000,
                                "before_variant": "fixed",
                                "after_variant": "suggestion",
                            },
                        },
                        "training": {
                            "stage": "0B",
                            "output_dir": str(root / "exp"),
                            "add_datetime_suffix": False,
                            "suffix": "_curriculum",
                        },
                    }
                ),
                encoding="utf-8",
            )

            runtime_config, run_dir, _ = create_runtime_config(
                config_path,
                prompt_variant="iqa_suggestion",
            )

        condition = runtime_config["condition"]
        self.assertEqual(condition["prompt_variant"], "iqa_suggestion")
        self.assertEqual(condition["prompt_schedule"]["before_variant"], "fixed")
        self.assertEqual(condition["prompt_schedule"]["after_variant"], "iqa_suggestion")
        self.assertTrue(condition["use_prompt"])
        self.assertTrue(condition["use_suggestions"])
        self.assertIn("prompt_iqa_suggestion", run_dir.name)


if __name__ == "__main__":
    unittest.main()
