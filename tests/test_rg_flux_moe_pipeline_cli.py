import datetime
import json
import tempfile
import unittest
from pathlib import Path

import yaml


class RGFluxMoEPipelineCliTests(unittest.TestCase):
    def test_cli_accepts_single_lora_sources_and_rejects_conflict(self):
        from tools.run_rg_flux_moe_pipeline import build_arg_parser, parse_args

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--moe_config",
                "configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml",
                "--single_lora_checkpoint",
                "exp/single/checkpoints/checkpoint-00032000/rg_flux_adapters",
                "--checkpoint_steps",
                "20000",
                "40000",
                "--dataset_dirs",
                "realLQ250=/data/RealLQ250/lq",
                "realLR200=/data/RealLR200/lq",
                "--inference_output_root",
                "eval/inference/moe",
            ]
        )
        self.assertEqual(args.single_lora_checkpoint, "exp/single/checkpoints/checkpoint-00032000/rg_flux_adapters")
        self.assertIsNone(args.single_lora_run_dir)

        args = parser.parse_args(
            [
                "--moe_config",
                "configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml",
                "--single_lora_run_dir",
                "exp/single",
                "--single_lora_checkpoint_step",
                "32000",
                "--checkpoint_steps",
                "latest",
                "--dataset_dirs",
                "realLQ250=/data/RealLQ250/lq",
                "--inference_output_root",
                "eval/inference/moe",
            ]
        )
        self.assertEqual(args.single_lora_checkpoint_step, "32000")

        args = parser.parse_args(
            [
                "--skip_stage1",
                "--skip_train",
                "--moe_run_dir",
                "exp/moe_run",
                "--checkpoint_steps",
                "20000",
                "--dataset_dirs",
                "realLQ250=/data/RealLQ250/lq",
                "--inference_output_root",
                "eval/inference/moe",
            ]
        )
        self.assertTrue(args.skip_stage1)
        self.assertTrue(args.skip_train)

        with self.assertRaises(ValueError):
            parse_args(
                [
                    "--moe_config",
                    "configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml",
                    "--single_lora_checkpoint",
                    "exp/single/checkpoints/checkpoint-00032000/rg_flux_adapters",
                    "--single_lora_run_dir",
                    "exp/single",
                    "--checkpoint_steps",
                    "20000",
                    "--dataset_dirs",
                    "realLQ250=/data/RealLQ250/lq",
                    "--inference_output_root",
                    "eval/inference/moe",
                ]
            )

    def test_resolves_single_lora_checkpoint_paths(self):
        from tools.run_rg_flux_moe_pipeline import resolve_single_lora_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "exp" / "single_run"
            ckpt_32000 = run_dir / "checkpoints" / "checkpoint-00032000" / "rg_flux_adapters"
            ckpt_64000 = run_dir / "checkpoints" / "checkpoint-00064000" / "rg_flux_adapters"
            ckpt_32000.mkdir(parents=True)
            ckpt_64000.mkdir(parents=True)

            self.assertEqual(
                resolve_single_lora_checkpoint(single_lora_checkpoint=str(ckpt_32000)),
                ckpt_32000,
            )
            self.assertEqual(
                resolve_single_lora_checkpoint(
                    single_lora_run_dir=str(run_dir),
                    single_lora_checkpoint_step="32000",
                ),
                ckpt_32000,
            )
            self.assertEqual(
                resolve_single_lora_checkpoint(
                    single_lora_run_dir=str(run_dir),
                    single_lora_checkpoint_step="latest",
                ),
                ckpt_64000,
            )
            with self.assertRaises(FileNotFoundError):
                resolve_single_lora_checkpoint(
                    single_lora_run_dir=str(run_dir),
                    single_lora_checkpoint_step="96000",
                )

    def test_moe_runtime_config_injects_stage1_resume_and_text_overrides(self):
        from tools.run_rg_flux_moe_pipeline import create_moe_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            moe_config_path = root / "moe.yaml"
            source_config = {
                "model": {"flux_backend": "flux2_klein", "lora_backend": "moe"},
                "text_encoding": {"mode": "cached", "cache_dir": "old_cache", "dtype": "bf16"},
                "condition": {"lr_cond_mode": "flux2_image_concat", "use_prompt": True, "use_suggestions": True},
                "data": {"crop_size": 512},
                "training": {
                    "stage": "0B",
                    "suffix": "_moe_stage0b512",
                    "output_dir": str(root / "exp_rg_flux_sr"),
                    "resume_ckpt": None,
                },
            }
            moe_config_path.write_text(yaml.safe_dump(source_config), encoding="utf-8")

            class Args:
                text_encoding_mode = "cached"
                text_embedding_cache = "datasets/text_embed_cache/fixed_prompt"
                use_prompt = False
                use_suggestions = False
                use_degradation_vector = None

            runtime_config, run_dir, runtime_path, stage1_output = create_moe_runtime_config(
                moe_config_path,
                Args,
                now=datetime.datetime(2026, 6, 30, 9, 15),
            )

            original_after = yaml.safe_load(moe_config_path.read_text(encoding="utf-8"))
            self.assertNotIn("exp_name", original_after["training"])
            self.assertIsNone(original_after["training"]["resume_ckpt"])
            self.assertEqual(runtime_config["model"]["flux_backend"], "flux2_klein")
            self.assertEqual(runtime_config["model"]["lora_backend"], "moe")
            self.assertTrue(runtime_config["training"]["exp_name"].endswith("_26063009"))
            self.assertEqual(runtime_config["training"]["resume_ckpt"], str(stage1_output))
            self.assertFalse(runtime_config["training"]["resume_training_state"])
            self.assertEqual(runtime_config["text_encoding"]["cache_dir"], "datasets/text_embed_cache/fixed_prompt")
            self.assertFalse(runtime_config["condition"]["use_prompt"])
            self.assertFalse(runtime_config["condition"]["use_suggestions"])
            self.assertEqual(runtime_path, run_dir / "pipeline_runtime_config.yaml")
            self.assertEqual(stage1_output, run_dir / "stage1_init" / "rg_flux_adapters")
            self.assertTrue(runtime_path.exists())

    def test_builds_stage1_train_inference_and_eval_commands(self):
        from tools.run_rg_flux_moe_pipeline import build_stage1_command
        from tools.run_rg_flux_pipeline import build_eval_command, build_inference_command, build_train_command

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "exp" / "moe_run"
            runtime_config = run_dir / "pipeline_runtime_config.yaml"
            stage1_output = run_dir / "stage1_init" / "rg_flux_adapters"
            single_lora = root / "single" / "checkpoints" / "checkpoint-00032000" / "rg_flux_adapters"
            checkpoint_dir = run_dir / "checkpoints" / "checkpoint-00020000"
            runtime_config.parent.mkdir(parents=True)
            runtime_config.write_text("training: {}\n", encoding="utf-8")
            single_lora.mkdir(parents=True)
            (checkpoint_dir / "rg_flux_adapters").mkdir(parents=True)

            class Args:
                accelerate_config = "configs/accelerate/zero3_bf16_cpu_offload.yaml"
                num_processes = 1
                prototype_num_samples = 128
                perturb_scale = 0.01
                init_device = "cuda"
                dtype = "bf16"
                dry_run_train = False
                dataset_dirs = ["realLQ250=/data/RealLQ250/lq", "realLR200=/data/RealLR200/lq"]
                inference_output_root = str(root / "eval" / "inference")
                text_encoding_mode = "cached"
                text_embedding_cache = "datasets/text_embed_cache/fixed_prompt"
                jsonl_path = None
                num_inference_steps = 25
                upscale = 4
                device = None
                lr_cond_mode = None
                min_size = None
                restore_input_size = False
                use_prompt = False
                use_suggestions = False
                use_degradation_vector = None
                metrics = ["clipiqa", "niqe"]
                metric_device = "cuda"

            stage1_cmd = build_stage1_command(Args, runtime_config, single_lora, stage1_output)
            self.assertIn("tools/init_flux2_lora_moe.py", stage1_cmd)
            self.assertIn("--single_lora_checkpoint", stage1_cmd)
            self.assertIn(str(single_lora), stage1_cmd)
            self.assertIn("--output", stage1_cmd)
            self.assertIn(str(stage1_output), stage1_cmd)

            train_cmd = build_train_command(Args, runtime_config)
            self.assertIn("train_rg_flux_sr.py", train_cmd)
            self.assertIn(str(runtime_config), train_cmd)

            inference_cmd, inference_manifest = build_inference_command(Args, run_dir, checkpoint_dir, runtime_config)
            self.assertIn("--config", inference_cmd)
            self.assertIn(str(runtime_config), inference_cmd)
            self.assertIn("--no-use_prompt", inference_cmd)
            self.assertIn("--no-use_suggestions", inference_cmd)
            self.assertEqual(
                inference_manifest,
                root / "eval" / "inference" / run_dir.name / "checkpoint-00020000" / "inference_manifest.json",
            )

            eval_cmd, metrics_dir = build_eval_command(Args, inference_manifest)
            self.assertIn("--inference_manifest", eval_cmd)
            self.assertIn(str(inference_manifest), eval_cmd)
            self.assertEqual(metrics_dir, inference_manifest.parent / "metrics")

    def test_moe_pipeline_manifest_records_stages(self):
        from tools.run_rg_flux_moe_pipeline import write_moe_pipeline_manifest

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "moe_run"
            manifest_path = write_moe_pipeline_manifest(
                run_dir=run_dir,
                runtime_config_path=run_dir / "pipeline_runtime_config.yaml",
                single_lora_checkpoint=Path("single/rg_flux_adapters"),
                stage1_output=run_dir / "stage1_init" / "rg_flux_adapters",
                checkpoint_steps=["20000", "latest"],
                records=[{"checkpoint_step": "checkpoint-00020000"}],
                stage1_returncode=0,
                train_returncode=0,
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_dir"], str(run_dir))
            self.assertEqual(payload["single_lora_checkpoint"], str(Path("single/rg_flux_adapters")))
            self.assertEqual(payload["stage1_returncode"], 0)
            self.assertEqual(payload["train_returncode"], 0)
            self.assertEqual(payload["records"][0]["checkpoint_step"], "checkpoint-00020000")


if __name__ == "__main__":
    unittest.main()
