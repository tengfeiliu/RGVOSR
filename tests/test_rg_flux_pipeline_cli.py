import datetime
import json
import tempfile
import unittest
from pathlib import Path

import yaml


class RGFluxPipelineCliTests(unittest.TestCase):
    def test_cli_parses_checkpoint_steps_dataset_dirs_and_skip_train(self):
        from tools.run_rg_flux_pipeline import build_arg_parser, parse_dataset_dirs

        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--skip_train",
                "--run_dir",
                "exp_rg_flux_sr/run",
                "--checkpoint_steps",
                "20000",
                "40000",
                "--dataset_dirs",
                "realLQ250=/data/RealLQ250/lq",
                "realLR200=/data/RealLR200/lq",
                "--inference_output_root",
                "eval/inference",
                "--resume_checkpoint",
                "exp/single/checkpoints/checkpoint-00024000",
                "--no-resume_training_state",
                "--max_steps",
                "64000",
                "--grad_accum_steps",
                "8",
                "--image_loss_crop_size",
                "256",
                "--stage_label",
                "_s0_control",
            ]
        )

        self.assertTrue(args.skip_train)
        self.assertEqual(args.checkpoint_steps, ["20000", "40000"])
        self.assertEqual(args.max_steps, 64000)
        self.assertFalse(args.resume_training_state)
        self.assertEqual(args.grad_accum_steps, 8)
        self.assertEqual(args.image_loss_crop_size, 256)
        self.assertEqual(args.stage_label, "_s0_control")
        self.assertEqual(
            parse_dataset_dirs(args.dataset_dirs),
            {
                "realLQ250": Path("/data/RealLQ250/lq"),
                "realLR200": Path("/data/RealLR200/lq"),
            },
        )

    def test_runtime_config_writes_fixed_exp_name_without_mutating_original(self):
        from tools.run_rg_flux_pipeline import create_runtime_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_config = root / "train.yaml"
            config = {
                "model": {"flux_backend": "flux2_klein"},
                "condition": {"lr_cond_mode": "flux2_image_concat"},
                "data": {"crop_size": 512},
                "training": {
                    "stage": "0B",
                    "suffix": "_stage0b512",
                    "output_dir": str(root / "exp_rg_flux_sr"),
                },
            }
            train_config.write_text(yaml.safe_dump(config), encoding="utf-8")

            runtime_config, run_dir, runtime_path = create_runtime_config(
                train_config,
                now=datetime.datetime(2026, 6, 28, 10, 5),
                resume_checkpoint="exp/single/checkpoints/checkpoint-00024000",
                resume_training_state=False,
                max_steps=64000,
                grad_accum_steps=8,
                image_loss_crop_size=256,
                stage_label="_s0_single_continue_control",
            )

            original_after = yaml.safe_load(train_config.read_text(encoding="utf-8"))
            self.assertNotIn("exp_name", original_after["training"])
            self.assertTrue(runtime_config["training"]["exp_name"].endswith("_26062810"))
            self.assertEqual(runtime_config["training"]["resolved_exp_name"], runtime_config["training"]["exp_name"])
            self.assertEqual(runtime_config["training"]["resolved_run_id"], "26062810")
            self.assertEqual(
                runtime_config["training"]["resume_ckpt"],
                "exp/single/checkpoints/checkpoint-00024000",
            )
            self.assertFalse(runtime_config["training"]["resume_training_state"])
            self.assertFalse(runtime_config["training"]["auto_resume"])
            self.assertEqual(runtime_config["training"]["max_steps"], 64000)
            self.assertEqual(runtime_config["training"]["grad_accum_steps"], 8)
            self.assertEqual(runtime_config["loss"]["image_loss_crop_size"], 256)
            self.assertIn("_s0_single_continue_control", run_dir.name)
            self.assertEqual(run_dir.name, runtime_config["training"]["exp_name"])
            self.assertEqual(runtime_path, run_dir / "pipeline_runtime_config.yaml")
            self.assertTrue(runtime_path.exists())

    def test_builds_train_inference_and_eval_commands_for_each_step(self):
        from tools.run_rg_flux_pipeline import (
            build_eval_command,
            build_bad_case_command,
            build_inference_command,
            build_train_command,
            checkpoint_artifact_paths,
            resolve_checkpoint_dir,
            write_run_summary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "exp" / "run_26062810"
            for step in ("checkpoint-00020000", "checkpoint-00040000"):
                (run_dir / "checkpoints" / step / "rg_flux_adapters").mkdir(parents=True)
            runtime_config = run_dir / "pipeline_runtime_config.yaml"
            runtime_config.parent.mkdir(parents=True, exist_ok=True)
            runtime_config.write_text("training: {}\n", encoding="utf-8")

            class Args:
                accelerate_config = "configs/accelerate/zero3_bf16_cpu_offload.yaml"
                num_processes = 1
                dataset_dirs = ["realLQ250=/data/RealLQ250/lq", "realLR200=/data/RealLR200/lq"]
                inference_output_root = str(root / "eval" / "inference")
                text_encoding_mode = "cached"
                text_embedding_cache = "datasets/text_embed_cache/flux2_klein_fixed_sr_prompt"
                num_inference_steps = 25
                upscale = 4
                dtype = "bf16"
                device = None
                jsonl_path = None
                lr_cond_mode = None
                min_size = None
                restore_input_size = False
                full_frame_inference = True
                use_prompt = None
                use_suggestions = None
                use_degradation_vector = None
                metrics = ["clipiqa", "niqe"]
                metric_device = "cuda"

            train_cmd = build_train_command(Args, runtime_config)
            self.assertIn("accelerate", train_cmd)
            self.assertIn("--config_file", train_cmd)
            self.assertIn(str(runtime_config), train_cmd)

            checkpoint_dir = resolve_checkpoint_dir(run_dir, "20000")
            inference_cmd, inference_manifest = build_inference_command(Args, run_dir, checkpoint_dir, runtime_config)
            self.assertIn("--run_dir", inference_cmd)
            self.assertIn(str(run_dir), inference_cmd)
            self.assertIn("--config", inference_cmd)
            self.assertIn(str(runtime_config), inference_cmd)
            self.assertIn("--checkpoint_step", inference_cmd)
            self.assertIn("checkpoint-00020000", inference_cmd)
            self.assertIn("--dataset_dirs", inference_cmd)
            self.assertIn("realLR200=/data/RealLR200/lq", inference_cmd)
            self.assertIn("--full_frame_inference", inference_cmd)
            self.assertEqual(
                inference_manifest,
                root / "eval" / "inference" / run_dir.name / "checkpoint-00020000" / "inference_manifest.json",
            )

            eval_cmd, metrics_dir = build_eval_command(Args, inference_manifest)
            self.assertIn("--inference_manifest", eval_cmd)
            self.assertIn(str(inference_manifest), eval_cmd)
            self.assertIn("--metrics", eval_cmd)
            self.assertIn("clipiqa", eval_cmd)
            self.assertEqual(metrics_dir, inference_manifest.parent / "metrics")
            self.assertIn("--output_dir", eval_cmd)
            self.assertIn(str(metrics_dir), eval_cmd)

            class RunContainedArgs(Args):
                inference_output_root = None
                bad_case_metrics = ["clipiqa", "maniqa"]
                bad_case_mode = "joint_mean"
                bad_case_worst_k = 25

            contained_cmd, contained_manifest = build_inference_command(
                RunContainedArgs,
                run_dir,
                checkpoint_dir,
                runtime_config,
            )
            artifact_paths = checkpoint_artifact_paths(run_dir, "checkpoint-00020000")
            self.assertIn("--output_dir", contained_cmd)
            self.assertIn(str(artifact_paths["inference_dir"]), contained_cmd)
            self.assertNotIn("--output_root", contained_cmd)
            self.assertEqual(contained_manifest, artifact_paths["inference_manifest"])

            contained_eval_cmd, contained_metrics_dir = build_eval_command(
                RunContainedArgs,
                contained_manifest,
                artifact_paths["metrics_dir"],
            )
            self.assertEqual(contained_metrics_dir, artifact_paths["metrics_dir"])
            self.assertIn(str(artifact_paths["metrics_dir"]), contained_eval_cmd)

            bad_case_cmd = build_bad_case_command(
                RunContainedArgs,
                contained_metrics_dir,
                artifact_paths["bad_cases_dir"],
                inference_manifest=contained_manifest,
            )
            self.assertIn("tools/analyze_rg_flux_bad_cases.py", bad_case_cmd)
            self.assertIn("--lq_dirs", bad_case_cmd)
            self.assertIn("--inference_manifest", bad_case_cmd)
            self.assertIn(str(contained_manifest), bad_case_cmd)
            self.assertIn("--font_size", bad_case_cmd)
            self.assertIn("40", bad_case_cmd)
            self.assertIn("realLQ250=/data/RealLQ250/lq", bad_case_cmd)
            self.assertIn(str(artifact_paths["bad_cases_dir"]), bad_case_cmd)

            summary_path = write_run_summary(
                run_dir,
                runtime_config,
                [
                    {
                        "checkpoint_step": "checkpoint-00020000",
                        "checkpoint_path": str(artifact_paths["checkpoint_path"]),
                        "inference_manifest": str(artifact_paths["inference_manifest"]),
                        "inference_output_dir": str(artifact_paths["inference_dir"]),
                        "metrics_output_dir": str(artifact_paths["metrics_dir"]),
                        "bad_cases_output_dir": str(artifact_paths["bad_cases_dir"]),
                    }
                ],
                pipeline_manifest_path=run_dir / "pipeline_manifest.json",
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["checkpoints"]["checkpoint-00020000"]["metrics_output_dir"],
                str(artifact_paths["metrics_dir"]),
            )

    def test_latest_and_missing_checkpoint_validation(self):
        from tools.run_rg_flux_pipeline import resolve_checkpoint_dir

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "checkpoints" / "checkpoint-00020000" / "rg_flux_adapters").mkdir(parents=True)
            (run_dir / "checkpoints" / "checkpoint-00040000" / "rg_flux_adapters").mkdir(parents=True)

            self.assertEqual(
                resolve_checkpoint_dir(run_dir, "latest"),
                run_dir / "checkpoints" / "checkpoint-00040000",
            )
            self.assertEqual(
                resolve_checkpoint_dir(run_dir, "20000"),
                run_dir / "checkpoints" / "checkpoint-00020000",
            )
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint_dir(run_dir, "60000")

    def test_pipeline_manifest_records_step_statuses(self):
        from tools.run_rg_flux_pipeline import write_pipeline_manifest

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            manifest_path = write_pipeline_manifest(
                run_dir=run_dir,
                runtime_config_path=run_dir / "pipeline_runtime_config.yaml",
                checkpoint_steps=["20000", "40000"],
                records=[
                    {
                        "checkpoint_step": "checkpoint-00020000",
                        "checkpoint_path": str(run_dir / "checkpoints" / "checkpoint-00020000" / "rg_flux_adapters"),
                        "inference_manifest": str(run_dir / "infer" / "inference_manifest.json"),
                        "metrics_output_dir": str(run_dir / "infer" / "metrics"),
                        "inference_returncode": 0,
                        "eval_returncode": 0,
                    }
                ],
                train_returncode=0,
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_dir"], str(run_dir))
            self.assertEqual(payload["checkpoint_steps"], ["20000", "40000"])
            self.assertEqual(payload["train_returncode"], 0)
            self.assertEqual(payload["records"][0]["checkpoint_step"], "checkpoint-00020000")


if __name__ == "__main__":
    unittest.main()
