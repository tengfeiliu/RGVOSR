import json
import argparse
import ast
import copy
import csv
import datetime
import inspect
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None


class RGFluxSRComponentTests(unittest.TestCase):
    def _make_pair(self, root: Path):
        hq_path = root / "hq.png"
        lq_path = root / "lq.png"
        Image.new("RGB", (64, 64), color=(128, 96, 64)).save(hq_path)
        Image.new("RGB", (16, 16), color=(64, 96, 128)).save(lq_path)
        return hq_path, lq_path

    def test_prompt_builder_uses_cleaned_profile_fields_in_order(self):
        from models.prompt_builder import build_sr_prompt

        profile = {
            "iqa": {
                "distortion_location": "IQA location: blur across the frame.",
                "distortion_severity": "IQA severity: moderate detail loss.",
                "distortion_type": "IQA type: blur and JPEG artifacts.",
                "overall_quality": "IQA overall: fidelity is limited.",
            },
            "suggestion": "Suggestion: recover fine textures.",
            "iaa": {"comprehensive": "IAA comprehensive: simple balanced composition."},
            "reasoning": {"degradation_analysis": "old result reasoning must not be used"},
        }

        prompt = build_sr_prompt(profile, use_prompt=True, use_suggestions=True)

        self.assertIn("distortion_location:", prompt)
        self.assertIn("IQA location: blur across the frame.", prompt)
        self.assertIn("IQA severity: moderate detail loss.", prompt)
        self.assertIn("IQA type: blur and JPEG artifacts.", prompt)
        self.assertIn("IQA overall: fidelity is limited.", prompt)
        self.assertIn("Suggestion: recover fine textures.", prompt)
        self.assertIn("IAA comprehensive: simple balanced composition.", prompt)
        self.assertNotIn("old result reasoning must not be used", prompt)
        self.assertIn("Avoid hallucinated details", prompt)

        iqa_index = prompt.index("IQA location")
        suggestion_index = prompt.index("Suggestion: recover fine textures")
        iaa_index = prompt.index("IAA comprehensive")
        requirements_index = prompt.index("Requirements:")
        self.assertLess(iqa_index, suggestion_index)
        self.assertLess(suggestion_index, iaa_index)
        self.assertLess(iaa_index, requirements_index)

    def test_flux_artist_to_does_not_move_text_pipeline(self):
        source = Path("models/flux_sr_artist.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        flux_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FluxSRArtist"
        )
        to_func = next(node for node in flux_class.body if isinstance(node, ast.FunctionDef) and node.name == "to")

        calls_text_pipeline_to = False
        for node in ast.walk(to_func):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "to":
                continue
            value = func.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "text_pipeline"
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"
            ):
                calls_text_pipeline_to = True

        self.assertFalse(calls_text_pipeline_to)

    def test_train_passes_resolved_zero3_config_without_global_hf_init(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        main_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        resolves_config_line = None
        calls_hf_ds_config = False
        artist_factory_line = None
        for node in ast.walk(main_func):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "HfDeepSpeedConfig":
                calls_hf_ds_config = True
            if isinstance(func, ast.Name) and func.id == "sync_deepspeed_config_for_training":
                resolves_config_line = node.lineno
            if isinstance(func, ast.Name) and func.id == "build_rg_flux_artist":
                artist_factory_line = node.lineno

        self.assertFalse(calls_hf_ds_config)
        self.assertIsNotNone(resolves_config_line)
        self.assertIsNotNone(artist_factory_line)
        self.assertLess(resolves_config_line, artist_factory_line)

    def test_flux_artist_scopes_hf_zero3_to_transformer_load_only(self):
        source = Path("models/flux_sr_artist.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        flux_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FluxSRArtist"
        )
        load_func = next(node for node in flux_class.body if isinstance(node, ast.FunctionDef) and node.name == "_load_flux_modules")

        hf_config_line = None
        transformer_load_line = None
        clear_line = None
        pipeline_load_line = None
        for node in ast.walk(load_func):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "HfDeepSpeedConfig":
                hf_config_line = node.lineno
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "from_pretrained"
                and isinstance(func.value, ast.Name)
                and func.value.id == "FluxTransformer2DModel"
            ):
                transformer_load_line = node.lineno
            if isinstance(func, ast.Name) and func.id == "_clear_hf_deepspeed_config":
                clear_line = node.lineno
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "from_pretrained"
                and isinstance(func.value, ast.Name)
                and func.value.id == "FluxPipeline"
            ):
                pipeline_load_line = node.lineno

        self.assertIsNotNone(hf_config_line)
        self.assertIsNotNone(transformer_load_line)
        self.assertIsNotNone(clear_line)
        self.assertIsNotNone(pipeline_load_line)
        self.assertLess(hf_config_line, transformer_load_line)
        self.assertLess(transformer_load_line, clear_line)
        self.assertLess(clear_line, pipeline_load_line)

    def test_zero3_cpu_offload_config_exists_for_two_gpu_smoke_test(self):
        config_path = Path("configs/accelerate/zero3_bf16_cpu_offload.yaml")
        self.assertTrue(config_path.exists())
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["distributed_type"], "DEEPSPEED")
        self.assertEqual(config["mixed_precision"], "bf16")
        self.assertEqual(config["deepspeed_config"]["zero_stage"], 3)
        self.assertEqual(config["deepspeed_config"]["offload_param_device"], "cpu")
        self.assertEqual(config["deepspeed_config"]["offload_optimizer_device"], "cpu")

    def test_zero3_param_offload_config_avoids_cpu_adam_for_flux2_smoke(self):
        config_path = Path("configs/accelerate/zero3_bf16_param_offload.yaml")
        self.assertTrue(config_path.exists())
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["distributed_type"], "DEEPSPEED")
        self.assertEqual(config["mixed_precision"], "bf16")
        self.assertEqual(config["num_processes"], 2)
        self.assertEqual(config["deepspeed_config"]["zero_stage"], 3)
        self.assertIn(config["deepspeed_config"]["offload_param_device"], {"cpu", "none"})
        self.assertEqual(config["deepspeed_config"]["offload_optimizer_device"], "none")
        self.assertGreaterEqual(int(config["deepspeed_config"]["gradient_accumulation_steps"]), 1)

    def test_hf_zero3_config_resolves_auto_batch_fields(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {"_deepspeed_auto_or_missing", "_deepspeed_int", "resolve_hf_zero3_config"}
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        helper = next((node for node in helpers if node.name == "resolve_hf_zero3_config"), None)
        self.assertIsNotNone(helper)

        namespace = {"copy": copy}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "train_rg_flux_sr.py", "exec"), namespace)
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "zero_optimization": {"stage": 3},
        }

        resolved = namespace["resolve_hf_zero3_config"](
            ds_config,
            per_device_batch=1,
            grad_accum_steps=8,
            num_processes=2,
        )

        self.assertEqual(resolved["train_micro_batch_size_per_gpu"], 1)
        self.assertEqual(resolved["gradient_accumulation_steps"], 8)
        self.assertEqual(resolved["train_batch_size"], 16)
        self.assertEqual(ds_config["train_batch_size"], "auto")

    def test_deepspeed_runtime_config_syncs_training_batch_and_disables_optimizer_offload(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {
            "_deepspeed_auto_or_missing",
            "_deepspeed_int",
            "_normalize_offload_device",
            "get_deepspeed_optimizer_offload_device",
            "resolve_hf_zero3_config",
            "set_deepspeed_optimizer_offload_device",
            "sync_deepspeed_config_for_training",
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]

        namespace = {"copy": copy}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "train_rg_flux_sr.py", "exec"), namespace)
        ds_config = {
            "gradient_accumulation_steps": 8,
            "offload_optimizer_device": "cpu",
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
            },
        }

        resolved = namespace["sync_deepspeed_config_for_training"](
            ds_config,
            per_device_batch=1,
            grad_accum_steps=1,
            num_processes=2,
            optimizer_offload_device="none",
        )

        self.assertIs(resolved, ds_config)
        self.assertEqual(ds_config["train_micro_batch_size_per_gpu"], 1)
        self.assertEqual(ds_config["gradient_accumulation_steps"], 1)
        self.assertEqual(ds_config["train_batch_size"], 2)
        self.assertEqual(ds_config["offload_optimizer_device"], "none")
        self.assertNotIn("offload_optimizer", ds_config["zero_optimization"])

    def test_checkpoint_resume_respects_auto_resume_flag(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {"find_latest_checkpoint", "resolve_resume_checkpoint"}
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        helper = next((node for node in helpers if node.name == "resolve_resume_checkpoint"), None)
        self.assertIsNotNone(helper)

        namespace = {"Path": Path}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "train_rg_flux_sr.py", "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "exp"
            latest = output_dir / "checkpoints" / "checkpoint-00000003"
            latest.mkdir(parents=True)
            manual = root / "manual-checkpoint"
            manual.mkdir()

            self.assertIsNone(namespace["resolve_resume_checkpoint"](output_dir, None, auto_resume=False))
            self.assertEqual(namespace["resolve_resume_checkpoint"](output_dir, None, auto_resume=True), latest)
            self.assertEqual(namespace["resolve_resume_checkpoint"](output_dir, str(manual), auto_resume=False), manual)

    def test_experiment_name_adds_datetime_run_id_and_avoids_collisions(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {"cfg", "cfg_bool", "make_experiment_name", "format_run_id", "resolve_experiment_name"}
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        namespace = {"datetime": datetime, "Path": Path}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "train_rg_flux_sr.py", "exec"), namespace)
        config = {
            "model": {"flux_backend": "flux2_klein"},
            "condition": {"lr_cond_mode": "flux2_image_concat"},
            "data": {"crop_size": 512},
            "training": {"stage": "0B", "suffix": "_stage0b512"},
        }
        now = datetime.datetime(2026, 6, 28, 10, 30)

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            exp_name, run_id = namespace["resolve_experiment_name"](config, output_root=output_root, now=now)
            self.assertTrue(exp_name.endswith("_26062810"))
            self.assertEqual(run_id, "26062810")
            (output_root / exp_name).mkdir()

            collided_name, collided_run_id = namespace["resolve_experiment_name"](
                config,
                output_root=output_root,
                now=now,
            )
            self.assertTrue(collided_name.endswith("_26062810_r02"))
            self.assertEqual(collided_run_id, "26062810")

        explicit = copy.deepcopy(config)
        explicit["training"]["exp_name"] = "manual_experiment"
        self.assertEqual(namespace["resolve_experiment_name"](explicit, now=now), ("manual_experiment", None))

        fixed = copy.deepcopy(config)
        fixed["training"]["run_id"] = 26070109
        self.assertTrue(namespace["resolve_experiment_name"](fixed, now=now)[0].endswith("_26070109"))

        disabled = copy.deepcopy(config)
        disabled["training"]["add_datetime_suffix"] = False
        disabled_name, disabled_run_id = namespace["resolve_experiment_name"](disabled, now=now)
        self.assertFalse(disabled_name.endswith("_26062810"))
        self.assertIsNone(disabled_run_id)

    def test_cfg_bool_parses_string_false_for_auto_resume(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {"cfg", "cfg_bool"}
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        namespace = {}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "train_rg_flux_sr.py", "exec"), namespace)

        config = {"training": {"auto_resume": "false"}}
        self.assertFalse(namespace["cfg_bool"](config, "training.auto_resume", True))
        config["training"]["auto_resume"] = "true"
        self.assertTrue(namespace["cfg_bool"](config, "training.auto_resume", False))

    def test_gradient_accumulation_plugin_uses_sync_each_batch(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "create_gradient_accumulation_plugin"
            ),
            None,
        )
        self.assertIsNotNone(helper)

        class FakeGradientAccumulationPlugin:
            def __init__(self, num_steps, sync_each_batch=False):
                self.num_steps = num_steps
                self.sync_each_batch = sync_each_batch

        namespace = {
            "GradientAccumulationPlugin": FakeGradientAccumulationPlugin,
            "inspect": inspect,
        }
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "train_rg_flux_sr.py", "exec"), namespace)

        plugin, supports_sync_each_batch = namespace["create_gradient_accumulation_plugin"](8)

        self.assertTrue(supports_sync_each_batch)
        self.assertEqual(plugin.num_steps, 8)
        self.assertTrue(plugin.sync_each_batch)

    def test_accelerator_receives_gradient_accumulation_plugin(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        main_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        plugin_create_line = None
        accelerator_line = None
        accelerator_has_plugin_kwarg = False
        for node in ast.walk(main_func):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "create_gradient_accumulation_plugin":
                plugin_create_line = node.lineno
            if isinstance(func, ast.Name) and func.id == "Accelerator":
                accelerator_line = node.lineno
                accelerator_has_plugin_kwarg = any(
                    keyword.arg == "gradient_accumulation_plugin" for keyword in node.keywords
                )

        self.assertIsNotNone(plugin_create_line)
        self.assertIsNotNone(accelerator_line)
        self.assertLess(plugin_create_line, accelerator_line)
        self.assertTrue(accelerator_has_plugin_kwarg)

    def test_zero3_sets_runtime_disable_gradient_checkpointing(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        main_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        sets_zero_stage = False
        sets_disable_checkpointing = False
        for node in ast.walk(main_func):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                slice_node = target.slice
                key = slice_node.value if isinstance(slice_node, ast.Constant) else None
                if key == "deepspeed_zero_stage":
                    sets_zero_stage = True
                if key == "disable_transformer_gradient_checkpointing":
                    sets_disable_checkpointing = True

        self.assertTrue(sets_zero_stage)
        self.assertTrue(sets_disable_checkpointing)

    def test_flux_artist_disables_checkpointing_when_runtime_requests_it(self):
        source = Path("models/flux_sr_artist.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        flux_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FluxSRArtist"
        )
        train_strategy = next(
            node for node in flux_class.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_train_strategy"
        )

        has_disable_call = False
        has_enable_call = False
        for node in ast.walk(train_strategy):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr == "disable_gradient_checkpointing":
                has_disable_call = True
            if func.attr == "enable_gradient_checkpointing":
                has_enable_call = True

        self.assertTrue(has_disable_call)
        self.assertTrue(has_enable_call)
        self.assertIn("disable_transformer_gradient_checkpointing", source)

    def test_default_config_uses_low_memory_frozen_encoder_devices(self):
        config = yaml.safe_load(Path("configs/train_rg_flux_sr_ms.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["model"].get("flux_backend", "flux1"), "flux1")
        self.assertEqual(config["model"]["text_encoder_device"], "cpu")
        self.assertEqual(config["model"]["vae_device"], "cpu")
        self.assertEqual(config["model"]["vae_dtype"], "fp32")
        self.assertLessEqual(config["model"]["max_prompt_sequence_length"], 128)

    def test_rg_flux_artist_factory_defaults_to_flux1_and_supports_flux2_klein(self):
        source = Path("models/rg_flux_artist_factory.py").read_text(encoding="utf-8")

        self.assertIn('model_config.get("flux_backend", "flux1")', source)
        self.assertIn("FluxSRArtist(config)", source)
        self.assertIn("Flux2KleinSRArtist", source)
        self.assertIn("flux2_klein", source)

    def test_train_and_inference_use_backend_factory(self):
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        inference_source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn("from models.rg_flux_artist_factory import build_rg_flux_artist", train_source)
        self.assertIn("artist = build_rg_flux_artist(config)", train_source)
        self.assertIn("from models.rg_flux_artist_factory import build_rg_flux_artist", inference_source)
        self.assertIn("artist = build_rg_flux_artist(config).to(device=device)", inference_source)

    def test_inference_jsonl_conditions_load_cleaned_profile_and_result(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {
            "_normalize_lookup_path",
            "path_lookup_aliases",
            "extend_lookup_aliases",
            "image_lookup_aliases",
            "load_jsonl_conditions",
            "condition_for_image",
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        namespace = {"json": json, "Path": Path}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "inference_rg_flux_sr.py", "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lq_path = root / "images" / "lq.png"
            hq_path = root / "images" / "hq.png"
            lq_path.parent.mkdir()
            jsonl_path = root / "valid.cleaned.jsonl"
            record = {
                "lq_path": str(lq_path),
                "hq_path": str(hq_path),
                "unipercept_raw": {
                    "profile": {
                        "iqa": {"overall_quality": "cleaned iqa quality"},
                        "suggestion": "cleaned suggestion",
                        "iaa": {"comprehensive": "cleaned iaa"},
                    }
                },
                "result": {"degradation_vector": {"blur": 0.7}, "score": 38},
            }
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            index = namespace["load_jsonl_conditions"](jsonl_path)
            by_full_path = namespace["condition_for_image"](index, lq_path)
            by_basename = namespace["condition_for_image"](index, Path("other-root") / lq_path.name)

        self.assertEqual(by_full_path["profile"]["iqa"]["overall_quality"], "cleaned iqa quality")
        self.assertEqual(by_full_path["result"]["degradation_vector"]["blur"], 0.7)
        self.assertEqual(by_basename["profile"]["suggestion"], "cleaned suggestion")

    def test_inference_jsonl_conditions_match_suffix_and_dataset_aliases(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {
            "_normalize_lookup_path",
            "path_lookup_aliases",
            "extend_lookup_aliases",
            "image_lookup_aliases",
            "load_jsonl_conditions",
            "condition_for_image",
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        namespace = {"json": json, "Path": Path}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "inference_rg_flux_sr.py", "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_path = root / "valid.cleaned.jsonl"
            profile_a = {"iqa": {"overall_quality": "real lq"}, "suggestion": "a"}
            profile_b = {"iqa": {"overall_quality": "real lr"}, "suggestion": "b"}
            records = [
                {
                    "dataset_name": "RealLQ250",
                    "lq_path": "/old/root/datasets/RealLQ250/lq/001.png",
                    "hq_path": "/old/root/datasets/RealLQ250/lq/001.png",
                    "unipercept_raw": {"profile": profile_a},
                },
                {
                    "dataset_name": "RealLR200",
                    "lq_path": "RealLR200/001.png",
                    "hq_path": "RealLR200/001.png",
                    "unipercept_raw": {"profile": profile_b},
                },
            ]
            jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            index = namespace["load_jsonl_conditions"](jsonl_path)
            suffix_match = namespace["condition_for_image"](
                index,
                Path("/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq/001.png"),
                dataset_name="RealLQ250",
            )
            dataset_match = namespace["condition_for_image"](
                index,
                Path("/root/autodl-tmp/datasets/omgsr_eval/RealLR200-xxx/RealLR200/001.png"),
                dataset_name="RealLR200",
            )

        self.assertEqual(suffix_match["profile"]["iqa"]["overall_quality"], "real lq")
        self.assertEqual(dataset_match["profile"]["iqa"]["overall_quality"], "real lr")

    def test_inference_jsonl_conditions_keep_records_without_profile_for_failure_logging(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {
            "_normalize_lookup_path",
            "path_lookup_aliases",
            "extend_lookup_aliases",
            "image_lookup_aliases",
            "load_jsonl_conditions",
            "condition_for_image",
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        namespace = {"json": json, "Path": Path}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "inference_rg_flux_sr.py", "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lq_path = root / "lq.png"
            jsonl_path = root / "valid.cleaned.jsonl"
            record = {
                "lq_path": str(lq_path),
                "hq_path": str(root / "hq.png"),
                "result": {"degradation_vector": {"noise": 0.4}},
            }
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            index = namespace["load_jsonl_conditions"](jsonl_path)
            condition = namespace["condition_for_image"](index, lq_path)

        self.assertIsNotNone(condition)
        self.assertIsNone(condition["profile"])
        self.assertEqual(condition["result"]["degradation_vector"]["noise"], 0.4)
        self.assertEqual(condition["record"]["lq_path"], str(lq_path))

    def test_inference_failure_log_records_missing_jsonl_match_and_missing_profile(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "append_inference_failure"
            ),
            None,
        )
        self.assertIsNotNone(helper)

        namespace = {"json": json, "Path": Path}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "inference_rg_flux_sr.py", "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "out" / "inference_failures.jsonl"
            namespace["append_inference_failure"](
                log_path,
                image_path=root / "missing.png",
                reason="missing_jsonl_match",
                condition=None,
            )
            namespace["append_inference_failure"](
                log_path,
                image_path=root / "no_profile.png",
                reason="missing_unipercept_raw.profile",
                condition={"record": {"lq_path": "lq.png", "hq_path": "hq.png"}},
            )

            failures = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(failures[0]["reason"], "missing_jsonl_match")
        self.assertEqual(failures[0]["image_path"], str(root / "missing.png"))
        self.assertEqual(failures[1]["reason"], "missing_unipercept_raw.profile")
        self.assertEqual(failures[1]["lq_path"], "lq.png")
        self.assertEqual(failures[1]["hq_path"], "hq.png")

    def test_inference_parser_supports_single_input_and_dataset_dirs(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {"build_arg_parser"}
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "inference_rg_flux_sr.py", "exec"), namespace)
        parser = namespace["build_arg_parser"]()

        single = parser.parse_args([
            "--input",
            "/data/RealLQ250/lq",
            "--output_dir",
            "eval/inference/RealLQ250",
            "--checkpoint",
            "ckpt",
        ])
        self.assertEqual(single.input, "/data/RealLQ250/lq")
        self.assertIsNone(single.dataset_dirs)

        multi = parser.parse_args([
            "--dataset_dirs",
            "realLQ250=/data/RealLQ250/lq",
            "realLR200=/data/RealLR200/lq",
            "--output_dir",
            "eval/inference",
            "--checkpoint",
            "ckpt",
        ])
        self.assertIsNone(multi.input)
        self.assertEqual(
            multi.dataset_dirs,
            ["realLQ250=/data/RealLQ250/lq", "realLR200=/data/RealLR200/lq"],
        )

        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--input",
                "/data/RealLQ250/lq",
                "--dataset_dirs",
                "realLR200=/data/RealLR200/lq",
                "--output_dir",
                "eval/inference",
                "--checkpoint",
                "ckpt",
            ])

    def test_inference_dataset_dir_helpers_resolve_output_subdirs(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {"parse_dataset_dirs", "resolve_inference_datasets"}
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        namespace = {"Path": Path}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "inference_rg_flux_sr.py", "exec"), namespace)

        parsed = namespace["parse_dataset_dirs"]([
            "realLQ250=/data/RealLQ250/lq",
            "realLR200=/data/RealLR200/lq",
        ])
        self.assertEqual(parsed, [
            ("realLQ250", Path("/data/RealLQ250/lq")),
            ("realLR200", Path("/data/RealLR200/lq")),
        ])

        args = argparse.Namespace(
            input=None,
            dataset_dirs=["realLQ250=/data/RealLQ250/lq", "realLR200=/data/RealLR200/lq"],
            output_dir="eval/inference",
        )
        datasets = namespace["resolve_inference_datasets"](args)
        self.assertEqual(datasets, [
            ("realLQ250", Path("/data/RealLQ250/lq"), Path("eval/inference/realLQ250")),
            ("realLR200", Path("/data/RealLR200/lq"), Path("eval/inference/realLR200")),
        ])

        single_args = argparse.Namespace(
            input="/data/RealLQ250/lq",
            dataset_dirs=None,
            output_dir="eval/inference/RealLQ250",
        )
        self.assertEqual(
            namespace["resolve_inference_datasets"](single_args),
            [("default", Path("/data/RealLQ250/lq"), Path("eval/inference/RealLQ250"))],
        )

        with self.assertRaises(ValueError):
            namespace["parse_dataset_dirs"](["missing_equals"])

    def test_inference_run_dir_resolves_checkpoint_step_and_manifest(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {
            "default_run_inference_dir",
            "find_latest_run_checkpoint",
            "format_checkpoint_step",
            "infer_checkpoint_step",
            "resolve_inference_run",
            "validate_checkpoint_adapter",
            "write_inference_manifest",
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        self.assertEqual({node.name for node in helpers}, helper_names)

        namespace = {"json": json, "Path": Path}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "inference_rg_flux_sr.py", "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "exp" / "rg_flux2_klein_26062810"
            ckpt_10000 = run_dir / "checkpoints" / "checkpoint-00010000" / "rg_flux_adapters"
            ckpt_32000 = run_dir / "checkpoints" / "checkpoint-00032000" / "rg_flux_adapters"
            ckpt_10000.mkdir(parents=True)
            ckpt_32000.mkdir(parents=True)

            args = argparse.Namespace(
                run_dir=str(run_dir),
                checkpoint_step="32000",
                output_root=str(root / "eval" / "inference"),
                checkpoint=None,
                output_dir=None,
            )
            resolved = namespace["resolve_inference_run"](args)
            self.assertEqual(resolved["checkpoint"], ckpt_32000)
            self.assertEqual(resolved["checkpoint_step"], "checkpoint-00032000")
            self.assertEqual(
                resolved["output_dir"],
                root / "eval" / "inference" / run_dir.name / "checkpoint-00032000",
            )

            latest_args = argparse.Namespace(
                run_dir=str(run_dir),
                checkpoint_step="latest",
                output_root=str(root / "eval" / "inference"),
                checkpoint=None,
                output_dir=None,
            )
            latest = namespace["resolve_inference_run"](latest_args)
            self.assertEqual(latest["checkpoint"], ckpt_32000)
            self.assertEqual(latest["checkpoint_step"], "checkpoint-00032000")

            default_args = argparse.Namespace(
                run_dir=str(run_dir),
                checkpoint_step="32000",
                output_root=None,
                checkpoint=None,
                output_dir=None,
            )
            default_resolved = namespace["resolve_inference_run"](default_args)
            self.assertEqual(
                default_resolved["output_dir"],
                run_dir / "inference" / "checkpoint-00032000",
            )

            exact_output_args = argparse.Namespace(
                run_dir=str(run_dir),
                checkpoint_step="32000",
                output_root=None,
                checkpoint=None,
                output_dir=str(root / "custom_output"),
            )
            exact_output = namespace["resolve_inference_run"](exact_output_args)
            self.assertEqual(exact_output["output_dir"], root / "custom_output")

            with self.assertRaises(ValueError):
                namespace["resolve_inference_run"](
                    argparse.Namespace(
                        run_dir=str(run_dir),
                        checkpoint_step="32000",
                        output_root=str(root / "out"),
                        checkpoint=None,
                        output_dir=str(root / "custom_output"),
                    )
                )

            with self.assertRaises(FileNotFoundError):
                namespace["resolve_inference_run"](
                    argparse.Namespace(
                        run_dir=str(run_dir),
                        checkpoint_step="99999",
                        output_root=None,
                        checkpoint=None,
                        output_dir=None,
                    )
                )

            legacy_args = argparse.Namespace(
                run_dir=None,
                checkpoint=str(ckpt_32000),
                output_dir=str(root / "legacy_output"),
                checkpoint_step=None,
                output_root=None,
            )
            legacy = namespace["resolve_inference_run"](legacy_args)
            self.assertEqual(legacy["checkpoint"], ckpt_32000)
            self.assertEqual(legacy["checkpoint_step"], "checkpoint-00032000")
            self.assertEqual(legacy["output_dir"], root / "legacy_output")

            manifest_path = resolved["output_dir"] / "inference_manifest.json"
            namespace["write_inference_manifest"](
                manifest_path=manifest_path,
                run_dir=run_dir,
                checkpoint_step=resolved["checkpoint_step"],
                checkpoint_path=resolved["checkpoint"],
                output_dir=resolved["output_dir"],
                datasets=[
                    ("realLQ250", Path("/data/RealLQ250/lq"), resolved["output_dir"] / "realLQ250"),
                    ("realLR200", Path("/data/RealLR200/lq"), resolved["output_dir"] / "realLR200"),
                ],
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_dir"], str(run_dir))
            self.assertEqual(payload["checkpoint_step"], "checkpoint-00032000")
            self.assertEqual(payload["checkpoint_path"], str(ckpt_32000))
            self.assertEqual(payload["output_dir"], str(resolved["output_dir"]))
            self.assertEqual(payload["datasets"][1]["name"], "realLR200")
            self.assertEqual(payload["datasets"][1]["output_dir"], str(resolved["output_dir"] / "realLR200"))

    def test_inference_loads_model_once_and_writes_per_dataset_failures(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertEqual(source.count("build_rg_flux_artist(config).to(device=device)"), 1)
        self.assertEqual(source.count('artist.load_trainable(resolved_run["checkpoint"], is_trainable=False)'), 1)
        self.assertIn("def run_inference_dataset(", source)
        self.assertIn('desc=f"RG-FLUX-SR inference [{dataset_name}]"', source)
        self.assertIn('failure_log_path = output_dir / "inference_failures.jsonl"', source)
        self.assertIn("output_dir / dataset_name", source)
        self.assertIn("write_inference_manifest(", source)
        helper_start = source.index("def run_inference_dataset(")
        helper_end = source.index("\ndef main(args):", helper_start)
        helper_source = source[helper_start:helper_end]
        self.assertNotIn("build_rg_flux_artist", helper_source)
        self.assertNotIn("load_trainable", helper_source)

    def test_flux2_klein_smoke_config_is_isolated(self):
        main_config = yaml.safe_load(Path("configs/train_rg_flux_sr_ms.yaml").read_text(encoding="utf-8"))
        config_path = Path("configs/train_rg_flux2_klein_sr_smoke_256.yaml")
        self.assertTrue(config_path.exists())
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(main_config["model"].get("flux_backend", "flux1"), "flux1")
        self.assertEqual(config["model"]["flux_backend"], "flux2_klein")
        self.assertIn("FLUX.2-klein", config["model"]["flux_model_path"])
        self.assertEqual(config["data"]["crop_size"], 256)
        self.assertLessEqual(config["condition"]["lr_token_count"], 8)
        self.assertEqual(config["condition"]["deg_token_count"], 0)
        self.assertEqual(config["training"]["grad_accum_steps"], 1)
        self.assertEqual(config["training"]["deepspeed_optimizer_offload_device"], "none")
        self.assertFalse(config["training"]["auto_resume"])
        self.assertEqual(config["training"]["suffix"], "_flux2_klein_smoke256_v2")

    def test_flux2_artist_has_separate_diffusers_components_and_checkpoint_names(self):
        source = Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8")

        self.assertIn("class Flux2KleinSRArtist", source)
        self.assertIn("AutoencoderKLFlux2", source)
        self.assertIn("Flux2KleinPipeline", source)
        self.assertIn("Flux2Transformer2DModel", source)
        self.assertIn("flux2_klein_lora_state.pt", source)
        self.assertIn("rg_flux_checkpoint_meta.json", source)
        self.assertNotIn("pooled_projections", source)
        self.assertIn("_gathered_named_parameter_state", source)
        self.assertIn("_load_state_dict_with_shape_check", source)

    def test_flux2_inference_aligns_dtype_after_loading_trainable_adapters(self):
        artist_source = Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8")
        inference_source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn("def align_inference_dtype(", artist_source)
        self.assertIn("self.transformer.to(dtype=dtype)", artist_source)
        self.assertIn("self.moe_router.to(device=device, dtype=torch.float32)", artist_source)
        self.assertIn("def resolve_inference_dtype(", inference_source)
        self.assertIn('cfg(config, "model.dtype", None)', inference_source)
        self.assertIn("Warning: --dtype", inference_source)
        self.assertIn('artist.load_trainable(resolved_run["checkpoint"], is_trainable=False)', inference_source)
        self.assertIn("artist.align_inference_dtype(dtype=dtype)", inference_source)

    @unittest.skipIf(torch is None, "torch is not installed in this environment")
    def test_moe_lora_layer_accepts_bf16_after_dtype_alignment(self):
        from models.lora_moe import SharedRoutedMoELoRALinear

        base = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        layer = SharedRoutedMoELoRALinear(base, rank=2, alpha=2, num_routed_experts=2)
        self.assertEqual(layer.shared_lora_A.dtype, torch.float32)

        layer.to(dtype=torch.bfloat16)
        x = torch.randn(1, 3, 4, dtype=torch.bfloat16)
        y = layer(x)

        self.assertEqual(layer.shared_lora_A.dtype, torch.bfloat16)
        self.assertEqual(y.dtype, torch.bfloat16)

    def test_flux2_native_image_concat_uses_pretrained_condition_layout(self):
        artist_source = Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8")
        encoder_source = Path("models/lr_condition_encoder.py").read_text(encoding="utf-8")
        inference_source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn('"flux2_image_concat"', encoder_source)
        self.assertIn('if mode == "flux2_image_concat":', encoder_source)
        self.assertIn("return z_lr.flatten(2).transpose(1, 2)", encoder_source)
        self.assertIn("def _condition_image_ids(", artist_source)
        self.assertIn("time_id=10", artist_source)
        self.assertIn("hidden_states = torch.cat([hidden_states, lr_tokens], dim=1)", artist_source)
        self.assertIn("img_ids = torch.cat([img_ids, lr_img_ids], dim=1)", artist_source)
        self.assertIn("packed_pred = packed_pred[:, :target_token_count]", artist_source)
        self.assertIn('"flux2_image_concat"', inference_source)

    def test_flux2_native_image_concat_uses_deterministic_lr_latents(self):
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        inference_source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        init_source = Path("tools/init_flux2_lora_moe.py").read_text(encoding="utf-8")

        expected = 'sample=lr_cond_mode != "flux2_image_concat"'
        self.assertGreaterEqual(train_source.count(expected), 2)
        self.assertIn(expected, inference_source)
        self.assertIn(expected, init_source)

    def test_flux2_stage0b_adds_trainable_decode_without_changing_inference_decode(self):
        artist_source = Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8")
        tree = ast.parse(artist_source)
        flux_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Flux2KleinSRArtist"
        )
        decode_for_loss = next(
            (node for node in flux_class.body if isinstance(node, ast.FunctionDef) and node.name == "decode_latents_for_loss"),
            None,
        )
        self.assertIsNotNone(decode_for_loss)
        decorator_names = [
            getattr(decorator, "attr", getattr(decorator, "id", ""))
            for decorator in decode_for_loss.decorator_list
        ]
        self.assertNotIn("no_grad", decorator_names)
        self.assertIn("self._denormalize_latents(latents)", artist_source)
        self.assertIn("_unpatchify_latents(latents, self.vae_latent_channels)", artist_source)
        self.assertIn("@torch.no_grad()\n    def decode_latents(self, latents):", artist_source)

    def test_stage0b_training_loss_helpers_and_inference_stays_clean(self):
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        inference_source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn("def charbonnier_loss(", train_source)
        self.assertIn("def warmup_weight(", train_source)
        self.assertIn("def should_compute_every(", train_source)
        self.assertIn("def compute_stage0b_supervised_losses(", train_source)
        self.assertIn('cfg(config, "loss.charb_weight", 0.0)', train_source)
        self.assertIn('cfg(config, "loss.down_weight", 0.0)', train_source)
        self.assertIn('cfg(config, "loss.lpips_weight", 0.0)', train_source)
        self.assertIn("z0_pred = z_t - sigma_view * v_pred", train_source)
        self.assertIn("decode_latents_for_loss(z0_for_image)", train_source)
        self.assertIn("loss_lpips_weight", train_source)
        self.assertIn("loss_total", train_source)
        self.assertNotIn("decode_latents_for_loss", inference_source)
        self.assertNotIn("compute_stage0b_supervised_losses", inference_source)

    def test_loss_history_recorder_writes_jsonl_csv_and_summary(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        wanted = {"normalize_loss_record_formats", "LossHistoryRecorder"}
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in wanted
            )
        ]
        self.assertEqual({node.name for node in nodes}, wanted)

        namespace = {
            "csv": csv,
            "datetime": datetime,
            "json": json,
            "Path": Path,
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "train_rg_flux_sr.py", "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            recorder = namespace["LossHistoryRecorder"](tmp, formats=["jsonl", "csv"])
            recorder.append(
                1,
                {
                    "loss_total": 1.5,
                    "loss_fm": 1.2,
                    "loss_charb": 0.3,
                    "router/entropy": 0.8,
                },
            )
            recorder.append(
                2,
                {
                    "loss_total": 1.0,
                    "loss_fm": 0.9,
                    "loss_charb": 0.2,
                    "router/entropy": 0.7,
                },
            )

            root = Path(tmp)
            jsonl_rows = [
                json.loads(line)
                for line in (root / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["global_step"] for row in jsonl_rows], [1, 2])
            self.assertEqual(jsonl_rows[0]["loss_total"], 1.5)
            self.assertIn("time", jsonl_rows[0])

            with (root / "loss_history.csv").open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual([row["global_step"] for row in csv_rows], ["1", "2"])
            self.assertEqual(csv_rows[1]["loss_fm"], "0.9")

            summary = json.loads((root / "loss_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["last_step"], 2)
            self.assertEqual(summary["last"]["loss_total"], 1.0)
            self.assertEqual(summary["min_loss_total"]["global_step"], 2)
            self.assertEqual(summary["min_loss_fm"]["global_step"], 2)

            plot_path = recorder.write_plot(step=2)
            self.assertEqual(plot_path, root / "loss_curves.png")
            self.assertTrue(plot_path.exists())
            self.assertEqual(plot_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue((root / "loss_curves_step-00000002.png").exists())

    def test_training_loop_records_loss_history_on_optimizer_steps(self):
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn("loss_history.jsonl", train_source)
        self.assertIn("loss_history.csv", train_source)
        self.assertIn("loss_summary.json", train_source)
        self.assertIn("loss_curves.png", train_source)
        self.assertIn('cfg(config, "training.loss_record_every", 1)', train_source)
        self.assertIn('cfg(config, "training.loss_plot_every", save_every)', train_source)
        self.assertIn('cfg(config, "training.loss_record_formats", ["jsonl", "csv"])', train_source)
        self.assertIn("loss_recorder.append(global_step, step_logs)", train_source)
        self.assertIn("loss_recorder.write_plot(step=global_step)", train_source)
        self.assertIn("step_logs = {", train_source)
        self.assertIn('"loss_total": loss.detach().item()', train_source)
        self.assertIn('"loss_lpips_weight": float(loss_lpips_weight)', train_source)

    def test_stage0b_image_loss_can_crop_decode_to_reduce_memory(self):
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        single_config = yaml.safe_load(Path("configs/train_rg_flux2_klein_sr_stage0b_512.yaml").read_text(encoding="utf-8"))
        moe_config = yaml.safe_load(Path("configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml").read_text(encoding="utf-8"))

        self.assertIn("def crop_image_loss_inputs(", train_source)
        self.assertIn('cfg(config, "loss.image_loss_crop_size", 0)', train_source)
        self.assertIn("z0_for_image, hq_for_image, lq_up_for_image", train_source)
        self.assertIn("return z0_crop, hq_crop, lq_up_crop, True", train_source)
        self.assertIn("lr_ref_batch = {} if crop_lq_ref else (batch or {})", train_source)
        for config in (single_config, moe_config):
            self.assertEqual(config["model"]["vae_dtype"], "bf16")
            self.assertEqual(config["loss"]["image_loss_crop_size"], 256)

    @unittest.skipIf(torch is None, "torch is not installed in this environment")
    def test_stage0b_loss_helpers_values_and_backward(self):
        from train_rg_flux_sr import charbonnier_loss, should_compute_every, warmup_weight

        x = torch.tensor([0.0, 1.0], requires_grad=True)
        y = torch.tensor([0.0, 0.0])
        loss = charbonnier_loss(x, y, eps=1e-3)
        self.assertGreater(float(loss.item()), 0.5)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

        self.assertEqual(warmup_weight(10, 20, 40, 0.25), 0.0)
        self.assertAlmostEqual(warmup_weight(30, 20, 40, 0.25), 0.125)
        self.assertEqual(warmup_weight(50, 20, 40, 0.25), 0.25)
        self.assertEqual(warmup_weight(10, 20, 20, 0.25), 0.0)
        self.assertEqual(warmup_weight(20, 20, 20, 0.25), 0.25)
        self.assertTrue(should_compute_every(7, 1))
        self.assertTrue(should_compute_every(8, 4))
        self.assertFalse(should_compute_every(9, 4))

    @unittest.skipIf(torch is None, "torch is not installed in this environment")
    def test_stage0b_zero_weights_skip_decode_and_charb_backward_uses_decode(self):
        from train_rg_flux_sr import compute_stage0b_supervised_losses

        class DummyArtist:
            def __init__(self):
                self.decode_calls = 0

            def decode_latents_for_loss(self, latents):
                self.decode_calls += 1
                return latents

        z_t = torch.randn(1, 3, 4, 4)
        v_pred = torch.randn(1, 3, 4, 4, requires_grad=True)
        z_hr = torch.randn(1, 3, 4, 4)
        sigma = torch.full((1,), 0.5)
        hq = torch.randn(1, 3, 4, 4)
        lq_up = torch.randn(1, 3, 4, 4)
        loss_fm = torch.nn.functional.mse_loss(v_pred.float(), torch.zeros_like(v_pred).float())

        zero_artist = DummyArtist()
        zero_losses = compute_stage0b_supervised_losses(
            artist=zero_artist,
            config={"loss": {}},
            global_step=1,
            loss_fm=loss_fm,
            z_t=z_t,
            v_pred=v_pred,
            sigma=sigma,
            z_hr=z_hr,
            hq=hq,
            lq_up=lq_up,
            batch={},
        )
        self.assertEqual(zero_artist.decode_calls, 0)
        self.assertEqual(float(zero_losses["loss_latent"].item()), 0.0)
        self.assertEqual(float(zero_losses["loss_charb"].item()), 0.0)
        self.assertEqual(float(zero_losses["loss_down"].item()), 0.0)
        self.assertEqual(float(zero_losses["loss_lpips"].item()), 0.0)

        charb_artist = DummyArtist()
        charb_losses = compute_stage0b_supervised_losses(
            artist=charb_artist,
            config={"loss": {"charb_weight": 1.0}},
            global_step=1,
            loss_fm=loss_fm,
            z_t=z_t,
            v_pred=v_pred,
            sigma=sigma,
            z_hr=z_hr,
            hq=hq,
            lq_up=lq_up,
            batch={},
        )
        self.assertEqual(charb_artist.decode_calls, 1)
        charb_losses["loss_charb"].backward()
        self.assertIsNotNone(v_pred.grad)

    def test_stage0b_configs_cover_single_and_moe_backends(self):
        single_path = Path("configs/train_rg_flux2_klein_sr_stage0b_512.yaml")
        moe_path = Path("configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml")
        self.assertTrue(single_path.exists())
        self.assertTrue(moe_path.exists())

        single_config = yaml.safe_load(single_path.read_text(encoding="utf-8"))
        moe_config = yaml.safe_load(moe_path.read_text(encoding="utf-8"))
        for config in (single_config, moe_config):
            self.assertEqual(config["model"]["flux_backend"], "flux2_klein")
            self.assertEqual(config["condition"]["lr_cond_mode"], "flux2_image_concat")
            self.assertEqual(config["loss"]["fm_weight"], 1.0)
            self.assertEqual(config["loss"]["latent_weight"], 0.10)
            self.assertEqual(config["loss"]["charb_weight"], 1.0)
            self.assertEqual(config["loss"]["down_weight"], 0.50)
            self.assertEqual(config["loss"]["down_mode"], "area")
            self.assertEqual(config["loss"]["lpips_weight"], 0.25)
            self.assertEqual(config["loss"]["lpips_warmup_start"], 2000)
            self.assertEqual(config["loss"]["lpips_warmup_end"], 6000)
            self.assertEqual(config["loss"]["lpips_resize"], 256)
            self.assertEqual(config["loss"]["image_loss_crop_size"], 256)
            self.assertEqual(config["loss"]["image_loss_every"], 1)
            self.assertEqual(config["loss"]["lpips_every"], 1)
        self.assertEqual(single_config["loss"]["router_div_weight"], 0.0)
        self.assertEqual(single_config["loss"]["router_entropy_weight"], 0.0)
        self.assertEqual(single_config["loss"]["router_balance_weight"], 0.0)
        self.assertEqual(moe_config["loss"]["router_div_weight"], 1.0e-3)
        self.assertEqual(moe_config["loss"]["router_entropy_weight"], 1.0e-4)
        self.assertEqual(moe_config["loss"]["router_balance_weight"], 1.0e-3)
        self.assertEqual(moe_config["training"]["lr_router"], 1.0e-4)
        self.assertEqual(moe_config["model"]["lora_moe"]["prototype_scale"], 1.0)
        self.assertNotEqual(single_config["model"].get("lora_backend"), "moe")
        self.assertEqual(moe_config["model"]["lora_backend"], "moe")
        self.assertIn("lora_moe", moe_config["model"])

    def test_flux2_lora_moe_backend_is_optional_and_checkpoint_is_separate(self):
        source = Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8")
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")
        config = yaml.safe_load(Path("configs/train_rg_flux2_klein_sr_moe_smoke_256.yaml").read_text(encoding="utf-8"))

        self.assertIn('self.lora_backend = str(_cfg(config, "model.lora_backend", "peft")).lower()', source)
        self.assertIn('if self.lora_backend == "moe":', source)
        self.assertIn('if self.lora_backend != "moe" and not bool(_cfg(self.config, "training.freeze_flux_transformer", True)):', source)
        self.assertIn("SharedRoutedMoELoRALinear", source)
        self.assertIn("ProfileLatentRouter", source)
        self.assertIn("flux2_klein_lora_moe_state.pt", source)
        self.assertIn("set_moe_training_schedule", train_source)
        self.assertIn("moe_auxiliary_losses", train_source)
        self.assertIn('cfg(config, "training.lr_router", 1e-4)', train_source)
        self.assertIn("model.lora_moe.init_from_single_lora", train_source)
        self.assertIn("initialize_moe_from_single_lora(init_single_lora)", train_source)
        moe_source = Path("models/lora_moe.py").read_text(encoding="utf-8")
        self.assertNotIn("class RouterOutput", moe_source)
        self.assertIn("return logits, alpha, features", moe_source)
        self.assertIn("GatheredParameters", moe_source)
        self.assertIn("_maybe_gathered_parameters(parameters)", moe_source)
        self.assertEqual(config["model"]["lora_backend"], "moe")
        self.assertEqual(config["model"]["lora_moe"]["top_k"], 2)
        self.assertTrue(config["training"]["freeze_flux_transformer"])
        self.assertGreater(config["loss"]["router_div_weight"], 0)

    def test_flux2_lora_moe_uses_calibrated_router_defaults(self):
        config_paths = (
            "configs/train_rg_flux2_klein_sr_moe_smoke_256.yaml",
            "configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml",
            "configs/train_rg_flux2_klein_sr_moe_stage0b_512_prompt_curriculum_precropped.yaml",
        )
        for config_path in config_paths:
            config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            self.assertEqual(config["model"]["lora_moe"]["prototype_scale"], 1.0)
            self.assertEqual(config["training"]["lr_router"], 1.0e-4)
            self.assertEqual(config["loss"]["router_div_weight"], 1.0e-3)
            self.assertEqual(config["loss"]["router_entropy_weight"], 1.0e-4)
            self.assertEqual(config["loss"]["router_balance_weight"], 1.0e-3)

    def test_flux2_lora_moe_init_tool_loads_single_lora_and_initializes_prototypes(self):
        source = Path("tools/init_flux2_lora_moe.py").read_text(encoding="utf-8")

        self.assertIn("initialize_moe_from_single_lora", source)
        self.assertIn("prototype_num_samples", source)
        self.assertIn("kmeans", source)
        self.assertIn("compute_router_features", source)
        self.assertIn("flux2_klein_lora_moe_state.pt", Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8"))

    def test_adapter_checkpoint_loader_reports_zero3_partitioned_tensors(self):
        source = Path("models/flux_sr_artist.py").read_text(encoding="utf-8")

        self.assertIn("def _load_state_dict_with_shape_check", source)
        self.assertIn("ZeRO-3 partitioned tensor", source)
        self.assertIn("tensor.numel() == 0", source)
        self.assertIn("module_parameters = dict(module.named_parameters())", source)
        self.assertIn('hasattr(expected_param, "ds_id")', source)
        self.assertIn("with _maybe_gathered_parameters(load_parameters):", source)
        self.assertIn("module.load_state_dict(state, strict=False)", source)

    def test_adapter_checkpoint_loader_requires_existing_checkpoint_dir(self):
        for source_path in ("models/flux_sr_artist.py", "models/flux2_klein_sr_artist.py"):
            source = Path(source_path).read_text(encoding="utf-8")

            self.assertIn("if not checkpoint_dir.exists():", source)
            self.assertIn("raise FileNotFoundError", source)
            self.assertIn("Adapter checkpoint directory does not exist", source)

    def test_flux1_and_flux2_checkpoint_saves_gather_zero3_trainable_parameters(self):
        flux1_source = Path("models/flux_sr_artist.py").read_text(encoding="utf-8")
        flux2_source = Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8")
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn("GatheredParameters", flux1_source)
        self.assertIn("_gathered_named_parameter_state", flux1_source)
        self.assertIn("_adapter_parameter_name", flux1_source)
        self.assertIn("_lora_parameter_name", flux1_source)
        self.assertIn("self.transformer,", flux1_source)
        self.assertIn("self.transformer,", flux2_source)
        self.assertIn("_gathered_named_parameter_state(self, _adapter_parameter_name, collect_state=save_files)", flux1_source)
        self.assertIn("_gathered_named_parameter_state(self, _adapter_parameter_name, collect_state=save_files)", flux2_source)
        self.assertIn("collect_state=save_files", flux1_source)
        self.assertIn("collect_state=save_files", flux2_source)
        self.assertIn("save_trainable(checkpoint_dir / \"rg_flux_adapters\", save_files=accelerator.is_main_process)", train_source)
        self.assertNotIn("if not accelerator.is_main_process:\n        return\n    checkpoint_dir = Path(checkpoint_dir)", train_source)

    def test_training_state_resume_uses_non_weights_only_load(self):
        train_source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn("def load_training_state", train_source)
        self.assertIn('torch.load(state_path, map_location="cpu", weights_only=False)', train_source)
        self.assertIn("state = load_training_state(state_path)", train_source)
        self.assertIn("def training_state_rank_path", train_source)
        self.assertIn("training_state_rank-{rank:05d}.pt", train_source)
        self.assertIn("optimizer_state = {rank: optimizer_state}", train_source)
        self.assertIn("_optimizer_requires_rank_state_dict", train_source)
        self.assertIn("_is_rank_keyed_optimizer_state", train_source)
        self.assertIn("resume_training_state=resume_training_state", train_source)
        self.assertIn("training.resume_training_state", train_source)
        self.assertIn("was saved without rank-local ZeRO-3 optimizer state", train_source)

    def test_flux2_encode_prompt_filters_kwargs_by_pipeline_signature(self):
        source = Path("models/flux2_klein_sr_artist.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_supported_call_kwargs"
            ),
            None,
        )
        self.assertIsNotNone(helper)

        namespace = {"inspect": inspect}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "models/flux2_klein_sr_artist.py", "exec"), namespace)

        def encode_prompt(prompt, max_sequence_length=None):
            return prompt, max_sequence_length

        kwargs = namespace["_supported_call_kwargs"](
            encode_prompt,
            {
                "prompt": ["test"],
                "prompt_2": None,
                "max_sequence_length": 128,
                "num_images_per_prompt": 1,
            },
        )

        self.assertEqual(kwargs, {"prompt": ["test"], "max_sequence_length": 128})
        self.assertIn("_supported_call_kwargs(self.text_pipeline.encode_prompt", source)

    def test_flux1_checkpoint_metadata_rejects_flux2_checkpoint(self):
        source = Path("models/flux_sr_artist.py").read_text(encoding="utf-8")

        self.assertIn('"flux_backend": "flux1"', source)
        self.assertIn("rg_flux_checkpoint_meta.json", source)
        self.assertIn("flux2_klein_lora_state.pt", source)
        self.assertIn("model.flux_backend: flux2_klein", source)

    def test_smoke_256_config_keeps_main_config_unchanged(self):
        main_config = yaml.safe_load(Path("configs/train_rg_flux_sr_ms.yaml").read_text(encoding="utf-8"))
        smoke_path = Path("configs/train_rg_flux_sr_ms_smoke_256.yaml")
        self.assertTrue(smoke_path.exists())
        smoke_config = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))

        self.assertEqual(main_config["data"]["crop_size"], 512)
        self.assertEqual(smoke_config["data"]["crop_size"], 256)
        self.assertLessEqual(smoke_config["condition"]["lr_token_count"], 16)
        self.assertEqual(smoke_config["condition"]["deg_token_count"], 0)
        self.assertLessEqual(smoke_config["model"]["max_prompt_sequence_length"], 64)
        self.assertEqual(smoke_config["training"]["grad_accum_steps"], 1)
        self.assertEqual(smoke_config["training"]["save_every"], 1)
        self.assertEqual(smoke_config["training"]["suffix"], "_smoke256")
        self.assertEqual(smoke_config["condition"]["lr_cond_mode"], "latent_adapter")

    def test_train_logs_dry_run_token_shape_diagnostics(self):
        source = Path("train_rg_flux_sr.py").read_text(encoding="utf-8")

        self.assertIn("estimated latent size", source)
        self.assertIn("packed image token count", source)
        self.assertIn("condition.lr_token_count", source)
        self.assertIn("condition.deg_token_count", source)

    def test_prompt_builder_can_disable_suggestions(self):
        from models.prompt_builder import build_sr_prompt

        prompt = build_sr_prompt(
            {
                "iqa": {"overall_quality": "IQA evidence remains."},
                "suggestion": "recover fine textures",
                "iaa": {"comprehensive": "IAA evidence remains."},
            },
            use_prompt=True,
            use_suggestions=False,
        )

        self.assertNotIn("recover fine textures", prompt)
        self.assertIn("IQA evidence remains.", prompt)
        self.assertIn("IAA evidence remains.", prompt)
        self.assertIn("Super-resolve this low-quality image", prompt)

    def test_jsonl_dataset_uses_cleaned_profile_prompt_and_keeps_result_conditions(self):
        if torch is None:
            self.skipTest("torch is not installed in this environment")
        from dataloaders.rg_flux_jsonl_dataset import RGFluxSRJsonlDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hq_path, lq_path = self._make_pair(root)
            jsonl_path = root / "valid.jsonl"
            record = {
                "hq_path": str(hq_path),
                "lq_path": str(lq_path),
                "raw_degradation_params": {"blur": 999.0},
                "raw_qwen_response": "must be ignored",
                "unipercept_raw": {
                    "profile": {
                        "iqa": {
                            "distortion_location": "cleaned iqa location",
                            "overall_quality": "cleaned iqa quality",
                        },
                        "suggestion": "cleaned profile suggestion",
                        "iaa": {"comprehensive": "cleaned iaa comprehensive"},
                    }
                },
                "result": {
                    "reasoning": {"degradation_analysis": "offline result"},
                    "suggestions": ["enhance edge sharpness"],
                    "score": 47,
                    "degradation_vector": {
                        "blur": 0.1,
                        "noise": 0.2,
                        "jpeg": 0.3,
                    },
                },
            }
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = RGFluxSRJsonlDataset(
                jsonl_path=str(jsonl_path),
                crop_size=32,
                scale=4,
                mode="val",
                use_prompt=True,
                use_degradation_vector=True,
                pre_cropped=False,
            )
            sample = dataset[0]

        self.assertEqual(sample["hq"].shape, (3, 32, 32))
        self.assertEqual(sample["lq_up"].shape, (3, 32, 32))
        self.assertGreaterEqual(float(sample["hq"].min()), -1.0)
        self.assertLessEqual(float(sample["hq"].max()), 1.0)
        self.assertIn("cleaned iqa location", sample["prompt"])
        self.assertIn("cleaned profile suggestion", sample["prompt"])
        self.assertIn("cleaned iaa comprehensive", sample["prompt"])
        self.assertNotIn("offline result", sample["prompt"])
        self.assertEqual(sample["degradation_vector"].shape, (8,))
        self.assertAlmostEqual(float(sample["degradation_vector"][0]), 0.1)
        self.assertAlmostEqual(float(sample["degradation_vector"][3]), 0.0)
        self.assertEqual(float(sample["score"]), 47.0)
        self.assertEqual(sample["suggestions"], ["enhance edge sharpness"])

    def test_jsonl_dataset_skips_records_without_cleaned_profile(self):
        if torch is None:
            self.skipTest("torch is not installed in this environment")
        from dataloaders.rg_flux_jsonl_dataset import RGFluxSRJsonlDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hq_path, lq_path = self._make_pair(root)
            jsonl_path = root / "valid.jsonl"
            missing_profile = {
                "hq_path": str(hq_path),
                "lq_path": str(lq_path),
                "result": {"reasoning": {"degradation_analysis": "old invalid reasoning"}},
            }
            valid_profile = {
                "hq_path": str(hq_path),
                "lq_path": str(lq_path),
                "unipercept_raw": {
                    "profile": {
                        "iqa": {"overall_quality": "valid cleaned quality"},
                        "suggestion": "valid cleaned suggestion",
                        "iaa": {"comprehensive": "valid cleaned aesthetic"},
                    }
                },
                "result": {"score": 12, "degradation_vector": {"blur": 0.4}},
            }
            jsonl_path.write_text(
                json.dumps(missing_profile) + "\n" + json.dumps(valid_profile) + "\n",
                encoding="utf-8",
            )

            dataset = RGFluxSRJsonlDataset(
                jsonl_path=str(jsonl_path),
                crop_size=32,
                scale=4,
                mode="val",
                use_prompt=True,
                pre_cropped=False,
            )
            sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertIn("valid cleaned quality", sample["prompt"])
        self.assertNotIn("old invalid reasoning", sample["prompt"])

    def test_prompt_curriculum_switches_on_optimizer_step_boundary(self):
        try:
            from train_rg_flux_sr import resolve_batch_prompts, resolve_prompt_schedule
        except ModuleNotFoundError as exc:
            self.skipTest(f"training dependencies are not installed: {exc}")

        config = {
            "condition": {
                "prompt_schedule": {
                    "enabled": True,
                    "switch_step": 10000,
                    "before_variant": "fixed",
                    "after_variant": "suggestion",
                }
            }
        }
        profile = {
            "iqa": {"overall_quality": "Moderate blur."},
            "suggestion": "Moderately restore stable edges.",
        }
        batch = {"prompt": ["legacy prompt"], "profile": [profile]}
        schedule = resolve_prompt_schedule(config)

        before_prompts, before_variant = resolve_batch_prompts(
            batch,
            config,
            global_step=9999,
            prompt_schedule=schedule,
        )
        after_prompts, after_variant = resolve_batch_prompts(
            batch,
            config,
            global_step=10000,
            prompt_schedule=schedule,
        )

        self.assertEqual(
            before_variant,
            {"variant": "fixed", "include_caption": False},
        )
        self.assertNotIn("Moderately restore stable edges.", before_prompts[0])
        self.assertEqual(
            after_variant,
            {"variant": "suggestion", "include_caption": False},
        )
        self.assertIn("Moderately restore stable edges.", after_prompts[0])

    def test_jsonl_dataset_full_frame_branch_preserves_composition_and_aligns_to_32(self):
        if torch is None:
            self.skipTest("torch is not installed in this environment")
        try:
            from dataloaders.rg_flux_jsonl_dataset import RGFluxSRJsonlDataset
        except ModuleNotFoundError as exc:
            self.skipTest(f"dataset dependencies are not installed: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hq_path = root / "hq_wide.png"
            lq_path = root / "lq_wide.png"
            Image.new("RGB", (1000, 700), color=(128, 96, 64)).save(hq_path)
            Image.new("RGB", (250, 175), color=(64, 96, 128)).save(lq_path)
            jsonl_path = root / "valid.jsonl"
            record = {
                "hq_path": str(hq_path),
                "lq_path": str(lq_path),
                "unipercept_raw": {
                    "profile": {
                        "iqa": {"overall_quality": "Moderate blur."},
                        "suggestion": "Restore stable edges.",
                    }
                },
                "result": {"score": 1, "degradation_vector": {}},
            }
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = RGFluxSRJsonlDataset(
                jsonl_path=str(jsonl_path),
                crop_size=512,
                scale=4,
                mode="train",
                return_profile=True,
                pre_cropped=False,
                mixed_crop_enabled=True,
                full_frame_ratio=1.0,
                full_frame_max_long_side=768,
                full_frame_align=32,
                full_frame_pad_mode="reflect",
            )
            sample = dataset[0]

        self.assertEqual(sample["spatial_mode"], "full_frame")
        self.assertEqual(tuple(sample["hq"].shape), (3, 544, 768))
        self.assertEqual(sample["lq_up"].shape, sample["hq"].shape)
        self.assertEqual(sample["lq"].shape, sample["hq"].shape)
        self.assertEqual(sample["profile"]["suggestion"], "Restore stable edges.")
        self.assertEqual(sample["hq"].shape[-2] % 32, 0)
        self.assertEqual(sample["hq"].shape[-1] % 32, 0)

    def test_curriculum_mixed_crop_config_is_opt_in_and_memory_bounded(self):
        old_config = yaml.safe_load(
            Path("configs/train_rg_flux2_klein_sr_stage0b_512.yaml").read_text(encoding="utf-8")
        )
        new_config = yaml.safe_load(
            Path(
                "configs/train_rg_flux2_klein_sr_stage0b_512_prompt_curriculum_mixedcrop.yaml"
            ).read_text(encoding="utf-8")
        )

        self.assertNotIn("mixed_crop", old_config["data"])
        self.assertNotIn("prompt_schedule", old_config["condition"])
        self.assertEqual(new_config["condition"]["prompt_schedule"]["switch_step"], 10000)
        self.assertEqual(new_config["condition"]["prompt_schedule"]["before_variant"], "fixed")
        self.assertEqual(new_config["data"]["mixed_crop"]["full_frame_ratio"], 0.25)
        self.assertEqual(new_config["data"]["mixed_crop"]["full_frame_max_long_side"], 768)
        self.assertEqual(new_config["loss"]["image_loss_crop_size"], 512)

    def test_degradation_vector_encoder_outputs_context_tokens(self):
        if torch is None:
            self.skipTest("torch is not installed in this environment")
        from models.degradation_vector_encoder import DegradationVectorEncoder

        encoder = DegradationVectorEncoder(
            in_dim=8,
            hidden_dim=16,
            context_dim=12,
            num_tokens=4,
            dropout=0.0,
        )
        tokens = encoder(torch.ones(2, 8))

        self.assertEqual(tokens.shape, (2, 4, 12))

    def test_lr_condition_encoder_latent_adapter_shape(self):
        if torch is None:
            self.skipTest("torch is not installed in this environment")
        from models.lr_condition_encoder import LRConditionEncoder

        encoder = LRConditionEncoder(
            latent_channels=16,
            context_dim=24,
            num_tokens=8,
            mode="latent_adapter",
            dropout=0.0,
        )
        tokens = encoder(torch.randn(2, 16, 8, 8))

        self.assertEqual(tokens.shape, (2, 8, 24))

    def test_flow_matching_helper_builds_velocity_target(self):
        if torch is None:
            self.skipTest("torch is not installed in this environment")
        from rg_flux_fm import build_flow_matching_inputs

        z_hr = torch.ones(2, 4, 2, 2)
        eps = torch.zeros_like(z_hr)
        sigma = torch.tensor([0.25, 0.75])

        z_t, v_target = build_flow_matching_inputs(z_hr, eps=eps, sigma=sigma)

        self.assertTrue(torch.allclose(z_t[0], torch.full_like(z_t[0], 0.75)))
        self.assertTrue(torch.allclose(z_t[1], torch.full_like(z_t[1], 0.25)))
        self.assertTrue(torch.allclose(v_target, -torch.ones_like(z_hr)))


if __name__ == "__main__":
    unittest.main()
