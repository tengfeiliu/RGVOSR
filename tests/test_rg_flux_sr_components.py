import json
import ast
import copy
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
        helper_names = {"load_jsonl_conditions", "condition_for_image"}
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

    def test_inference_jsonl_conditions_keep_records_without_profile_for_failure_logging(self):
        source = Path("inference_rg_flux_sr.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper_names = {"load_jsonl_conditions", "condition_for_image"}
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
        self.assertIn("module.load_state_dict(state, strict=False)", source)

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
            )
            sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertIn("valid cleaned quality", sample["prompt"])
        self.assertNotIn("old invalid reasoning", sample["prompt"])

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
