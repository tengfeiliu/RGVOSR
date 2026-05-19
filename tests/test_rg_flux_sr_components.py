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

    def test_prompt_builder_uses_reasoning_and_suggestions(self):
        from models.prompt_builder import build_sr_prompt

        result = {
            "reasoning": {
                "degradation_analysis": "blur and JPEG artifacts",
                "texture_edge_analysis": "edges are weak",
                "semantic_risk_analysis": "text may be fragile",
                "sr_strategy": "restore conservatively",
            },
            "suggestions": ["recover fine textures", "avoid hallucinated details"],
        }

        prompt = build_sr_prompt(result, use_prompt=True, use_suggestions=True)

        self.assertIn("blur and JPEG artifacts", prompt)
        self.assertIn("- recover fine textures", prompt)
        self.assertIn("Avoid hallucinated details", prompt)

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
        flux_artist_line = None
        for node in ast.walk(main_func):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "HfDeepSpeedConfig":
                calls_hf_ds_config = True
            if isinstance(func, ast.Name) and func.id == "resolve_hf_zero3_config":
                resolves_config_line = node.lineno
            if isinstance(func, ast.Name) and func.id == "FluxSRArtist":
                flux_artist_line = node.lineno

        self.assertFalse(calls_hf_ds_config)
        self.assertIsNotNone(resolves_config_line)
        self.assertIsNotNone(flux_artist_line)
        self.assertLess(resolves_config_line, flux_artist_line)

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

    def test_default_config_uses_low_memory_frozen_encoder_devices(self):
        config = yaml.safe_load(Path("configs/train_rg_flux_sr_ms.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["model"]["text_encoder_device"], "cpu")
        self.assertEqual(config["model"]["vae_device"], "cpu")
        self.assertEqual(config["model"]["vae_dtype"], "fp32")
        self.assertLessEqual(config["model"]["max_prompt_sequence_length"], 128)

    def test_prompt_builder_can_disable_suggestions(self):
        from models.prompt_builder import build_sr_prompt

        prompt = build_sr_prompt(
            {"suggestions": ["recover fine textures"]},
            use_prompt=True,
            use_suggestions=False,
        )

        self.assertNotIn("recover fine textures", prompt)
        self.assertIn("Super-resolve this low-quality image", prompt)

    def test_jsonl_dataset_reads_pair_and_ignores_raw_fields(self):
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
        self.assertIn("offline result", sample["prompt"])
        self.assertEqual(sample["degradation_vector"].shape, (8,))
        self.assertAlmostEqual(float(sample["degradation_vector"][0]), 0.1)
        self.assertAlmostEqual(float(sample["degradation_vector"][3]), 0.0)
        self.assertEqual(float(sample["score"]), 47.0)
        self.assertEqual(sample["suggestions"], ["enhance edge sharpness"])

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
