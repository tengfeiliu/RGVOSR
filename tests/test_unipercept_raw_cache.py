import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


class UniPerceptRawCacheTests(unittest.TestCase):
    def _make_image(self, path):
        Image.new("RGB", (16, 16), color=(32, 64, 96)).save(path)
        return path

    def test_dataset_config_uses_first_column_and_expands_nested_txt_lists(self):
        from tools.generate_unipercept_raw_cache import list_hq_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = self._make_image(root / "a.png")
            image_b = self._make_image(root / "b.jpg")
            shard = root / "shard.txt"
            shard.write_text(f"{image_a}\n", encoding="utf-8")
            dataset_config = root / "train_dataset_txt.txt"
            dataset_config.write_text(f"{shard}, 1\n{image_b}, 2\n", encoding="utf-8")

            images = list_hq_images(dataset_config)

        self.assertEqual(images, [image_a, image_b])

    def test_default_result_has_empty_raw_training_fields_and_zero_vector(self):
        from dataloaders.degradation_meta import DEGRADATION_KEYS, REASONING_KEYS
        from tools.generate_unipercept_raw_cache import default_empty_result

        result = default_empty_result()

        self.assertEqual(result["suggestions"], [])
        self.assertEqual(result["score"], 0)
        self.assertEqual(set(result["reasoning"].keys()), set(REASONING_KEYS))
        self.assertTrue(all(value == "" for value in result["reasoning"].values()))
        self.assertEqual(set(result["degradation_vector"].keys()), set(DEGRADATION_KEYS))
        self.assertTrue(all(value == 0.0 for value in result["degradation_vector"].values()))

    def test_process_image_preserves_unipercept_raw_outputs(self):
        from tools import generate_unipercept_raw_cache as module

        class FakeArgs:
            lq_output_dir = "unused_lq"
            resize_bak = True

        class FakeDegradation:
            def degrade_process(self, hq, resize_bak=True, return_meta=True):
                return None, "fake_lq_tensor", {"stage": "fake"}

        class FakeAnalyzer:
            def analyze(self, image_path):
                return {
                    "iaa": {"score": 0.1, "label": "aesthetic"},
                    "iqa": {"score": 0.2, "label": "quality"},
                    "ista": {"score": 0.3, "label": "text-alignment"},
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self._make_image(root / "hq.png")

            with mock.patch.object(module, "load_image_tensor", return_value="fake_hq_tensor"), mock.patch.object(
                module, "save_lq_tensor"
            ):
                record = module.process_image(image, FakeArgs(), FakeDegradation(), "cpu", FakeAnalyzer())

        self.assertEqual(record["hq_path"], str(image))
        self.assertIn("lq_path", record)
        self.assertEqual(record["raw_degradation_params"], {"stage": "fake"})
        self.assertEqual(record["unipercept_raw"]["iaa"]["label"], "aesthetic")
        self.assertEqual(record["unipercept_raw"]["iqa"]["score"], 0.2)
        self.assertEqual(record["result"]["degradation_vector"]["blur"], 0.0)

    def test_resume_seen_paths_use_hq_path(self):
        from tools.generate_unipercept_raw_cache import load_seen_hq_paths

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "valid.jsonl"
            invalid = Path(tmp) / "invalid.jsonl"
            output.write_text(json.dumps({"hq_path": "a.png"}) + "\n", encoding="utf-8")
            invalid.write_text(json.dumps({"hq_path": "b.png"}) + "\n", encoding="utf-8")

            seen = load_seen_hq_paths(output, invalid)

        self.assertEqual(seen, {"a.png", "b.png"})

    def test_reward_backend_requires_local_model_path(self):
        from tools import generate_unipercept_raw_cache as module

        with self.assertRaisesRegex(ValueError, "--unipercept-model-path is required"):
            module.UniPerceptRawAnalyzer(device="cpu", backend="reward")

    def test_reward_backend_forces_local_hf_loading(self):
        from tools import generate_unipercept_raw_cache as module

        captured = {}

        class FakeRewardInferencer:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                captured["hf_hub_offline"] = os.environ.get("HF_HUB_OFFLINE")
                captured["transformers_offline"] = os.environ.get("TRANSFORMERS_OFFLINE")

        fake_module = types.SimpleNamespace(UniPerceptRewardInferencer=FakeRewardInferencer)

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "UniPercept-model"
            model_dir.mkdir()
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
                "sys.modules", {"unipercept_reward": fake_module}
            ):
                module.UniPerceptRawAnalyzer(device="cpu", model_path=model_dir, backend="reward")

        self.assertEqual(Path(captured["kwargs"]["model_path"]).resolve(), model_dir.resolve())
        self.assertEqual(captured["hf_hub_offline"], "1")
        self.assertEqual(captured["transformers_offline"], "1")

    def test_conversation_backend_calls_unipercept_repo_script_for_each_domain(self):
        from tools import generate_unipercept_raw_cache as module

        class Completed:
            stdout = "raw response"

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "UniPercept"
            script = repo / "src" / "eval" / "conversation.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('ok')\n", encoding="utf-8")
            model_dir = Path(tmp) / "UniPercept-model"
            model_dir.mkdir()

            with mock.patch.object(module.subprocess, "run", return_value=Completed()) as run:
                analyzer = module.UniPerceptRawAnalyzer(
                    device="cuda",
                    model_path=model_dir,
                    unipercept_repo=repo,
                    backend="conversation",
                )
                result = analyzer.analyze("lq.png")

        self.assertEqual(result, {"iaa": "raw response", "iqa": "raw response", "ista": "raw response"})
        self.assertEqual(run.call_count, 3)
        first_command = run.call_args_list[0].args[0]
        self.assertIn(str(script), first_command)
        self.assertIn("--model_path", first_command)
        self.assertIn("--image", first_command)
        self.assertIn("--prompt", first_command)
        first_env = run.call_args_list[0].kwargs["env"]
        self.assertEqual(first_env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(first_env["TRANSFORMERS_OFFLINE"], "1")

    def test_profile_prompt_keys_are_stable(self):
        from tools import generate_unipercept_raw_cache as module

        self.assertEqual(
            list(module.IAA_PROFILE_PROMPTS.keys()),
            [
                "composition_design",
                "visual_elements_structure",
                "technical_execution",
                "originality_creativity",
                "theme_communication",
                "emotion_viewer_response",
                "overall_gestalt",
                "comprehensive",
            ],
        )
        self.assertEqual(
            list(module.IQA_PROFILE_PROMPTS.keys()),
            [
                "distortion_location",
                "distortion_severity",
                "distortion_type",
                "overall_quality",
            ],
        )
        self.assertIn("Prompt for ISTA Structural Annotation", module.ISTA_STRUCTURAL_ANNOTATION_PROMPT)
        self.assertIn("Base Morphology", module.ISTA_STRUCTURAL_ANNOTATION_PROMPT)
        self.assertIn("SceneType", module.ISTA_STRUCTURAL_ANNOTATION_PROMPT)

    def test_extract_conversation_answer_strips_stdout_noise(self):
        from tools.generate_unipercept_raw_cache import extract_conversation_answer

        raw = "Loading model...\nUSER: ignored prompt\nASSISTANT: The image has visible blur.\n"

        self.assertEqual(extract_conversation_answer(raw), "The image has visible blur.")

    def test_ista_profile_response_parses_fenced_json_and_preserves_raw_fallback(self):
        from tools.generate_unipercept_raw_cache import normalize_ista_profile_response

        fenced = """```json
        {
          "SceneType": "Composite Scene",
          "SceneName": "Urban street",
          "Components": []
        }
        ```"""

        parsed = normalize_ista_profile_response(fenced)
        self.assertEqual(parsed["structural_annotation"]["SceneName"], "Urban street")
        self.assertIn("Urban street", parsed["raw_structural_annotation"])

        raw = normalize_ista_profile_response("not json")
        self.assertEqual(raw["structural_annotation"], {})
        self.assertEqual(raw["raw_structural_annotation"], "not json")

    def test_profile_backend_calls_all_profile_prompts_and_preserves_scores(self):
        from tools import generate_unipercept_raw_cache as module

        class Completed:
            stdout = "ASSISTANT: profile response"

        class FakeRewardInferencer:
            def reward(self, image_paths):
                return [{"iaa": 100, "iqa": 20, "ista": 80, "extra": "kept"}]

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "UniPercept"
            script = repo / "src" / "eval" / "conversation.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('ok')\n", encoding="utf-8")
            model_dir = Path(tmp) / "UniPercept-model"
            model_dir.mkdir()

            with mock.patch.object(
                module.UniPerceptRawAnalyzer,
                "_load_reward_inferencer",
                return_value=FakeRewardInferencer(),
            ), mock.patch.object(module.subprocess, "run", return_value=Completed()) as run:
                analyzer = module.UniPerceptRawAnalyzer(
                    device="cuda",
                    model_path=model_dir,
                    unipercept_repo=repo,
                    backend="profile",
                )
                result = analyzer.analyze("lq.png")

        self.assertEqual(run.call_count, 13)
        self.assertEqual(result["iaa"], 100)
        self.assertEqual(result["iqa"], 20)
        self.assertEqual(result["ista"], 80)
        self.assertEqual(list(result["profile"]["iaa"].keys()), list(module.IAA_PROFILE_PROMPTS.keys()))
        self.assertEqual(list(result["profile"]["iqa"].keys()), list(module.IQA_PROFILE_PROMPTS.keys()))
        self.assertIn("raw_structural_annotation", result["profile"]["ista"])

    def test_process_image_writes_profile_reasoning_and_keeps_degradation_score(self):
        from dataloaders.degradation_meta import compute_score
        from tools import generate_unipercept_raw_cache as module

        class FakeArgs:
            lq_output_dir = "unused_lq"
            resize_bak = True

        class FakeDegradation:
            def degrade_process(self, hq, resize_bak=True, return_meta=True):
                return (
                    None,
                    "fake_lq_tensor",
                    {
                        "stage": "fake",
                        "degradation_vector": {
                            "blur": 0.4,
                            "noise": 0.2,
                            "jpeg": 0.3,
                            "ringing": 0.1,
                            "texture_loss": 0.5,
                            "color_shift": 0.0,
                        },
                    },
                )

        class FakeAnalyzer:
            def analyze(self, image_path):
                return {
                    "iaa": 100,
                    "iqa": 0,
                    "ista": 80,
                    "raw_reward": {"iaa": 100, "iqa": 0, "ista": 80},
                    "profile": {
                        "iaa": {
                            "comprehensive": "The image has limited aesthetic appeal because details are weak.",
                        },
                        "iqa": {
                            "distortion_location": "Blur affects the whole image.",
                            "distortion_severity": "The distortion is moderate.",
                            "overall_quality": "Overall quality is limited by blur and texture loss.",
                        },
                        "ista": {
                            "structural_annotation": {
                                "SceneType": "Composite Scene",
                                "SceneName": "Garden scene",
                                "Components": [
                                    {
                                        "ComponentName": "Foliage",
                                        "DescriptionContent": {
                                            "PhysicalStructure": {
                                                "BaseMorphology": ["matted"],
                                                "Arrangement": ["layered"],
                                            },
                                            "MaterialRepresentation": {
                                                "MaterialClass": ["Foliage"],
                                                "SurfaceProperties": ["Matte"],
                                            },
                                            "GeometricComposition": {
                                                "PlanarContour": ["N/A"],
                                                "VolumetricForm": ["N/A"],
                                            },
                                            "SemanticPerception": {
                                                "FunctionalInference": ["N/A"],
                                                "StyleType": ["N/A"],
                                            },
                                        },
                                    }
                                ],
                            },
                            "raw_structural_annotation": "{}",
                        },
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self._make_image(root / "hq.png")

            with mock.patch.object(module, "load_image_tensor", return_value="fake_hq_tensor"), mock.patch.object(
                module, "save_lq_tensor"
            ):
                record = module.process_image(image, FakeArgs(), FakeDegradation(), "cpu", FakeAnalyzer())

        reasoning = record["result"]["reasoning"]
        self.assertIn("Overall quality is limited", reasoning["degradation_analysis"])
        self.assertIn("Garden scene", reasoning["texture_edge_analysis"])
        self.assertIn("limited aesthetic appeal", reasoning["semantic_risk_analysis"])
        self.assertIn("recover fine textures", record["result"]["suggestions"])
        self.assertEqual(record["result"]["score"], compute_score(record["result"]["degradation_vector"]))
        self.assertNotEqual(record["result"]["score"], 60)


if __name__ == "__main__":
    unittest.main()
