import json
import tempfile
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

    def test_conversation_backend_calls_unipercept_repo_script_for_each_domain(self):
        from tools import generate_unipercept_raw_cache as module

        class Completed:
            stdout = "raw response"

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "UniPercept"
            script = repo / "src" / "eval" / "conversation.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('ok')\n", encoding="utf-8")

            with mock.patch.object(module.subprocess, "run", return_value=Completed()) as run:
                analyzer = module.UniPerceptRawAnalyzer(
                    device="cuda",
                    model_path="/models/UniPercept",
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


if __name__ == "__main__":
    unittest.main()
