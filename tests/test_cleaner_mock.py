import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def sample_profile():
    return {
        "iaa": {
            "composition_design": "- Low resolution harms the composition.\n- The framing is static.",
            "emotion_viewer_response": "- The mood is subdued.",
        },
        "iqa": {
            "distortion_type": "- Blur\n- Noise",
            "overall_quality": "- Composition is weak.\n- Blur reduces recognizability.",
        },
        "ista": {"unchanged": True},
    }


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No fake response left")
        return self.responses.pop(0)


class ProfileCleanerMockTests(unittest.TestCase):
    def test_clean_one_runs_prompt_b_then_prompt_c(self):
        from profile_cleaner.cleaner import ProfileCleaner
        from profile_cleaner.validators import validate_strict_separation

        prompt_b_response = {
            "iaa": {
                "composition_design": "- The framing is static.",
                "emotion_viewer_response": "- The mood is subdued.",
            },
            "iqa": {
                "distortion_type": "- Blur\n- Noise",
                "overall_quality": "- Blur reduces recognizability.",
            },
            "ista": {"unchanged": True},
        }
        prompt_c_response = dict(prompt_b_response)
        llm = FakeLLMClient([json.dumps(prompt_b_response), json.dumps(prompt_c_response)])

        cleaned = ProfileCleaner(llm, max_retries=0).clean_one(sample_profile())

        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("Original profile", llm.prompts[0])
        self.assertIn("Now validate and repair", llm.prompts[1])
        self.assertEqual(cleaned["ista"], {"unchanged": True})
        self.assertTrue(validate_strict_separation(cleaned)["valid"])

    def test_invalid_llm_output_uses_json_repair_prompt(self):
        from profile_cleaner.cleaner import ProfileCleaner

        repaired = {
            "iaa": {"composition_design": "- The layout is simple."},
            "iqa": {"overall_quality": "- Blur is visible."},
            "ista": {"unchanged": True},
        }
        llm = FakeLLMClient(
            [
                "not json",
                json.dumps(repaired),
                json.dumps(repaired),
            ]
        )

        cleaned = ProfileCleaner(llm, max_retries=0).clean_one(sample_profile())

        self.assertEqual(cleaned["iaa"]["composition_design"], "- The layout is simple.")
        self.assertIn("JSON structure repair agent", llm.prompts[1])

    def test_local_fallback_removes_forbidden_sentences(self):
        from profile_cleaner.cleaner import ProfileCleaner
        from profile_cleaner.validators import validate_strict_separation

        contaminated = sample_profile()
        llm = FakeLLMClient([json.dumps(contaminated), json.dumps(contaminated)])

        cleaned = ProfileCleaner(llm, max_retries=0).clean_one(sample_profile())

        self.assertNotIn("Low resolution", cleaned["iaa"]["composition_design"])
        self.assertNotIn("Composition", cleaned["iqa"]["overall_quality"])
        self.assertTrue(validate_strict_separation(cleaned)["valid"])

    def test_cli_jsonl_replaces_only_nested_profile_and_keeps_failures(self):
        from profile_cleaner import cli

        clean = {
            "iaa": {"composition_design": "- The framing is stable."},
            "iqa": {"overall_quality": "- Blur is visible."},
            "ista": {"unchanged": True},
        }

        class DummyCleaner:
            def clean_one(self, profile):
                if profile.get("fail"):
                    raise RuntimeError("boom")
                return clean

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            error_log = root / "errors.jsonl"
            records = [
                {
                    "hq_path": "a.png",
                    "unipercept_raw": {"iaa": 1, "iqa": 2, "profile": sample_profile(), "raw_reward": {"x": 1}},
                    "result": {"keep": True},
                },
                {
                    "hq_path": "b.png",
                    "unipercept_raw": {"profile": {**sample_profile(), "fail": True}},
                    "result": {"keep": True},
                },
            ]
            input_path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

            with mock.patch.object(cli, "build_cleaner", return_value=DummyCleaner()):
                exit_code = cli.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--jsonl",
                        "--error-log",
                        str(error_log),
                    ]
                )

            written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            errors = [json.loads(line) for line in error_log.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(exit_code, 0)
        self.assertEqual(written[0]["unipercept_raw"]["profile"], clean)
        self.assertEqual(written[0]["unipercept_raw"]["raw_reward"], {"x": 1})
        self.assertEqual(written[0]["result"], {"keep": True})
        self.assertEqual(written[1], records[1])
        self.assertEqual(errors[0]["item_index"], 1)
        self.assertIn("boom", errors[0]["error"])

    def test_cli_dry_run_does_not_write_output_or_error_log(self):
        from profile_cleaner import cli

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            error_log = root / "errors.jsonl"
            input_path.write_text(json.dumps({"hq_path": "missing-profile.png"}) + "\n", encoding="utf-8")

            with mock.patch("sys.stdout", new=io.StringIO()):
                exit_code = cli.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--jsonl",
                        "--dry-run",
                        "--error-log",
                        str(error_log),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(output_path.exists())
            self.assertFalse(error_log.exists())


if __name__ == "__main__":
    unittest.main()
