import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_iqa_sr_suggestion_jsonl import (
    FALLBACK_SUGGESTION,
    PRESERVATION_SUFFIX,
    compile_suggestion,
    convert_record,
    get_iqa,
    normalize_selected_degradations,
    render_user_prompt,
    record_resume_key,
    replace_suggestion,
)


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, _iqa):
        return copy.deepcopy(self.payload)


def make_record():
    return {
        "hq_path": "/data/hq.png",
        "lq_path": "/data/lq.png",
        "raw_degradation_params": {"keep": "unchanged"},
        "unipercept_raw": {
            "profile": {
                "iaa": {"comprehensive": "Do not send this to Qwen."},
                "iqa": {
                    "distortion_type": "Moderate blur and visible noise.",
                    "distortion_location": "Noise is visible in the sky and background.",
                    "distortion_severity": "Moderate blur reduces edge clarity; mild noise is present.",
                    "overall_quality": "Blur and noise reduce technical fidelity.",
                },
                "suggestion": "Old unsafe suggestion.",
                "ista": {"keep": "unchanged"},
            }
        },
        "result": {"keep": "unchanged"},
    }


class IqaSrSuggestionJsonlTests(unittest.TestCase):
    def test_user_prompt_contains_only_iqa_payload(self):
        record = make_record()
        prompt = render_user_prompt(get_iqa(record))
        self.assertIn("distortion_type", prompt)
        self.assertNotIn("Do not send this to Qwen", prompt)
        self.assertNotIn("Old unsafe suggestion", prompt)

    def test_compiler_ignores_model_free_text_and_preserves_record_structure(self):
        record = make_record()
        original = copy.deepcopy(record)
        client = FakeClient(
            {
                "selected_degradations": [
                    {
                        "type": "blur",
                        "level": "moderate",
                        "evidence_fields": ["distortion_type", "distortion_severity"],
                    },
                    {
                        "type": "noise",
                        "level": "mild",
                        "evidence_fields": ["distortion_type", "distortion_severity"],
                    },
                ],
                "rejected_claims": [],
                "suggestion": "Strongly recolor the sky and enhance contrast.",
            }
        )

        converted, audit = convert_record(record, client)
        suggestion = converted["unipercept_raw"]["profile"]["suggestion"]
        self.assertIn("Moderately reduce visible blur", suggestion)
        self.assertIn("Mildly suppress visible noise", suggestion)
        self.assertTrue(suggestion.endswith(PRESERVATION_SUFFIX))
        self.assertNotIn("sky", suggestion.lower())
        self.assertNotIn("contrast", suggestion.lower())

        converted_profile = converted["unipercept_raw"]["profile"]
        original_profile = original["unipercept_raw"]["profile"]
        converted_profile_without_suggestion = copy.deepcopy(converted_profile)
        original_profile_without_suggestion = copy.deepcopy(original_profile)
        converted_profile_without_suggestion.pop("suggestion")
        original_profile_without_suggestion.pop("suggestion")
        self.assertEqual(converted_profile_without_suggestion, original_profile_without_suggestion)
        self.assertEqual(converted["raw_degradation_params"], original["raw_degradation_params"])
        self.assertEqual(converted["result"], original["result"])
        self.assertEqual(len(audit["selected_degradations"]), 2)

    def test_location_only_evidence_is_rejected(self):
        iqa = {
            "distortion_type": "Blur only.",
            "distortion_location": "Noise appears in the sky.",
            "distortion_severity": "Moderate blur.",
            "overall_quality": "Noise is locally visible.",
        }
        selected = normalize_selected_degradations(
            {
                "selected_degradations": [
                    {"type": "noise", "level": "moderate", "evidence_fields": ["distortion_location"]}
                ]
            },
            iqa,
        )
        self.assertEqual(selected, [])
        self.assertEqual(compile_suggestion(selected), FALLBACK_SUGGESTION)

    def test_minimal_evidence_is_rejected(self):
        iqa = {
            "distortion_type": "Compression artifacts are minimal.",
            "distortion_location": "Compression is negligible.",
            "distortion_severity": "Minimal compression is present.",
            "overall_quality": "Technical quality is otherwise stable.",
        }
        selected = normalize_selected_degradations(
            {
                "selected_degradations": [
                    {
                        "type": "compression_artifacts",
                        "level": "moderate",
                        "evidence_fields": ["distortion_type", "distortion_severity"],
                    }
                ]
            },
            iqa,
        )
        self.assertEqual(selected, [])

    def test_moderate_level_is_downgraded_without_moderate_evidence(self):
        iqa = {
            "distortion_type": "Slight blur.",
            "distortion_location": "Blur is visible.",
            "distortion_severity": "Mild blur affects clarity.",
            "overall_quality": "Slightly reduced fidelity.",
        }
        selected = normalize_selected_degradations(
            {
                "selected_degradations": [
                    {"type": "blur", "level": "moderate", "evidence_fields": ["distortion_type"]}
                ]
            },
            iqa,
        )
        self.assertEqual(selected[0]["level"], "mild")

    def test_replace_suggestion_does_not_mutate_source(self):
        record = make_record()
        converted = replace_suggestion(record, "New suggestion.")
        self.assertEqual(record["unipercept_raw"]["profile"]["suggestion"], "Old unsafe suggestion.")
        self.assertEqual(converted["unipercept_raw"]["profile"]["suggestion"], "New suggestion.")
        self.assertEqual(converted["annotation_status"]["suggestion"], "complete")

    def test_resume_prefers_sample_id_and_falls_back_to_hq_path(self):
        record = make_record()
        record["sample_id"] = "crop-sample-1"
        self.assertEqual(
            record_resume_key(record),
            ("sample_id", "crop-sample-1"),
        )
        record.pop("sample_id")
        self.assertEqual(
            record_resume_key(record),
            ("hq_path", "/data/hq.png"),
        )


if __name__ == "__main__":
    unittest.main()
