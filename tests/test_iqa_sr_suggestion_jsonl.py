import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_iqa_sr_suggestion_jsonl import (
    FALLBACK_SUGGESTION,
    PRESERVATION_SUFFIX,
    SUGGESTION_MAX_WORDS,
    build_parser,
    compile_suggestion,
    count_words,
    convert_record,
    get_iqa,
    normalize_selected_degradations,
    render_user_prompt,
    record_resume_key,
    replace_suggestion,
)
from models.prompt_builder import PROFILE_WORD_LIMITS, normalize_bounded_text


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
                        "target": "the sky and background",
                        "location_evidence": "Noise is visible in the sky and background.",
                    },
                ],
                "rejected_claims": [],
                "suggestion": "Strongly recolor the sky and enhance contrast.",
            }
        )

        converted, audit = convert_record(record, client)
        suggestion = converted["unipercept_raw"]["profile"]["suggestion"]
        self.assertIn("Moderately reduce visible blur", suggestion)
        self.assertIn(
            "Mildly suppress noise in the sky and background",
            suggestion,
        )
        self.assertTrue(suggestion.endswith(PRESERVATION_SUFFIX))
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
        self.assertTrue(audit["include_location"])
        self.assertEqual(
            audit["selected_degradations"][1]["location_status"],
            "accepted",
        )

    def test_face_location_is_preserved_from_crop_local_iqa(self):
        record = make_record()
        record["unipercept_raw"]["profile"]["iqa"] = {
            "distortion_type": "Moderate blur is present.",
            "distortion_location": (
                "The person's face near the center is blurred."
            ),
            "distortion_severity": (
                "Moderate blur reduces facial edge clarity."
            ),
            "overall_quality": "Local blur reduces technical fidelity.",
        }
        client = FakeClient(
            {
                "selected_degradations": [
                    {
                        "type": "blur",
                        "level": "moderate",
                        "evidence_fields": [
                            "distortion_type",
                            "distortion_location",
                            "distortion_severity",
                        ],
                        "target": "person's face near the center",
                        "location_evidence": (
                            "The person's face near the center is blurred."
                        ),
                    }
                ],
                "rejected_claims": [],
                "suggestion": "Retouch and beautify the person's face.",
            }
        )

        converted, audit = convert_record(record, client)
        suggestion = converted["unipercept_raw"]["profile"]["suggestion"]

        self.assertIn("person's face near the center", suggestion)
        self.assertNotIn("retouch", suggestion.lower())
        self.assertNotIn("beautify", suggestion.lower())
        self.assertEqual(
            audit["selected_degradations"][0]["location_status"],
            "accepted",
        )

    def test_unsupported_target_falls_back_to_generic_action(self):
        iqa = {
            "distortion_type": "Moderate blur is present.",
            "distortion_location": "Blur is visible on the face near center.",
            "distortion_severity": "Moderate blur reduces edge clarity.",
            "overall_quality": "Blur reduces technical fidelity.",
        }
        selected = normalize_selected_degradations(
            {
                "selected_degradations": [
                    {
                        "type": "blur",
                        "level": "moderate",
                        "target": "the license plate",
                        "location_evidence": (
                            "Blur is visible on the face near center."
                        ),
                    }
                ]
            },
            iqa,
        )

        self.assertEqual(selected[0]["location_status"], "rejected")
        suggestion = compile_suggestion(selected)
        self.assertIn("Moderately reduce visible blur", suggestion)
        self.assertNotIn("license plate", suggestion)

    def test_location_evidence_must_support_the_same_degradation(self):
        iqa = {
            "distortion_type": "Moderate blur and noise are present.",
            "distortion_location": (
                "Noise is visible in the sky. "
                "Blur affects the face near center."
            ),
            "distortion_severity": "Moderate blur and mild noise.",
            "overall_quality": "Blur and noise reduce fidelity.",
        }
        selected = normalize_selected_degradations(
            {
                "selected_degradations": [
                    {
                        "type": "blur",
                        "level": "moderate",
                        "target": "the sky",
                        "location_evidence": "Noise is visible in the sky.",
                    }
                ]
            },
            iqa,
        )

        self.assertEqual(selected[0]["location_status"], "rejected")
        self.assertNotIn("sky", compile_suggestion(selected))

    def test_target_must_match_the_degradation_within_a_mixed_clause(self):
        iqa = {
            "distortion_type": "Moderate blur and noise are present.",
            "distortion_location": (
                "Blur affects the face near center, while noise is visible "
                "in the sky."
            ),
            "distortion_severity": "Moderate blur and mild noise.",
            "overall_quality": "Blur and noise reduce fidelity.",
        }
        selected = normalize_selected_degradations(
            {
                "selected_degradations": [
                    {
                        "type": "blur",
                        "level": "moderate",
                        "target": "the sky",
                        "location_evidence": (
                            "Blur affects the face near center, while noise "
                            "is visible in the sky."
                        ),
                    }
                ]
            },
            iqa,
        )

        self.assertEqual(selected[0]["location_status"], "rejected")
        self.assertIn(
            "another degradation type",
            selected[0]["location_reason"],
        )
        suggestion = compile_suggestion(selected)
        self.assertIn("Moderately reduce visible blur", suggestion)
        self.assertNotIn("sky", suggestion)

    def test_location_can_be_disabled_for_legacy_random_crops(self):
        record = make_record()
        client = FakeClient(
            {
                "selected_degradations": [
                    {
                        "type": "noise",
                        "level": "mild",
                        "target": "the sky and background",
                        "location_evidence": (
                            "Noise is visible in the sky and background."
                        ),
                    }
                ],
                "rejected_claims": [],
                "suggestion": "Denoise the sky.",
            }
        )

        converted, audit = convert_record(
            record,
            client,
            include_location=False,
        )
        suggestion = converted["unipercept_raw"]["profile"]["suggestion"]

        self.assertNotIn("sky", suggestion.lower())
        self.assertIn("Mildly suppress visible noise", suggestion)
        self.assertFalse(audit["include_location"])
        self.assertEqual(
            audit["selected_degradations"][0]["location_status"],
            "disabled",
        )

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

    def test_suggestion_word_budget_is_100(self):
        self.assertEqual(SUGGESTION_MAX_WORDS, 100)
        self.assertEqual(PROFILE_WORD_LIMITS["suggestion"], 100)

        one_hundred = " ".join(f"word{index}" for index in range(100))
        one_hundred_one = one_hundred + " overflow"
        self.assertEqual(
            len(normalize_bounded_text(one_hundred, 100).split()),
            100,
        )
        normalized = normalize_bounded_text(one_hundred_one, 100)
        self.assertEqual(len(normalized.split()), 100)
        self.assertNotIn("overflow", normalized)

    def test_three_location_actions_stay_within_word_budget(self):
        selected = [
            {
                "type": "blur",
                "level": "moderate",
                "target": "the face near the center",
                "location_status": "accepted",
            },
            {
                "type": "noise",
                "level": "mild",
                "target": "the sky and background",
                "location_status": "accepted",
            },
            {
                "type": "compression_artifacts",
                "level": "moderate",
                "target": "the text along the lower edge",
                "location_status": "accepted",
            },
        ]
        suggestion = compile_suggestion(selected)

        self.assertLessEqual(count_words(suggestion), 100)
        self.assertIn("face near the center", suggestion)
        self.assertIn("sky and background", suggestion)
        self.assertIn("text along the lower edge", suggestion)

    def test_cli_defaults_to_location_and_supports_opt_out(self):
        parser = build_parser()
        default_args = parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl"])
        legacy_args = parser.parse_args(
            [
                "--input",
                "in.jsonl",
                "--output",
                "out.jsonl",
                "--no-include-location",
            ]
        )

        self.assertTrue(default_args.include_location)
        self.assertFalse(legacy_args.include_location)

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
