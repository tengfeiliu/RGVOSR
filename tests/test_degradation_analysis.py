import json
import unittest


class DegradationAnalysisTests(unittest.TestCase):
    def test_parse_repairs_fenced_json(self):
        from tools.qwen_semantic_risk_analyzer import parse_json_object

        raw = """```json
        {
          "reasoning": {
            "degradation_analysis": "low-quality input",
            "texture_edge_analysis": "thin edges are weak",
            "semantic_risk_analysis": "text is visible",
            "sr_strategy": "restore carefully"
          },
          "suggestions": ["improve text readability", "avoid hallucinated details"],
          "score": 0,
          "degradation_vector": {
            "text_region_risk": 0.8,
            "hallucination_risk": 0.5
          }
        }
        ```"""

        parsed = parse_json_object(raw)
        self.assertEqual(parsed["degradation_vector"]["text_region_risk"], 0.8)

    def test_semantic_validation_rejects_empty_suggestions(self):
        from tools.qwen_semantic_risk_analyzer import normalize_semantic_result

        payload = {
            "reasoning": {
                "degradation_analysis": "blurred",
                "texture_edge_analysis": "weak",
                "semantic_risk_analysis": "none",
                "sr_strategy": "careful",
            },
            "suggestions": [],
            "score": 0,
            "degradation_vector": {
                "text_region_risk": 0.0,
                "hallucination_risk": 0.0,
            },
        }

        with self.assertRaises(ValueError):
            normalize_semantic_result(payload)

    def test_semantic_validation_rejects_risky_suggestion_with_zero_risk(self):
        from tools.qwen_semantic_risk_analyzer import normalize_semantic_result

        payload = {
            "reasoning": {
                "degradation_analysis": "some artifacts",
                "texture_edge_analysis": "some edges",
                "semantic_risk_analysis": "text is visible",
                "sr_strategy": "improve text",
            },
            "suggestions": ["improve text readability"],
            "score": 0,
            "degradation_vector": {
                "text_region_risk": 0.0,
                "hallucination_risk": 0.0,
            },
        }

        with self.assertRaises(ValueError):
            normalize_semantic_result(payload)

    def test_merge_uses_physical_values_and_semantic_risks(self):
        from dataloaders.degradation_meta import merge_analysis_result

        physical = {
            "blur": 0.6,
            "noise": 0.2,
            "jpeg": 0.4,
            "ringing": 0.3,
            "texture_loss": 0.7,
            "color_shift": 0.1,
        }
        semantic = {
            "reasoning": {
                "degradation_analysis": "physical fields are supplied by the pipeline",
                "texture_edge_analysis": "thin texture regions are fragile",
                "semantic_risk_analysis": "small text is visible",
                "sr_strategy": "keep details anchored",
            },
            "suggestions": ["recover fine textures", "improve text readability"],
            "degradation_vector": {
                "text_region_risk": 0.8,
                "hallucination_risk": 0.2,
            },
        }

        result = merge_analysis_result(physical, semantic)

        self.assertEqual(result["degradation_vector"]["blur"], 0.6)
        self.assertEqual(result["degradation_vector"]["text_region_risk"], 0.8)
        self.assertGreaterEqual(result["degradation_vector"]["hallucination_risk"], 0.28)
        self.assertLess(result["score"], 100)
        self.assertGreaterEqual(result["score"], 0)

    def test_jsonl_schema_is_serializable(self):
        from dataloaders.degradation_meta import make_cache_record

        record = make_cache_record(
            hq_path="hq.png",
            lq_path="lq.png",
            raw_degradation_params={"stage": "second_order"},
            result={
                "reasoning": {
                    "degradation_analysis": "x",
                    "texture_edge_analysis": "x",
                    "semantic_risk_analysis": "x",
                    "sr_strategy": "x",
                },
                "suggestions": ["suppress noise"],
                "score": 80,
                "degradation_vector": {
                    "blur": 0.1,
                    "noise": 0.2,
                    "jpeg": 0.3,
                    "ringing": 0.1,
                    "texture_loss": 0.2,
                    "text_region_risk": 0.0,
                    "color_shift": 0.0,
                    "hallucination_risk": 0.1,
                },
            },
        )

        encoded = json.dumps(record, ensure_ascii=False)
        self.assertIn("raw_degradation_params", encoded)


if __name__ == "__main__":
    unittest.main()
