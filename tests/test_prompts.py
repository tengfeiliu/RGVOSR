import unittest


class PromptTemplateTests(unittest.TestCase):
    def test_render_prompt_b_mentions_suggestion_and_token_budgets(self):
        from profile_cleaner.prompts import render_prompt_b

        prompt = render_prompt_b({"iaa": {}, "iqa": {}, "ista": {}})
        prompt_lower = prompt.lower()

        self.assertIn("suggestion", prompt_lower)
        self.assertIn("no more than 50 tokens", prompt_lower)
        self.assertIn("around 350 tokens", prompt_lower)
        self.assertIn("no more than 80 tokens", prompt_lower)
        self.assertIn("token budgets", prompt_lower)
        self.assertIn("do not interpret tokens as english characters", prompt_lower)
        self.assertIn("distortion_location", prompt_lower)
        self.assertIn("distortion_severity", prompt_lower)
        self.assertIn("distortion_type", prompt_lower)
        self.assertIn("overall_quality", prompt_lower)
        self.assertIn("must all be present and non-empty", prompt_lower)
        self.assertIn("based on the original profile evidence", prompt_lower)
        self.assertIn("restored outside this prompt", prompt_lower)
        self.assertNotIn("ista, if it existed", prompt_lower)
        self.assertNotIn("characters total", prompt_lower)
        self.assertIn("moderately reduce blur", prompt_lower)


if __name__ == "__main__":
    unittest.main()
