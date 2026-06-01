import unittest


class PromptTemplateTests(unittest.TestCase):
    def test_render_prompt_b_mentions_suggestion_and_token_budgets(self):
        from profile_cleaner.prompts import render_prompt_b

        prompt = render_prompt_b({"iaa": {}, "iqa": {}, "ista": {}})
        prompt_lower = prompt.lower()

        self.assertIn("suggestion", prompt_lower)
        self.assertIn("no more than 20 tokens", prompt_lower)
        self.assertIn("around 90 tokens", prompt_lower)
        self.assertIn("no more than 25 tokens", prompt_lower)
        self.assertIn("token budgets", prompt_lower)
        self.assertNotIn("characters total", prompt_lower)
        self.assertIn("moderately reduce blur", prompt_lower)


if __name__ == "__main__":
    unittest.main()
