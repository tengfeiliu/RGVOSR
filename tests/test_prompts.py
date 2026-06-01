import unittest


class PromptTemplateTests(unittest.TestCase):
    def test_render_prompt_b_mentions_suggestion_and_character_budgets(self):
        from profile_cleaner.prompts import render_prompt_b

        prompt = render_prompt_b({"iaa": {}, "iqa": {}, "ista": {}})
        prompt_lower = prompt.lower()

        self.assertIn("suggestion", prompt_lower)
        self.assertIn("no more than 50 characters", prompt_lower)
        self.assertIn("around 370 characters", prompt_lower)
        self.assertIn("no more than 80 characters", prompt_lower)
        self.assertIn("moderately reduce blur", prompt_lower)


if __name__ == "__main__":
    unittest.main()
