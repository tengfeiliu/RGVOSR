import unittest


class ProfileCleanerValidatorTests(unittest.TestCase):
    def test_iaa_forbidden_quality_terms_are_detected(self):
        from profile_cleaner.validators import contains_forbidden_in_iaa

        terms = contains_forbidden_in_iaa("Low resolution and blurriness reduce the layout impact.")

        self.assertIn("low resolution", terms)
        self.assertIn("blurriness", terms)

    def test_iqa_forbidden_aesthetic_terms_are_detected(self):
        from profile_cleaner.validators import contains_forbidden_in_iqa

        terms = contains_forbidden_in_iqa("Composition and viewer engagement are weak.")

        self.assertIn("composition", terms)
        self.assertIn("viewer", terms)
        self.assertIn("engagement", terms)

    def test_valid_profile_has_no_strict_separation_violations(self):
        from profile_cleaner.validators import validate_strict_separation

        profile = {
            "iaa": {
                "composition_design": "- The layout uses stable framing and balanced spatial organization.",
                "emotion_viewer_response": "- The scene has a subdued emotional tone.",
            },
            "iqa": {
                "distortion_type": "- Blur and noise are visible.",
                "overall_quality": "- Edge clarity and recognizability are limited.",
            },
            "ista": {"kept": True},
        }

        report = validate_strict_separation(profile)

        self.assertTrue(report["valid"])
        self.assertEqual(report["iaa_violations"], [])
        self.assertEqual(report["iqa_violations"], [])

    def test_strict_separation_reports_recursive_paths(self):
        from profile_cleaner.validators import validate_strict_separation

        profile = {
            "iaa": {"nested": {"field": ["The image has noise.", "The framing is plain."]}},
            "iqa": {"nested": {"field": ["The visual hierarchy is weak.", "Blur is visible."]}},
        }

        report = validate_strict_separation(profile)

        self.assertFalse(report["valid"])
        self.assertEqual(report["iaa_violations"][0]["path"], "profile.iaa.nested.field[0]")
        self.assertEqual(report["iqa_violations"][0]["path"], "profile.iqa.nested.field[0]")


if __name__ == "__main__":
    unittest.main()
