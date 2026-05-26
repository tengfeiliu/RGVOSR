import json
import tempfile
import unittest
from pathlib import Path


class ProfileCleanerJsonUtilsTests(unittest.TestCase):
    def test_parse_pure_json(self):
        from profile_cleaner.json_utils import parse_json_strict_or_extract

        self.assertEqual(parse_json_strict_or_extract('{"a": 1}'), {"a": 1})

    def test_parse_fenced_json(self):
        from profile_cleaner.json_utils import parse_json_strict_or_extract

        raw = """```json
        {"profile": {"iaa": {}, "iqa": {}}}
        ```"""

        self.assertEqual(parse_json_strict_or_extract(raw)["profile"], {"iaa": {}, "iqa": {}})

    def test_extract_json_with_surrounding_text(self):
        from profile_cleaner.json_utils import parse_json_strict_or_extract

        raw = 'Here is the result:\n{"profile": {"iaa": {"x": "y"}, "iqa": {}}}\nDone.'

        self.assertEqual(parse_json_strict_or_extract(raw)["profile"]["iaa"]["x"], "y")

    def test_invalid_json_raises(self):
        from profile_cleaner.json_utils import parse_json_strict_or_extract

        with self.assertRaises(ValueError):
            parse_json_strict_or_extract("not json at all")

    def test_jsonl_round_trip_preserves_unicode(self):
        from profile_cleaner.json_utils import load_json_or_jsonl, save_json_or_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.jsonl"
            save_json_or_jsonl([{"text": "中文"}], path, jsonl=True, overwrite=False)
            loaded = load_json_or_jsonl(path, jsonl=True)

        self.assertEqual(loaded, [{"text": "中文"}])
        self.assertEqual(json.dumps(loaded[0], ensure_ascii=False), '{"text": "中文"}')


if __name__ == "__main__":
    unittest.main()
