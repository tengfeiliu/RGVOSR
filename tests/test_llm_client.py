import os
import types
import unittest
from unittest import mock


class ProfileCleanerLLMClientTests(unittest.TestCase):
    def test_defaults_use_qwen_dashscope_compatible_api(self):
        from profile_cleaner.config import DEFAULT_BASE_URL, DEFAULT_MODEL

        self.assertEqual(DEFAULT_MODEL, "qwen2.5-vl-72b-instruct")
        self.assertEqual(DEFAULT_BASE_URL, "https://dashscope.aliyuncs.com/compatible-mode/v1")

    def test_dashscope_api_key_is_accepted_by_default(self):
        calls = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                calls["kwargs"] = kwargs
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=lambda **create_kwargs: types.SimpleNamespace(
                            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
                        )
                    )
                )

        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        with mock.patch.dict("sys.modules", {"openai": fake_openai_module}), mock.patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "dashscope-key",
            },
            clear=True,
        ):
            from profile_cleaner.llm_client import LLMClient

            client = LLMClient()

        self.assertEqual(calls["kwargs"]["api_key"], "dashscope-key")
        self.assertEqual(calls["kwargs"]["base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(client.model, "qwen2.5-vl-72b-instruct")


if __name__ == "__main__":
    unittest.main()
