"""OpenAI-compatible LLM client for profile cleaning."""

import os

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TEMPERATURE


class LLMClient:
    """Small wrapper around the OpenAI Python SDK chat completions API."""

    def __init__(self, api_key=None, base_url=None, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.getenv("PROFILE_CLEANER_MODEL") or DEFAULT_MODEL
        env_temperature = os.getenv("PROFILE_CLEANER_TEMPERATURE")
        self.temperature = float(temperature if temperature is not None else env_temperature or DEFAULT_TEMPERATURE)
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY or OPENAI_API_KEY is required unless --api-key is provided")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The profile cleaner requires the openai Python package.") from exc

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)

    def complete(self, prompt: str) -> str:
        """Return model text for a single prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
