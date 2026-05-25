"""Wrapper around UniPercept's src/eval/conversation.py.

Patches InternVLChatConfig.has_no_defaults_at_init=True before
transformers.from_pretrained triggers a logger.info(f"Model config {config}")
that would otherwise call the parameterless InternVLChatConfig() and raise
"Unsupported architecture:" on the empty default config.

Run with cwd set to the UniPercept repo root. Forwards argv to conversation.py.
"""
import runpy
import sys
from pathlib import Path


def _patch_config():
    sys.path.insert(0, "src")
    from internvl.model.internvl_chat.configuration_internvl_chat import (
        InternVLChatConfig,
    )
    InternVLChatConfig.has_no_defaults_at_init = True


def main():
    _patch_config()
    script = Path("src/eval/conversation.py").resolve()
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
