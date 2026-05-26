"""Configuration defaults for the profile cleaner."""

DEFAULT_MODEL = "qwen2.5-vl-72b-instruct"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 2
PROFILE_PATH = ("unipercept_raw", "profile")
IAA_PLACEHOLDER = "The field provides limited aesthetic evidence based on the available visual description."
IQA_PLACEHOLDER = "The field provides limited objective quality evidence based on the available visual description."
DEFAULT_ERROR_LOG = "profile_cleaner_errors.jsonl"
