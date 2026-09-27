"""Shared LLM backend configuration for all pipeline stages.

Resolves MODEL_ID / API_BASE / API_KEY from (in priority order) real
environment variables, a `.env` file, then local-llamafile defaults.
`MODEL_ID` follows any-llm's `<provider>:<model>` convention, so pointing at
a cloud backend (e.g. `anthropic:claude-sonnet-4-5`) is just a config change
— see `.env.example`.
"""

import os
from pathlib import Path

_ENV_LOADED = False


def _load_dotenv(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value


def get_model_config() -> tuple[str, str, str]:
    """Return (model_id, api_base, api_key)."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        _load_dotenv()
        _ENV_LOADED = True

    model_id = os.environ.get("MODEL_ID", "openai:Qwen2.5-7B-Instruct-Q8_0")
    api_base = os.environ.get("API_BASE", "http://localhost:8080/v1")
    api_key = os.environ.get("API_KEY", "whatever")
    return model_id, api_base, api_key
