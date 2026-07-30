"""
Central helper for constructing the Claude chat client.

Newer reasoning-style models (e.g. claude-sonnet-5, claude-opus-4-8) DEPRECATE the
`temperature` parameter and return a 400 error if it is supplied. This helper
omits `temperature` for those models so the same code works across every model.
"""

from langchain_anthropic import ChatAnthropic

# Models that reject the `temperature` parameter.
NO_TEMPERATURE_MODELS = {
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-fable-5",
}


def supports_temperature(model: str) -> bool:
    return model not in NO_TEMPERATURE_MODELS


def make_chat(model: str, api_key: str, temperature=None, max_tokens: int = 4096,
              timeout=None, max_retries=None, **kwargs) -> ChatAnthropic:
    """Build a ChatAnthropic client, omitting `temperature` when unsupported."""
    params = {"model": model, "max_tokens": max_tokens, "anthropic_api_key": api_key}
    if timeout is not None:
        params["default_request_timeout"] = timeout
    if max_retries is not None:
        params["max_retries"] = max_retries
    if temperature is not None and supports_temperature(model):
        params["temperature"] = temperature
    params.update(kwargs)
    return ChatAnthropic(**params)
