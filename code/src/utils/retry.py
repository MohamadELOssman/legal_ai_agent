"""Small retry helper for LLM calls (transient connection/timeout resilience)."""

import time
from loguru import logger


def invoke_with_retry(llm, messages, retries: int = 4, base_delay: float = 2.0):
    """Invoke a LangChain chat model with retries + linear backoff.

    Guards against transient connection drops / timeouts so a single blip does
    not fail (and waste) a whole evaluation run. Re-raises after the last attempt.
    """
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"LLM call failed ({type(e).__name__}); "
                           f"retry {attempt + 1}/{retries - 1}")
            time.sleep(base_delay * (attempt + 1))
