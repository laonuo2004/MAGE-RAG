"""Helpers for LiteLLM/OpenAI-compatible chat completion calls."""

from openai import OpenAI

from utils.config_utils import require_config_value


def text_content_parts(text):
    """Return OpenAI content-parts for a text-only prompt."""
    return [{"type": "text", "text": "" if text is None else str(text)}]


def build_openai_client(cfg):
    """Build the project-wide OpenAI-compatible client from Hydra config."""
    return OpenAI(
        api_key=require_config_value(cfg, "litellm.api_key"),
        base_url=require_config_value(cfg, "litellm.base_url"),
    )


def completion_content(completion):
    """Return the first message content from a chat completion."""
    if isinstance(completion, str):
        return completion
    message = completion.choices[0].message
    content = message.content
    if content is None:
        raise ValueError("LLM response content is None")
    return content


def call_llm_messages(
    client,
    model_name,
    messages,
    *,
    max_tokens=1024,
    temperature=0.0,
    retries=3,
    logger=None,
    log_prefix="LLM call",
    failure_value=None,
):
    """Call an OpenAI-compatible chat completion endpoint with retries.

    ``failure_value`` may be a value or a callable that receives the last
    exception. This keeps benchmark-specific failure semantics out of the
    retry loop.
    """
    last_error = None
    for attempt in range(1, int(retries) + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            completion_content(completion)
            if logger is not None:
                logger.debug("Raw LLM completion: %s", completion)
            return completion
        except Exception as exc:
            last_error = exc
            if logger is not None:
                logger.warning(
                    "%s failed. model=%s attempt=%s/%s error=%s",
                    log_prefix,
                    model_name,
                    attempt,
                    retries,
                    exc,
                )

    if callable(failure_value):
        return failure_value(last_error)
    return failure_value
