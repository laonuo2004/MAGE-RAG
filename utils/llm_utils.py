"""Helpers for LiteLLM/OpenAI-compatible chat completion calls."""

import html
import re
import time
from typing import Any, Mapping

from openai import OpenAI

from utils.config_utils import require_config_value

XML_TAG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")


def text_content_parts(text):
    """Return OpenAI content-parts for a text-only prompt."""
    return [{"type": "text", "text": "" if text is None else str(text)}]


def _xml_text(value: Any) -> str:
    return "" if value is None else str(value)


def _xml_attribute_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return _xml_text(value)


def _format_xml_attributes(attributes: Mapping[str, Any] | None) -> str:
    if not attributes:
        return ""
    rendered = []
    for key, value in attributes.items():
        if value is None:
            continue
        if not XML_TAG_RE.match(str(key)):
            raise ValueError(f"Invalid XML attribute name: {key}")
        escaped_value = html.escape(_xml_attribute_text(value), quote=True)
        rendered.append(f'{key}="{escaped_value}"')
    return "" if not rendered else " " + " ".join(rendered)


def _indent_xml_text(text: str, indent: int) -> str:
    if indent <= 0:
        return text
    prefix = " " * indent
    return "\n".join(f"{prefix}{line}" for line in text.split("\n"))


def _cdata_text(text: str) -> str:
    return f"<![CDATA[{text.replace(']]>', ']]]]><![CDATA[>')}]]>"


def xml_block(
    tag: str,
    *values: Any,
    attributes: Mapping[str, Any] | None = None,
    template: str | None = None,
    template_kwargs: Mapping[str, Any] | None = None,
    escape: bool = False,
    cdata: bool = False,
    inline: bool = False,
    indent: int = 0,
) -> str:
    """Render a small XML-like prompt block.

    The default output intentionally stays simple and prompt-friendly:
    ``<tag>\nvalue\n</tag>``. Optional arguments cover common prompt-building
    needs without forcing callers into manual string interpolation.
    """
    if not XML_TAG_RE.match(str(tag)):
        raise ValueError(f"Invalid XML tag name: {tag}")
    if cdata and escape:
        raise ValueError("cdata and escape cannot both be enabled")
    if indent < 0:
        raise ValueError("indent must be non-negative")

    string_values = tuple(_xml_text(value) for value in values)
    if template is not None:
        named_values = {key: _xml_text(value) for key, value in (template_kwargs or {}).items()}
        content = template.format(*string_values, **named_values)
    elif not string_values:
        content = ""
    elif len(string_values) == 1:
        content = string_values[0]
    else:
        content = "\n".join(string_values)

    if cdata:
        content = _cdata_text(content)
    elif escape:
        content = html.escape(content, quote=False)

    attr_text = _format_xml_attributes(attributes)
    prefix = " " * indent
    open_tag = f"{prefix}<{tag}{attr_text}>"
    close_tag = f"{prefix}</{tag}>"
    if inline:
        return f"{open_tag}{content}{close_tag.lstrip()}"
    return f"{open_tag}\n{_indent_xml_text(content, indent)}\n{close_tag}"


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
    total_attempts = int(retries)
    last_error = None
    for attempt in range(1, total_attempts + 1):
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
            if attempt == total_attempts:
                if logger is not None:
                    logger.warning(
                        "%s failed. model=%s attempt=%s/%s error=%s",
                        log_prefix,
                        model_name,
                        attempt,
                        retries,
                        exc,
                    )
                continue

            delay = 5 * (2 ** (attempt - 1))
            if logger is not None:
                logger.warning(
                    "%s failed. model=%s attempt=%s/%s retry_in=%ss error=%s",
                    log_prefix,
                    model_name,
                    attempt,
                    retries,
                    delay,
                    exc,
                )
            time.sleep(delay)

    if callable(failure_value):
        return failure_value(last_error)
    return failure_value
