import re
from html import unescape
from typing import Any


TEXTUAL_NODE_TYPES = {
    "title",
    "paragraph",
    "list",
    "image",
    "chart",
    "table",
    "equation_interline",
    "code",
    "algorithm",
}

AUXILIARY_NODE_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "page_aside_text",
    "page_footnote",
}


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = unescape(text).replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def concatenate_spans(spans: Any) -> str:
    if not isinstance(spans, list):
        return normalize_text(spans)
    parts = []
    for span in spans:
        if not isinstance(span, dict):
            text = _span_text(span)
        elif span.get("type") == "equation_inline":
            text = f"${_span_text(span.get('content')).strip()}$"
        else:
            text = _span_text(span.get("content"))
        if text:
            parts.append(text)
    return normalize_text("".join(parts))


def _span_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = unescape(text).replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def block_text(block: dict[str, Any]) -> str:
    content = block.get("content") or {}
    block_type = block.get("type")
    if block_type == "title":
        return concatenate_spans(content.get("title_content"))
    if block_type == "paragraph":
        return concatenate_spans(content.get("paragraph_content"))
    if block_type == "list":
        return "\n".join(
            concatenate_spans(item.get("item_content"))
            for item in content.get("list_items", [])
            if concatenate_spans(item.get("item_content"))
        )
    if block_type == "image":
        return normalize_text("\n".join([
            normalize_text(content.get("content")),
            concatenate_spans(content.get("image_caption")),
            concatenate_spans(content.get("image_footnote")),
        ]))
    if block_type == "chart":
        return normalize_text("\n".join([
            normalize_text(content.get("content")),
            concatenate_spans(content.get("chart_caption")),
            concatenate_spans(content.get("chart_footnote")),
        ]))
    if block_type == "table":
        return normalize_text("\n".join([
            concatenate_spans(content.get("table_caption")),
            normalize_text(content.get("html")),
            concatenate_spans(content.get("table_footnote")),
        ]))
    if block_type == "equation_interline":
        return normalize_text(content.get("math_content"))
    if block_type == "code":
        return normalize_text("\n".join([
            concatenate_spans(content.get("code_caption")),
            concatenate_spans(content.get("code_content")),
        ]))
    if block_type == "algorithm":
        return normalize_text("\n".join([
            concatenate_spans(content.get("algorithm_caption")),
            concatenate_spans(content.get("algorithm_content")),
        ]))
    return normalize_text(content)


def short_abstract(text: str, *, max_chars: int = 300) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."

