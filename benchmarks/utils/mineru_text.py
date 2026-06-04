import json
import logging
from pathlib import Path
from typing import Any

from benchmarks.evidence_graph.content import block_text, concatenate_spans, normalize_text
from benchmarks.evidence_graph.mineru_loader import load_mineru_document
from utils.llm_utils import call_llm_messages, completion_content

logger = logging.getLogger(__name__)

VISUAL_BLOCK_TYPES = {"image", "chart", "table", "equation_interline"}

VISUAL_DESCRIPTION_PROMPT = (
    "Describe the document visual element using only the visible image and extracted local text. "
    "Focus on facts that help document question answering: labels, values, trends, structure, captions, and footnotes. "
    "If the visual evidence is unavailable, summarize only the extracted local text. Return concise plain text."
)


def build_mineru_text_pages(
    *,
    mineru_dir: str | Path,
    output_path: str | Path | None = None,
    client=None,
    model_name: str | None = None,
    overwrite: bool = False,
    max_tokens: int = 512,
) -> dict[str, Any]:
    mineru_dir = Path(mineru_dir)
    output_path = Path(output_path) if output_path is not None else mineru_dir / "vlm_text_pages.json"
    if output_path.exists() and not overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    pages, content_pages, content_path = load_mineru_document(mineru_dir)
    page_payloads = []
    for page_index, blocks in enumerate(content_pages):
        page_payloads.append(
            {
                "page_index": page_index,
                "page_number": page_index + 1,
                "text": build_page_text(
                    blocks if isinstance(blocks, list) else [],
                    mineru_dir=mineru_dir,
                    client=client,
                    model_name=model_name,
                    max_tokens=max_tokens,
                ),
            }
        )

    artifact = {
        "source": {
            "mineru_dir": str(mineru_dir),
            "content_list_v2": str(content_path),
            "page_count": len(pages),
        },
        "pages": page_payloads,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    logger.info("Saved MinerU VLM text pages to %s", output_path)
    return artifact


def build_page_text(blocks: list[dict[str, Any]], *, mineru_dir: Path, client=None, model_name=None, max_tokens=512) -> str:
    parts = []
    for block_index, block in enumerate(blocks):
        text = build_block_text(
            block,
            mineru_dir=mineru_dir,
            client=client,
            model_name=model_name,
            max_tokens=max_tokens,
        )
        if text:
            block_type = block.get("type") or "block"
            parts.append(f"[{block_type} {block_index}]\n{text}")
    return normalize_text("\n\n".join(parts)) or "[EMPTY PAGE]"


def build_block_text(block: dict[str, Any], *, mineru_dir: Path, client=None, model_name=None, max_tokens=512) -> str:
    block_type = block.get("type")
    extracted_text = normalize_text(block_text(block))
    if block_type not in VISUAL_BLOCK_TYPES:
        return extracted_text

    llm_description = ""
    if client is not None and model_name:
        image_path = _block_image_path(block, mineru_dir)
        llm_description = describe_visual_block(
            client=client,
            model_name=model_name,
            block_type=str(block_type),
            extracted_text=extracted_text,
            image_path=image_path,
            max_tokens=max_tokens,
        )

    fields = _visual_text_fields(block)
    sections = [f"{block_type}"]
    if fields:
        sections.append(fields)
    if llm_description:
        sections.append(f"Visual description: {llm_description}")
    elif extracted_text:
        sections.append(extracted_text)
    return normalize_text("\n".join(sections))


def describe_visual_block(*, client, model_name: str, block_type: str, extracted_text: str, image_path: str, max_tokens: int = 512) -> str:
    content = []
    if image_path and Path(image_path).exists():
        from benchmarks.utils.document_preprocess import encode_image_file_to_base64

        suffix = Path(image_path).suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{encode_image_file_to_base64(image_path)}"},
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"{VISUAL_DESCRIPTION_PROMPT}\n\n"
                f"Element type: {block_type}\n"
                f"Extracted local text:\n{extracted_text or '[NO EXTRACTED TEXT]'}"
            ),
        }
    )
    completion = call_llm_messages(
        client,
        model_name,
        [{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0.0,
        logger=logger,
        log_prefix="MinerU visual block description",
        failure_value="",
    )
    if not completion:
        return ""
    return normalize_text(completion_content(completion))


def _visual_text_fields(block: dict[str, Any]) -> str:
    content = block.get("content") or {}
    block_type = block.get("type")
    if block_type == "image":
        return normalize_text("\n".join([concatenate_spans(content.get("image_caption")), concatenate_spans(content.get("image_footnote"))]))
    if block_type == "chart":
        return normalize_text("\n".join([concatenate_spans(content.get("chart_caption")), concatenate_spans(content.get("chart_footnote"))]))
    if block_type == "table":
        return normalize_text(
            "\n".join(
                [
                    concatenate_spans(content.get("table_caption")),
                    normalize_text(content.get("html")),
                    concatenate_spans(content.get("table_footnote")),
                ]
            )
        )
    if block_type == "equation_interline":
        return normalize_text(content.get("math_content"))
    return ""


def _block_image_path(block: dict[str, Any], mineru_dir: Path) -> str:
    source = (block.get("content") or {}).get("image_source") or {}
    raw_path = str(source.get("path") or "")
    if not raw_path:
        return ""
    path = Path(raw_path)
    return str(path if path.is_absolute() else mineru_dir / path)
