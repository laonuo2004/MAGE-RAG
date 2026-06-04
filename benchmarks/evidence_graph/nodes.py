from pathlib import Path
from typing import Any

from benchmarks.evidence_graph.content import (
    AUXILIARY_NODE_TYPES,
    block_text,
    concatenate_spans,
    short_abstract,
)
from benchmarks.evidence_graph.schema import EvidenceNode


def build_nodes(
    *,
    doc_id: str,
    doc_key: str,
    pages: list[dict[str, Any]],
    content_pages: list[list[dict[str, Any]]],
    page_image_paths: dict[int, Path],
    page_embedding_path: Path,
    node_embedding_dir: Path,
    mineru_dir: Path,
    skip_llm: bool,
) -> list[EvidenceNode]:
    nodes: list[EvidenceNode] = []
    for page in pages:
        page_index = int(page.get("page_idx"))
        nodes.append(
            EvidenceNode(
                id=f"{doc_key}:page:{page_index}",
                type="page",
                doc_id=doc_id,
                page_index=page_index,
                abstract=_page_abstract(content_pages, page_index, skip_llm),
                embedding_path=str(page_embedding_path),
                fields={
                    "page_size": page.get("page_size"),
                    "image_path": str(page_image_paths.get(page_index, "")),
                },
            )
        )

    for page_index, blocks in enumerate(content_pages):
        if not isinstance(blocks, list):
            continue
        element_index = 0
        for block_index, block in enumerate(blocks):
            block_type = block.get("type")
            if block_type in AUXILIARY_NODE_TYPES:
                continue
            node = _element_node(
                doc_id=doc_id,
                doc_key=doc_key,
                page_index=page_index,
                block_index=block_index,
                element_index=element_index,
                block=block,
                node_embedding_dir=node_embedding_dir,
                mineru_dir=mineru_dir,
                skip_llm=skip_llm,
            )
            if node is not None:
                nodes.append(node)
                element_index += 1
    return nodes


def _page_abstract(content_pages: list[list[dict[str, Any]]], page_index: int, skip_llm: bool) -> str:
    if page_index >= len(content_pages):
        return ""
    text = "\n".join(block_text(block) for block in content_pages[page_index] if block.get("type") not in AUXILIARY_NODE_TYPES)
    return short_abstract(text)


def _element_node(
    *,
    doc_id: str,
    doc_key: str,
    page_index: int,
    block_index: int,
    element_index: int,
    block: dict[str, Any],
    node_embedding_dir: Path,
    mineru_dir: Path,
    skip_llm: bool,
) -> EvidenceNode | None:
    block_type = block.get("type")
    if not block_type:
        return None
    content = block.get("content") or {}
    text = block_text(block)
    node_id = f"{doc_key}:page:{page_index}:block:{block_index}:{block_type}"
    fields = _type_fields(block_type, content, mineru_dir)
    return EvidenceNode(
        id=node_id,
        type=str(block_type),
        doc_id=doc_id,
        page_index=page_index,
        index=element_index,
        bbox=block.get("bbox"),
        abstract=short_abstract(text),
        embedding_path=str(node_embedding_dir / f"page_{page_index}_block_{block_index}_{block_type}.safetensors"),
        fields=fields,
        metadata={"source_block_index": block_index, "source_text": text},
    )


def _image_path(content: dict[str, Any], mineru_dir: Path) -> str:
    source = content.get("image_source") or {}
    raw_path = str(source.get("path") or "")
    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str(mineru_dir / path)


def _type_fields(block_type: str, content: dict[str, Any], mineru_dir: Path) -> dict[str, Any]:
    if block_type == "title":
        return {"title": concatenate_spans(content.get("title_content")), "level": int(content.get("level") or 1)}
    if block_type == "paragraph":
        return {"paragraph": concatenate_spans(content.get("paragraph_content"))}
    if block_type == "list":
        return {
            "list_type": str(content.get("list_type") or ""),
            "items": [concatenate_spans(item.get("item_content")) for item in content.get("list_items", [])],
        }
    if block_type == "image":
        return {
            "image_path": _image_path(content, mineru_dir),
            "content": str(content.get("content") or ""),
            "caption": concatenate_spans(content.get("image_caption")),
            "footnote": concatenate_spans(content.get("image_footnote")),
        }
    if block_type == "chart":
        return {
            "image_path": _image_path(content, mineru_dir),
            "content": str(content.get("content") or ""),
            "caption": concatenate_spans(content.get("chart_caption")),
            "footnote": concatenate_spans(content.get("chart_footnote")),
        }
    if block_type == "table":
        return {
            "image_path": _image_path(content, mineru_dir),
            "html": str(content.get("html") or ""),
            "table_type": str(content.get("table_type") or ""),
            "table_nest_level": int(content.get("table_nest_level") or 0),
            "caption": concatenate_spans(content.get("table_caption")),
            "footnote": concatenate_spans(content.get("table_footnote")),
        }
    if block_type == "equation_interline":
        return {
            "latex": str(content.get("math_content") or ""),
            "math_type": str(content.get("math_type") or ""),
            "image_path": _image_path(content, mineru_dir),
        }
    if block_type == "code":
        return {
            "language": str(content.get("code_language") or ""),
            "code": concatenate_spans(content.get("code_content")),
            "caption": concatenate_spans(content.get("code_caption")),
        }
    if block_type == "algorithm":
        return {
            "algorithm": concatenate_spans(content.get("algorithm_content")),
            "caption": concatenate_spans(content.get("algorithm_caption")),
        }
    return {}

