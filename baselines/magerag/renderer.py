from __future__ import annotations

import os
import html

from baselines.image import VISION_SYSTEM_PROMPT
from baselines.magerag.state import PRUNED
from benchmarks.utils.document_preprocess import encode_image_file_to_base64, encode_pil_image_to_base64
from benchmarks.utils.data_utils import mmlongbench_png_page_path
from utils.config_utils import get_config_value, require_config_value
from utils.image_crop import crop_image_to_normalized_bbox_1000
from utils.llm_utils import xml_block


class ReaderRenderer:
    """
    Stage III answering 的输入渲染器。

    Online 阶段产出的是 evidence graph state；reader 需要的是 LVLM message content。
    这里负责把打开的文本证据、页面截图和可选 crop 排成最终回答上下文。
    """

    def __init__(self, cfg, include_page_images: bool = True, include_opened_node_images: bool = True):
        self.cfg = cfg
        self.include_page_images = bool(include_page_images)
        self.include_opened_node_images = bool(include_opened_node_images)
        self.not_answerable_text = str(
            get_config_value(cfg, "baselines.reader.not_answerable_text", "Not answerable.")
        )
        self.include_self_check_instruction = bool(
            get_config_value(cfg, "baselines.reader.include_self_check_instruction", True)
        )
        self.include_opened_node_crops = bool(
            get_config_value(cfg, "baselines.reader.include_opened_node_crops", True)
        )

    def render(self, benchmark_name: str, sample: dict, state) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": self._reader_prompt_open(sample["question"], state)}]
        image_refs = self._reader_image_refs(benchmark_name, sample, state)
        image_ref_by_key = {_image_ref_key(ref): ref for ref in image_refs}

        content.append({"type": "text", "text": "  <evidence>\n"})
        for page_index in self._reader_page_indices(state):
            page_ref = image_ref_by_key.get(("page", page_index, None))
            content.append({"type": "text", "text": self._page_evidence_open(state, page_index, page_ref)})
            if page_ref:
                image_part = self._page_image_part(benchmark_name, sample, page_index)
                if image_part is not None:
                    content.append(image_part)
            for node_id in self._reader_node_ids_for_page(state, page_index):
                node_ref = image_ref_by_key.get(("opened_node", page_index, node_id))
                content.append({"type": "text", "text": self._node_evidence_xml(state, node_id, node_ref)})
                if node_ref:
                    image_part = self._opened_node_image_part(benchmark_name, sample, state, node_id)
                    if image_part is not None:
                        content.append(image_part)
            content.append({"type": "text", "text": "    </page>\n"})
        content.append({"type": "text", "text": "  </evidence>\n</reader_prompt>"})
        if not any(isinstance(part, dict) and part.get("type") == "image_url" for part in content):
            return [{"type": "text", "text": "".join(str(part.get("text") or "") for part in content if isinstance(part, dict) and part.get("type") == "text")}]
        return content

    def trace_input(self, benchmark_name: str, sample: dict, state, content: list[dict]) -> dict:
        text_parts = [
            str(part.get("text") or "")
            for part in _iter_content_parts(content)
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        image_refs = self._reader_image_refs(benchmark_name, sample, state)
        return {
            "messages": _sanitize_messages([{"role": "user", "content": content}], image_refs),
            "text_parts": text_parts,
            "prompt_text": "".join(text_parts),
            "image_refs": image_refs,
            "content_part_count": len(list(_iter_content_parts(content))),
        }

    def _text_context(self, benchmark_name: str, question: str, state) -> str:
        return self._reader_prompt_open(question, state) + "  <evidence>\n  </evidence>\n</reader_prompt>"

    def _full_text_context(self, question: str, state) -> str:
        return self._text_context("", question, state)

    def _compact_text_context(self, benchmark_name: str, question: str, state) -> str:
        return self._text_context(benchmark_name, question, state)

    def _reader_prompt_open(self, question: str, state) -> str:
        page_indices = self._reader_page_indices(state)
        lines = [
            "<reader_prompt>",
            xml_block("role", VISION_SYSTEM_PROMPT.strip(), escape=True, inline=True, indent=2),
            xml_block("objective", "Answer the question using only the provided document evidence.", escape=True, inline=True, indent=2),
            xml_block("question", question, escape=True, inline=True, indent=2),
            xml_block("retrieved_page_indices", ", ".join(str(page_index) for page_index in page_indices), escape=True, inline=True, indent=2),
            xml_block(
                "evidence_policy",
                "Use the structured evidence XML and the interleaved document images together.\n"
                "Treat page, table, chart, figure, and image screenshots as primary evidence when they are provided.\n"
                "Use page and element abstracts as concise evidence summaries.\n"
                "If visual evidence conflicts with extracted text, trust the visible document content.\n"
                "Do not use outside knowledge or infer facts not supported by provided evidence.",
                escape=True,
                indent=2,
            ),
            xml_block(
                "answer_policy",
                f"If the answer cannot be found, answer exactly: {self.not_answerable_text}\n"
                "Do not answer with None, null, [], or an empty string.\n"
                "Do not answer with evidence node ids, page ids, block ids, or bracketed provenance labels.\n"
                "If the question asks for multiple items, include every item supported by the evidence.\n"
                "If the evidence gives a full title, table name, figure name, person name, organization name, or other named answer, preserve the complete supported name rather than shortening it.",
                escape=True,
                indent=2,
            ),
            xml_block(
                "thinking_policy",
                "Before answering, output one <think>...</think> block.\n"
                "Use think to compare the question against page abstracts, node abstracts, and interleaved images.\n"
                "Verify that the final answer is directly supported by at least one evidence item.\n"
                "If support is missing or ambiguous, choose Not answerable.",
                escape=True,
                indent=2,
            ),
            xml_block(
                "output_schema",
                "Return exactly one <think> block followed by exactly one <answer> block.\n"
                "The <answer> block must contain only the final answer text.\n"
                "Output sequence:\n<think>...</think>\n<answer>[final_answer]</answer>\n"
                "No other text outside these two XML blocks.",
                escape=True,
                indent=2,
            ),
        ]
        if self.include_self_check_instruction:
            lines.append(xml_block(
                "self_check",
                "Before answering, verify that the answer is directly supported by the provided image or text evidence and is not merely copied from the question.",
                escape=True,
                indent=2,
            ))
        lines.append(self._reader_few_shot_examples_xml())
        return "\n".join(lines) + "\n"

    def _reader_few_shot_examples_xml(self) -> str:
        return "\n".join([
            "  <few_shot_examples>",
            "    <example>",
            "      <problem>Answer a directly supported author question.</problem>",
            "      <example_input><question>Who was the principal author of the report?</question><evidence><page index=\"5\"><abstract>Consultant qualifications identify the principal author.</abstract><node type=\"paragraph\"><abstract>Principal author is Franklin Maggi, Architectural Historian.</abstract></node></page></evidence></example_input>",
            "      <example_output><think>The paragraph abstract directly states the principal author and gives the full name.</think><answer>Franklin Maggi</answer></example_output>",
            "    </example>",
            "    <example>",
            "      <problem>Return Not answerable when evidence does not support the requested fact.</problem>",
            "      <example_input><question>What was the final approval date?</question><evidence><page index=\"1\"><abstract>Project overview without approval date.</abstract><node type=\"paragraph\"><abstract>The project includes residential units and commercial space.</abstract></node></page></evidence></example_input>",
            f"      <example_output><think>The evidence describes the project but does not state a final approval date.</think><answer>{_esc_xml(self.not_answerable_text)}</answer></example_output>",
            "    </example>",
            "    <example>",
            "      <problem>Preserve complete table or figure names.</problem>",
            "      <example_input><question>What is the name of the table?</question><evidence><page index=\"13\"><abstract>A page containing a full table caption.</abstract><node type=\"table\"><abstract>Table 15: Leading destination of exports (UGX Billion): July-June</abstract></node></page></evidence></example_input>",
            "      <example_output><think>The table abstract includes the complete table caption, so the answer should not be shortened to only the table number.</think><answer>Table 15: Leading destination of exports (UGX Billion): July-June</answer></example_output>",
            "    </example>",
            "  </few_shot_examples>",
        ])

    def _page_evidence_open(self, state, page_index: int, image_ref: dict | None) -> str:
        page_node_id = state.graph.page_node_id(page_index)
        page = state.graph.node(page_node_id) if page_node_id in state.graph.nodes else {}
        parts = [f'    <page index="{page_index}">']
        abstract = str(page.get("abstract") or "")
        if abstract:
            parts.append(xml_block("abstract", abstract, escape=True, inline=True, indent=6))
        if image_ref:
            parts.append("      " + _empty_xml("image", {"ref": image_ref.get("ref"), "kind": "page"}))
        return "\n".join(parts) + "\n"

    def _node_evidence_xml(self, state, node_id: str, image_ref: dict | None) -> str:
        node = state.graph.node(node_id)
        parts = [f'      <node type="{_esc_xml(node.get("type", ""))}">']
        abstract = str(node.get("abstract") or node.get("caption") or node.get("title") or "")
        if abstract:
            parts.append(xml_block("abstract", abstract, escape=True, inline=True, indent=8))
        if node.get("bbox") is not None:
            parts.append(xml_block("bbox", str(node.get("bbox")), escape=True, inline=True, indent=8))
        if image_ref:
            parts.append("        " + _empty_xml("image", {"ref": image_ref.get("ref"), "kind": image_ref.get("kind")}))
        parts.append("      </node>")
        return "\n".join(parts) + "\n"

    def _reader_node_ids_for_page(self, state, page_index: int) -> list[str]:
        if self._page_is_pruned(state, page_index):
            return []
        node_ids = []
        for node_id in state.opened_node_ids() + state.active_node_ids():
            if node_id in node_ids or node_id in state.pruned_node_ids():
                continue
            node = state.graph.node(node_id)
            if state.graph.is_page_node(node):
                continue
            if state.graph.node_page_index(node) != page_index:
                continue
            node_ids.append(node_id)
        return node_ids

    def _reader_page_indices(self, state) -> list[int]:
        # 显式题目页优先于检索页；其他页面按激活顺序补充，保持“人类阅读路径”的顺序。
        prioritized = []
        deferred = []
        seen = set()
        for item in state.trace:
            if item.get("action") != "ActivatePage" or not item.get("ok"):
                continue
            payload = item.get("payload") or {}
            if "page_index" not in payload:
                continue
            page_index = int(payload["page_index"])
            if self._page_is_pruned(state, page_index):
                continue
            if page_index in seen:
                continue
            seen.add(page_index)
            if payload.get("source") == "question_page_scope":
                prioritized.append(page_index)
            else:
                deferred.append(page_index)
        page_indices = prioritized + deferred
        for node_id in state.active_node_ids() + state.opened_node_ids():
            if node_id in state.pruned_node_ids():
                continue
            page_index = state.graph.node_page_index(node_id)
            if self._page_is_pruned(state, page_index):
                continue
            if page_index in seen:
                continue
            seen.add(page_index)
            page_indices.append(page_index)
        return page_indices

    def _page_is_pruned(self, state, page_index: int) -> bool:
        page_node_id = state.graph.page_node_id(int(page_index))
        return state.state_of(page_node_id) == PRUNED

    def _page_image_part(self, benchmark_name: str, sample: dict, page_index: int):
        if benchmark_name == "mmlongbench":
            from PIL import Image

            benchmark_cfg = require_config_value(self.cfg, "benchmarks")
            path = self._page_image_path(benchmark_name, sample, page_index)
            if not _image_file_exists(path):
                return None
            with Image.open(path) as image:
                encoded = encode_pil_image_to_base64(image)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        image_path = self._page_image_path(benchmark_name, sample, page_index)
        if not _image_file_exists(image_path):
            return None
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_file_to_base64(image_path)}"}}

    def _page_image_path(self, benchmark_name: str, sample: dict, page_index: int) -> str:
        if benchmark_name == "mmlongbench":
            benchmark_cfg = require_config_value(self.cfg, "benchmarks")
            return mmlongbench_png_page_path(benchmark_cfg, sample["doc_id"], page_index)
        return os.path.join(
            str(require_config_value(self.cfg, "benchmarks.image_prefix")),
            sample["doc_no"][:4],
            f'{sample["doc_no"]}_{page_index}.png',
        )

    def _reader_image_refs(self, benchmark_name: str, sample: dict, state) -> list[dict]:
        refs = []
        page_indices = self._reader_page_indices(state)
        node_image_ids = set(self._candidate_node_image_ids(state))
        for page_index in page_indices:
            if self.include_page_images:
                image_path = self._page_image_path(benchmark_name, sample, page_index)
                if _image_file_exists(image_path):
                    refs.append({
                        "kind": "page",
                        "ref": f"page_image_{len(refs) + 1}",
                        "image_index": len(refs),
                        "page_index": page_index,
                        "image_path": image_path,
                        "label": "",
                        "mime_type": _image_mime_type(image_path),
                    })
            for node_id in self._reader_node_ids_for_page(state, page_index):
                if node_id not in node_image_ids:
                    continue
                node = state.graph.node(node_id)
                image_path = self._node_image_path(node)
                if _image_file_exists(image_path):
                    refs.append({
                        "kind": "opened_node_image",
                        "ref": f"node_image_{len(refs) + 1}",
                        "image_index": len(refs),
                        "node_id": node_id,
                        "page_index": page_index,
                        "image_path": str(image_path),
                        "mime_type": _image_mime_type(str(image_path)),
                    })
                elif self.include_opened_node_crops and self._can_crop_node(node):
                    page_image_path = self._page_image_path(benchmark_name, sample, page_index)
                    if _image_file_exists(page_image_path):
                        refs.append({
                            "kind": "opened_node_crop",
                            "ref": f"node_image_{len(refs) + 1}",
                            "image_index": len(refs),
                            "node_id": node_id,
                            "page_index": page_index,
                            "image_path": page_image_path,
                            "bbox": node.get("bbox"),
                            "bbox_coordinate_system": "normalized_1000",
                            "mime_type": "image/jpeg",
                        })
        return refs

    def _candidate_node_image_ids(self, state) -> list[str]:
        if not self.include_opened_node_images:
            return []
        node_ids = []
        for page_index in self._reader_page_indices(state):
            for node_id in self._reader_node_ids_for_page(state, page_index):
                node = state.graph.node(node_id)
                if self._node_image_path(node) or (self.include_opened_node_crops and self._can_crop_node(node)):
                    node_ids.append(node_id)
        return node_ids

    def _opened_node_image_part(self, benchmark_name: str, sample: dict, state, node_id: str):
        node = state.graph.node(node_id)
        image_path = self._node_image_path(node)
        if _image_file_exists(image_path):
            return {"type": "image_url", "image_url": {"url": f"data:{_image_mime_type(str(image_path))};base64,{encode_image_file_to_base64(str(image_path))}"}}
        return self._opened_node_crop_part(benchmark_name, sample, state, node_id)

    def _node_image_path(self, node: dict) -> str | None:
        value = node.get("image_path") or node.get("crop_path")
        if not value and isinstance(node.get("metadata"), dict):
            value = node["metadata"].get("image_path") or node["metadata"].get("crop_path")
        return str(value) if value else None

    def _can_crop_node(self, node: dict) -> bool:
        bbox = node.get("bbox")
        return isinstance(bbox, (list, tuple)) and len(bbox) == 4


    def _opened_node_crop_part(self, benchmark_name: str, sample: dict, state, node_id: str):
        from PIL import Image

        node = state.graph.node(node_id)
        bbox = node.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        page_index = state.graph.node_page_index(node)
        path = self._page_image_path(benchmark_name, sample, page_index)
        if not _image_file_exists(path):
            return None
        with Image.open(path) as image:
            crop = crop_image_to_normalized_bbox_1000(image, bbox)
            if crop is None:
                return None
            encoded = encode_pil_image_to_base64(crop)
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def _empty_xml(tag: str, attrs: dict) -> str:
    attr_text = " ".join(f'{_esc_xml(key)}="{_esc_xml(value)}"' for key, value in attrs.items() if value is not None)
    if attr_text:
        return f"<{tag} {attr_text}/>"
    return f"<{tag}/>"


def _image_ref_key(ref: dict):
    if ref.get("kind") == "page":
        return ("page", ref.get("page_index"), None)
    return ("opened_node", ref.get("page_index"), ref.get("node_id"))


def _iter_content_parts(content):
    if isinstance(content, list):
        for part in content:
            yield part
    elif isinstance(content, dict):
        yield content


def _sanitize_messages(messages: list[dict], image_refs: list[dict]) -> list[dict]:
    image_iter = iter(image_refs)
    sanitized = []
    for message in messages:
        content = message.get("content")
        clean_parts = []
        for part in _iter_content_parts(content):
            if not isinstance(part, dict):
                clean_parts.append(part)
                continue
            if part.get("type") != "image_url":
                clean_parts.append(dict(part))
                continue
            ref = next(image_iter, None)
            clean_parts.append({
                "type": "image_ref",
                "image_ref": ref or {},
            })
        sanitized.append({
            "role": message.get("role"),
            "content": clean_parts,
        })
    return sanitized


def _image_mime_type(path: str) -> str:
    suffix = os.path.splitext(str(path).lower())[1]
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _image_file_exists(path) -> bool:
    return bool(path) and os.path.isfile(str(path))


def _esc_xml(value) -> str:
    return html.escape(str(value or ""), quote=True)
