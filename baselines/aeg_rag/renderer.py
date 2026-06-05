from __future__ import annotations

import os
import re

from baselines.image import VISION_SYSTEM_PROMPT
from benchmarks.utils.document_preprocess import encode_image_file_to_base64, encode_pil_image_to_base64
from benchmarks.utils.data_utils import mmlongbench_png_page_path
from utils.config_utils import get_config_value, require_config_value


class ReaderRenderer:
    def __init__(self, cfg, include_page_images: bool = True, include_opened_node_images: bool = True, raw_text_limit: int = 8192):
        self.cfg = cfg
        self.include_page_images = bool(include_page_images)
        self.include_opened_node_images = bool(include_opened_node_images)
        self.raw_text_limit = int(raw_text_limit)
        self.reader_text_mode = str(get_config_value(cfg, "baselines.renderer.reader_text_mode", "full"))
        self.mmlongbench_page_text_char_limit = int(
            get_config_value(cfg, "baselines.renderer.mmlongbench_page_text_char_limit", 1200)
        )
        self.mmlongbench_page_text_max_pages = int(
            get_config_value(cfg, "baselines.renderer.mmlongbench_page_text_max_pages", 3)
        )
        self.mmlongbench_prompt_mode = str(
            get_config_value(cfg, "baselines.renderer.mmlongbench_prompt_mode", "format")
        )
        self.mmlongbench_include_image_page_labels = bool(
            get_config_value(cfg, "baselines.renderer.mmlongbench_include_image_page_labels", False)
        )
        self.include_opened_node_text = bool(
            get_config_value(cfg, "baselines.renderer.include_opened_node_text", False)
        )
        self.include_opened_node_text_longdocurl = bool(
            get_config_value(cfg, "baselines.renderer.include_opened_node_text_longdocurl", self.include_opened_node_text)
        )
        self.include_opened_node_text_mmlongbench = bool(
            get_config_value(cfg, "baselines.renderer.include_opened_node_text_mmlongbench", self.include_opened_node_text)
        )
        self.opened_node_text_char_limit = int(
            get_config_value(cfg, "baselines.renderer.opened_node_text_char_limit", 1200)
        )
        self.mmlongbench_include_opened_node_crops = bool(
            get_config_value(cfg, "baselines.renderer.mmlongbench_include_opened_node_crops", False)
        )
        self.mmlongbench_max_opened_node_crops = int(
            get_config_value(cfg, "baselines.renderer.mmlongbench_max_opened_node_crops", 4)
        )

    def render(self, benchmark_name: str, sample: dict, state) -> list[dict]:
        text = self._text_context(benchmark_name, sample["question"], state)
        content = [{"type": "text", "text": text}]
        if self.include_page_images:
            page_indices = self._reader_page_indices(state)
            for image_index, page_index in enumerate(page_indices):
                image_part = self._page_image_part(benchmark_name, sample, page_index)
                if image_part is not None:
                    label = self._image_label(benchmark_name, image_index, len(page_indices), page_index)
                    if label:
                        content.append({"type": "text", "text": label})
                    content.append(image_part)
        if benchmark_name == "mmlongbench" and self.mmlongbench_include_opened_node_crops:
            for crop_index, node_id in enumerate(self._candidate_crop_node_ids(sample["question"], state)):
                crop_part = self._opened_node_crop_part(benchmark_name, sample, state, node_id)
                if crop_part is None:
                    continue
                node = state.graph.node(node_id)
                page_number = state.graph.node_page_index(node) + 1
                node_type = str(node.get("type") or "evidence")
                content.append({"type": "text", "text": f"Relevant {node_type} crop {crop_index + 1} from document page {page_number}:\n"})
                content.append(crop_part)
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
            "image_refs": image_refs,
            "content_part_count": len(list(_iter_content_parts(content))),
        }

    def _text_context(self, benchmark_name: str, question: str, state) -> str:
        if self.reader_text_mode == "compact":
            return self._compact_text_context(benchmark_name, question, state)
        return self._full_text_context(question, state)

    def _full_text_context(self, question: str, state) -> str:
        lines = [
            VISION_SYSTEM_PROMPT.strip(),
            f"Question: {question}",
            "",
            "Answer using the document content only. Evidence node ids are for provenance only.",
            "Do not answer with evidence node ids, page ids, block ids, or bracketed provenance labels.",
        ]
        if self._is_locating_question(question):
            lines.extend([
                "For title/table locating questions, return the exact visible title or table name from the document.",
                "Do not return an evidence identifier such as 4106951:page:72:block:0:title.",
            ])
        candidate_strings = self._candidate_answer_strings(state) if self._is_locating_question(question) else []
        if candidate_strings:
            lines.extend(["", "Candidate answer strings from opened evidence:"])
            for value in candidate_strings:
                lines.append(f"- {value}")
        lines.extend(["", "Active evidence graph:"])
        for node_id in state.active_node_ids():
            node = state.graph.node(node_id)
            lines.append(
                f"- provenance_id={node_id} | type={node.get('type')} | page={state.graph.node_page_index(node) + 1} | "
                f"abstract={node.get('abstract', '')}"
            )
        lines.append("")
        lines.append("Opened evidence:")
        for node_id in state.opened_node_ids():
            node = state.graph.node(node_id)
            raw_text = state.graph.node_text(node_id)[: self.raw_text_limit]
            lines.append(
                f"Evidence item: provenance_id={node_id} type={node.get('type')} "
                f"page={state.graph.node_page_index(node) + 1} bbox={node.get('bbox')}\n{raw_text}"
            )
        if state.summaries:
            lines.append("")
            lines.append("Online summaries:")
            for summary in state.summaries:
                lines.append(
                    f"- {summary['summary_id']} sources={','.join(summary.get('source_node_ids') or [])}: "
                    f"{summary.get('text', '')}"
                )
        return "\n".join(lines)

    def _compact_text_context(self, benchmark_name: str, question: str, state) -> str:
        if benchmark_name == "mmlongbench":
            page_numbers = [page_index + 1 for page_index in self._reader_page_indices(state)]
            lines = [str(question)]
            if self.mmlongbench_prompt_mode != "plain" and page_numbers:
                lines.append(f"Retrieved document pages: {', '.join(str(page_number) for page_number in page_numbers)}.")
            if self.mmlongbench_prompt_mode != "plain":
                lines.extend([
                    "",
                    "Answer using the document images first. Use retrieved text only as OCR hints, and ignore it if it conflicts with visible evidence.",
                    "If the answer cannot be found, answer exactly: Not answerable.",
                    "Do not answer with None, null, [], or an empty string.",
                    "Return only the final answer. For list questions, return a JSON-style list of answer strings.",
                    "For color questions, use common color names rather than hex codes.",
                ])
            elif self._is_color_question(question):
                lines.extend([
                    "",
                    "For color questions, use common color names rather than hex codes.",
                ])
            domain_hints = self._mmlongbench_domain_hints(question)
            if domain_hints:
                lines.extend(["", *domain_hints])
            page_snippets = self._mmlongbench_page_text_snippets(state)
            if page_snippets:
                lines.extend(["", "Retrieved page text snippets:"])
                lines.extend(page_snippets)
        else:
            lines = [
                VISION_SYSTEM_PROMPT
                + "Following is our question: \n"
                + f"<question>{question}</question>"
                + "\n"
            ]
        candidate_strings = self._candidate_answer_strings(state) if self._is_locating_question(question) else []
        if candidate_strings:
            lines.extend(["", "Candidate visible labels from retrieved evidence:"])
            for value in candidate_strings[:40]:
                lines.append(f"- {value}")
            lines.append("Use these labels only when they match the visible document content.")
        opened_snippets = self._opened_node_text_snippets(state) if self._include_opened_text_for(benchmark_name) else []
        if opened_snippets:
            lines.extend(["", "Opened evidence text:"])
            lines.extend(opened_snippets)
        return "\n".join(lines)

    def _include_opened_text_for(self, benchmark_name: str) -> bool:
        if benchmark_name == "longdocurl":
            return self.include_opened_node_text_longdocurl
        if benchmark_name == "mmlongbench":
            return self.include_opened_node_text_mmlongbench
        return self.include_opened_node_text

    def _image_label(self, benchmark_name: str, image_index: int, total_images: int, page_index: int) -> str:
        if self.reader_text_mode == "compact":
            if benchmark_name == "mmlongbench":
                if not self.mmlongbench_include_image_page_labels:
                    return ""
                return f"Document page {page_index + 1}:\n"
            return f"Below is the {image_index + 1}-th image (total {total_images} images).\n"
        return (
            f"Below is retrieved page image {image_index + 1} of {total_images} "
            f"(document page {page_index + 1}).\n"
        )

    def _mmlongbench_page_text_snippets(self, state) -> list[str]:
        snippets = []
        for page_index in self._reader_page_indices(state)[: self.mmlongbench_page_text_max_pages]:
            page_node_id = state.graph.page_node_id(page_index)
            if page_node_id not in state.graph.nodes:
                continue
            text = state.graph.node_text(page_node_id) or str(state.graph.node(page_node_id).get("abstract") or "")
            text = " ".join(str(text).split())
            if not text:
                continue
            snippets.append(f"- document page {page_index + 1}: {text[: self.mmlongbench_page_text_char_limit]}")
        return snippets

    def _opened_node_text_snippets(self, state) -> list[str]:
        snippets = []
        for node_id in state.opened_node_ids():
            node = state.graph.node(node_id)
            text = state.graph.node_text(node_id) or str(node.get("abstract") or "")
            text = " ".join(str(text).split())
            if not text:
                continue
            node_type = str(node.get("type") or "")
            page_number = state.graph.node_page_index(node) + 1
            snippets.append(
                f"- page {page_number} {node_type}: {text[: self.opened_node_text_char_limit]}"
            )
        return snippets

    def _is_locating_question(self, question: str) -> bool:
        text = str(question or "").lower()
        return any(marker in text for marker in (
            "select titles",
            "select table names",
            "which section",
            "which title",
            "what's name of the table",
            "what is the name of the table",
            "what's name of the figure",
            "what is the name of the figure",
            "list names of the other tables",
            "list names of the other figures",
            "which tables provide",
            "which figures provide",
            "where can we find",
            "best matches",
        ))

    def _is_color_question(self, question: str) -> bool:
        text = str(question or "").lower()
        return "color" in text or "colour" in text

    def _mmlongbench_domain_hints(self, question: str) -> list[str]:
        text = str(question or "").lower()
        hints = []
        if re.search(r"\bpages?\s*\d|\bslides?\s*\d|\bpage\s+range\b|\bslide\s+range\b", text):
            hints.extend([
                "For questions that name specific pages or slides, answer only from those named pages or slides.",
                "If the retrieved pages do not include the requested page or slide scope, answer exactly: Not answerable.",
            ])
        if re.search(r"\bfig(?:ure)?\.?\s+\w+|\btable\s+\w+", text):
            hints.extend([
                "For questions that name a specific figure or table, use that named figure or table as the primary evidence.",
                "Do not answer from a similarly themed figure or table with a different number.",
            ])
        if re.search(r"\bhow many (?:people|respondents|adults|participants|users|students|samples)\b", text):
            hints.extend([
                "For people-count questions, do not return a percentage when the question asks how many people.",
                "If evidence gives a percentage, identify the correct denominator or subgroup base and compute the count.",
            ])
        if re.search(r"\b(list|all|enumerate)\b", text) or re.search(r"\bhow many\b", text):
            hints.extend([
                "For exhaustive list or count questions, first enumerate the included items from the requested scope, then return the final list or count.",
                "Exclude nearby or related items that are outside the requested scope.",
            ])
        if " and " in text or " or " in text:
            hints.extend([
                "For compound questions, solve each requested part separately and combine the subanswers only after every part has supporting evidence.",
            ])
        return hints

    def _candidate_answer_strings(self, state) -> list[str]:
        values = []
        seen = set()
        for node_id in state.opened_node_ids() + state.active_node_ids():
            node = state.graph.node(node_id)
            node_type = str(node.get("type") or "").lower()
            if node_type not in {"title", "table", "figure", "chart"}:
                continue
            if node_type in {"table", "figure", "chart"}:
                candidate_values = (
                    node.get("caption"),
                    node.get("title"),
                    node.get("name"),
                    state.graph.node_text(node_id).splitlines()[0] if state.graph.node_text(node_id) else "",
                    node.get("abstract"),
                )
            else:
                candidate_values = (
                    node.get("title"),
                    node.get("name"),
                    node.get("caption"),
                    state.graph.node_text(node_id).splitlines()[0] if state.graph.node_text(node_id) else "",
                    node.get("abstract"),
                )
            for value in candidate_values:
                value = str(value or "").strip()
                if not value or value in seen:
                    continue
                if ":page:" in value or ":block:" in value:
                    continue
                seen.add(value)
                values.append(value)
                break
        return values

    def _reader_page_indices(self, state) -> list[int]:
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
            if page_index in seen:
                continue
            seen.add(page_index)
            page_indices.append(page_index)
        return page_indices

    def _page_image_part(self, benchmark_name: str, sample: dict, page_index: int):
        if benchmark_name == "mmlongbench":
            from PIL import Image

            benchmark_cfg = require_config_value(self.cfg, "benchmarks")
            path = self._page_image_path(benchmark_name, sample, page_index)
            if not os.path.exists(path):
                return None
            with Image.open(path) as image:
                encoded = encode_pil_image_to_base64(image)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        image_path = self._page_image_path(benchmark_name, sample, page_index)
        if not os.path.exists(image_path):
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
        if self.include_page_images:
            page_indices = self._reader_page_indices(state)
            for image_index, page_index in enumerate(page_indices):
                image_path = self._page_image_path(benchmark_name, sample, page_index)
                if not os.path.exists(image_path):
                    continue
                refs.append({
                    "kind": "page",
                    "image_index": image_index,
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "image_path": image_path,
                    "label": self._image_label(benchmark_name, image_index, len(page_indices), page_index),
                    "mime_type": _image_mime_type(image_path),
                })
        if benchmark_name == "mmlongbench" and self.mmlongbench_include_opened_node_crops:
            for crop_index, node_id in enumerate(self._candidate_crop_node_ids(sample["question"], state)):
                node = state.graph.node(node_id)
                image_path = node.get("image_path") or node.get("crop_path")
                if not image_path and isinstance(node.get("metadata"), dict):
                    image_path = node["metadata"].get("image_path") or node["metadata"].get("crop_path")
                if not image_path or not os.path.exists(str(image_path)):
                    continue
                page_index = state.graph.node_page_index(node)
                refs.append({
                    "kind": "opened_node_crop",
                    "image_index": len(refs),
                    "crop_index": crop_index,
                    "node_id": node_id,
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "image_path": str(image_path),
                    "mime_type": _image_mime_type(str(image_path)),
                })
        return refs

    def _candidate_crop_node_ids(self, question: str, state) -> list[str]:
        candidates = []
        seen = set()
        for node_id in state.opened_node_ids():
            if node_id in seen or node_id in state.pruned_node_ids():
                continue
            seen.add(node_id)
            node = state.graph.node(node_id)
            if not node.get("bbox"):
                continue
            node_type = str(node.get("type") or "").lower()
            if node_type not in {"table", "figure", "chart", "image", "paragraph", "text", "title"}:
                continue
            score = self._crop_relevance_score(question, node, state.graph.node_text(node_id))
            if score <= 0:
                continue
            candidates.append((score, node_id))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [node_id for _, node_id in candidates[: self.mmlongbench_max_opened_node_crops]]

    def _crop_relevance_score(self, question: str, node: dict, node_text: str) -> int:
        node_type = str(node.get("type") or "").lower()
        question_text = str(question or "").lower()
        haystack = " ".join([
            str(node.get("abstract") or ""),
            str(node.get("caption") or ""),
            str(node.get("title") or ""),
            str(node_text or ""),
        ]).lower()
        score = 0
        if node_type in {"table", "figure", "chart", "image"}:
            score += 8
        elif node_type in {"title", "paragraph", "text"}:
            score += 3
        type_markers = {
            "table": ("table", "row", "column", "chart"),
            "chart": ("chart", "plot", "bar", "line", "axis", "color", "percentage", "percent"),
            "figure": ("figure", "image", "map", "diagram", "screenshot", "photo", "color"),
            "image": ("figure", "image", "map", "diagram", "screenshot", "photo", "color"),
            "title": ("title", "heading", "section"),
        }
        for marker in type_markers.get(node_type, ()):
            if marker in question_text:
                score += 12
                break
        terms = {
            term.strip(".,:;!?()[]{}\"'")
            for term in question_text.split()
            if len(term.strip(".,:;!?()[]{}\"'")) >= 4
        }
        if terms:
            score += min(12, sum(1 for term in terms if term in haystack))
        return score

    def _opened_node_crop_part(self, benchmark_name: str, sample: dict, state, node_id: str):
        if benchmark_name != "mmlongbench":
            return None
        from PIL import Image

        node = state.graph.node(node_id)
        bbox = node.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        benchmark_cfg = require_config_value(self.cfg, "benchmarks")
        page_index = state.graph.node_page_index(node)
        path = mmlongbench_png_page_path(benchmark_cfg, sample["doc_id"], page_index)
        if not os.path.exists(path):
            return None
        with Image.open(path) as image:
            width, height = image.size
            left, top, right, bottom = [int(round(float(value))) for value in bbox]
            pad_x = max(8, int((right - left) * 0.06))
            pad_y = max(8, int((bottom - top) * 0.06))
            left = max(0, left - pad_x)
            top = max(0, top - pad_y)
            right = min(width, right + pad_x)
            bottom = min(height, bottom + pad_y)
            if right - left < 20 or bottom - top < 20:
                return None
            crop = image.crop((left, top, right, bottom))
            encoded = encode_pil_image_to_base64(crop)
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


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
