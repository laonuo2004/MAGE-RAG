from __future__ import annotations

import html
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from baselines.aeg_rag.actions import CandidateAction
from baselines.aeg_rag.state import EvidenceAgentState
from benchmarks.utils.document_preprocess import encode_image_file_to_base64
from utils.llm_utils import call_llm_messages, completion_content

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorDecision:
    stop: bool = False
    selected_actions: list[dict[str, Any]] = field(default_factory=list)
    search_query: str | None = None
    prune_requests: list[dict[str, str]] = field(default_factory=list)
    summarize_requests: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""


class XMLEvaluator:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        retries: int = 1,
        raw_text_char_limit: int = 1200,
        include_images_for_opened_nodes: bool = False,
        max_candidate_actions: int = 120,
        candidate_preview_char_limit: int = 160,
        max_selected_actions_per_iteration: int = 4,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.retries = int(retries)
        self.raw_text_char_limit = int(raw_text_char_limit)
        self.include_images_for_opened_nodes = bool(include_images_for_opened_nodes)
        self.max_candidate_actions = int(max_candidate_actions)
        self.candidate_preview_char_limit = int(candidate_preview_char_limit)
        self.max_selected_actions_per_iteration = max(1, int(max_selected_actions_per_iteration))

    def build_context_xml(self, question: str, state: EvidenceAgentState, candidates: list[CandidateAction]) -> str:
        allowed_pages = sorted(state.graph.allowed_pages or [])
        parts = ["<agent_step_context>", f"  <question>{_esc(question)}</question>"]
        parts.append(f'  <allowed_scope graph_escape="{str(state.graph_escape).lower()}">')
        for page_index in allowed_pages:
            parts.append(f'    <page index="{page_index}" number="{page_index + 1}"/>')
        parts.append("  </allowed_scope>")
        parts.append("  <evidence_state>")
        parts.append("    <active_nodes>")
        for node_id in state.active_node_ids():
            parts.append(self._node_xml(state, node_id, detail=False))
        parts.append("    </active_nodes>")
        parts.append("    <opened_nodes>")
        for node_id in state.opened_node_ids():
            parts.append(self._node_xml(state, node_id, detail=True))
        parts.append("    </opened_nodes>")
        parts.append("    <pruned_nodes>")
        for node_id in state.pruned_node_ids():
            node = state.graph.node(node_id)
            parts.append(
                f'      <node id="{_esc(node_id)}" type="{_esc(node.get("type", ""))}">'
                f"<reason>{_esc(state.prune_reasons.get(node_id, ''))}</reason></node>"
            )
        parts.append("    </pruned_nodes>")
        parts.append("    <summaries>")
        for summary in state.summaries:
            source_ids = ",".join(summary.get("source_node_ids") or [])
            parts.append(
                f'      <summary id="{_esc(summary["summary_id"])}" source_node_ids="{_esc(source_ids)}">'
                f'{_esc(summary.get("text", ""))}</summary>'
            )
        parts.append("    </summaries>")
        parts.append("  </evidence_state>")
        parts.append(f"  <recent_trace>{_esc(json.dumps(_compact_recent_trace(state.trace[-5:]), ensure_ascii=False))}</recent_trace>")
        parts.append("  <candidate_actions>")
        for index, candidate in enumerate(candidates[: self.max_candidate_actions], start=1):
            attrs = " ".join(
                f'{_esc(str(key))}="{_esc(str(value))}"'
                for key, value in candidate.payload.items()
                if key in {"node_id", "page_index", "edge_id"}
            )
            if attrs:
                attrs = " " + attrs
            parts.append(
                f'    <action index="{index}" type="{_esc(candidate.action_type)}"{attrs}>'
                f"<preview>{_esc(_truncate(candidate.preview, self.candidate_preview_char_limit))}</preview></action>"
            )
        if not candidates:
            parts.append("    <none/>")
        parts.append("  </candidate_actions>")
        parts.append("</agent_step_context>")
        return "\n".join(parts)

    def call(self, client, question: str, state: EvidenceAgentState, candidates: list[CandidateAction]) -> tuple[EvaluatorDecision, str]:
        prompt = self.build_prompt(question, state, candidates)
        content = [{"type": "text", "text": prompt}]
        content.extend(self._opened_node_image_parts(state))
        completion = call_llm_messages(
            client,
            self.model_name,
            [{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=self.temperature,
            retries=self.retries,
            logger=logger,
            log_prefix="AEG-RAG evaluator",
            failure_value=lambda exc: f"<agent_decision><stop>true</stop><reason>Failed: {html.escape(str(exc))}</reason></agent_decision>",
        )
        raw = completion_content(completion)
        return parse_agent_decision_xml(raw), str(raw)

    def build_prompt(self, question: str, state: EvidenceAgentState, candidates: list[CandidateAction]) -> str:
        domain_guidance = self._domain_guidance(question)
        if domain_guidance:
            domain_guidance = "\n" + domain_guidance + "\n"
        return (
            "You are the AEG-RAG online evidence controller. Read the XML context and return only "
            "an <agent_decision> XML document matching the requested schema. Candidate actions are "
            "numbered with short integer indexes. To execute a candidate, place only its number under "
            "<selected_actions> as <action index=\"...\">; do not copy long node_id, edge_id, or "
            "candidate id strings. Select at most "
            f"{self.max_selected_actions_per_iteration} high-value actions per round. Prefer stopping "
            "once the opened evidence is sufficient instead of exploring weakly related actions. "
            "If <candidate_actions> contains <none/>, do not emit selected_actions; "
            "either stop if the evidence is sufficient or issue a search_request. Do not emit top-level "
            "<action> elements.\n"
            + domain_guidance
            + self.build_context_xml(question, state, candidates)
        )

    def _domain_guidance(self, question: str) -> str:
        text = str(question or "").lower()
        lines = []
        if re.search(r"\bpages?\s*\d|\bslides?\s*\d|\bpage\s+range\b|\bslide\s+range\b", text):
            lines.extend([
                "For questions naming specific pages or slides, first verify the requested page or slide scope.",
                "Do not answer from unrelated retrieved pages when the requested scope is missing.",
                "If candidate actions cannot reach the requested scope, issue a search_request for the page/slide and target evidence.",
            ])
        finance_markers = (
            "ratio",
            "cash ratio",
            "debt to total assets",
            "total debt",
            "total assets",
            "gross profit",
            "payables turnover",
            "current liabilities",
            "short-term investments",
            "accounts payable",
        )
        if any(marker in text for marker in finance_markers):
            lines.extend([
                "For financial ratio questions, identify the formula and required fields before selecting actions.",
                "If any required financial field is missing from opened evidence, issue a search_request for that field and year.",
                "Prefer annual statement tables over performance graphs or narrative summaries for financial calculations.",
            ])
        if re.search(r"\b(list|all|enumerate)\b", text) or re.search(r"\bwhich\s+(?:\w+\s+){0,3}(?:items?|sections?|pages?|figures?|examples?)\b", text):
            lines.extend([
                "For list or exhaustive questions, keep searching until all requested items and scopes are covered.",
                "Do not stop after finding only one matching item when the question asks for all items, multiple examples, or a list.",
                "Avoid adding nearby or related items unless the question explicitly asks for them.",
            ])
        return "\n".join(lines)

    def trace_input(self, question: str, state: EvidenceAgentState, candidates: list[CandidateAction]) -> dict[str, Any]:
        return {
            "context_xml": self.build_context_xml(question, state, candidates),
            "candidate_actions": [
                {
                    "index": index,
                    "id": candidate.id,
                    "action_type": candidate.action_type,
                    "payload": candidate.payload,
                    "preview": _truncate(candidate.preview, self.candidate_preview_char_limit),
                }
                for index, candidate in enumerate(candidates[: self.max_candidate_actions], start=1)
            ],
            "opened_image_refs": self._opened_node_image_refs(state),
        }

    def _node_xml(self, state: EvidenceAgentState, node_id: str, detail: bool) -> str:
        node = state.graph.node(node_id)
        attrs = (
            f'id="{_esc(node_id)}" type="{_esc(node.get("type", ""))}" '
            f'page_index="{state.graph.node_page_index(node)}"'
        )
        body = [f"<abstract>{_esc(node.get('abstract', ''))}</abstract>"]
        if detail:
            text = state.graph.node_text(node_id)
            truncated = len(text) > self.raw_text_char_limit
            body.append(
                f'<content truncated="{str(truncated).lower()}">'
                f"{_esc(text[:self.raw_text_char_limit])}</content>"
            )
            if node.get("bbox") is not None:
                body.append(f"<bbox>{_esc(str(node.get('bbox')))}</bbox>")
            if self._node_image_path(node):
                body.append('<image ref="image_0"/>')
        return f"      <node {attrs}>{''.join(body)}</node>"

    def _opened_node_image_parts(self, state: EvidenceAgentState) -> list[dict[str, Any]]:
        if not self.include_images_for_opened_nodes:
            return []
        parts = []
        for node_id in state.opened_node_ids():
            node = state.graph.node(node_id)
            image_path = self._node_image_path(node)
            if not image_path:
                continue
            parts.append({"type": "text", "text": f"Opened node image for {node_id}:\n"})
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{_image_mime_type(str(image_path))};base64,{encode_image_file_to_base64(str(image_path))}"},
            })
        return parts

    def _opened_node_image_refs(self, state: EvidenceAgentState) -> list[dict[str, Any]]:
        refs = []
        for node_id in state.opened_node_ids():
            node = state.graph.node(node_id)
            image_path = self._node_image_path(node)
            if not image_path:
                continue
            refs.append({
                "node_id": node_id,
                "page_index": state.graph.node_page_index(node),
                "image_path": image_path,
            })
        return refs

    def _node_image_path(self, node: dict[str, Any]) -> str | None:
        image_path = node.get("image_path") or node.get("crop_path")
        if not image_path and isinstance(node.get("metadata"), dict):
            image_path = node["metadata"].get("image_path") or node["metadata"].get("crop_path")
        return str(image_path) if image_path else None


def parse_agent_decision_xml(raw_xml: str) -> EvaluatorDecision:
    xml_text = _extract_agent_decision_xml(raw_xml)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return _parse_agent_decision_xml_loose(xml_text)
    if root.tag != "agent_decision":
        raise ValueError("Evaluator XML root must be <agent_decision>.")
    decision = EvaluatorDecision(stop=_text_bool(root.findtext("stop")), reason=str(root.findtext("reason") or ""))
    selected = root.find("selected_actions")
    if selected is not None:
        for action in selected.findall("action"):
            decision.selected_actions.append({
                "candidate_id": str(action.attrib.get("candidate_id") or ""),
                "candidate_index": _optional_int(action.attrib.get("index") or action.attrib.get("candidate_index")),
                "utility": str(action.attrib.get("utility") or ""),
                "reason": str(action.findtext("reason") or ""),
            })
    for action in root.findall("action"):
        candidate_id = str(action.attrib.get("candidate_id") or action.attrib.get("id") or "")
        candidate_index = _optional_int(action.attrib.get("index") or action.attrib.get("candidate_index"))
        if candidate_id or candidate_index is not None:
            decision.selected_actions.append({
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "utility": str(action.attrib.get("utility") or ""),
                "reason": str(action.findtext("reason") or ""),
            })
    search_request = root.find("search_request")
    if search_request is not None and search_request.findtext("query") is not None:
        decision.search_query = str(search_request.findtext("query") or "")
    prune_requests = root.find("prune_requests")
    if prune_requests is not None:
        for node in prune_requests.findall("node"):
            decision.prune_requests.append({
                "node_id": str(node.attrib.get("id") or ""),
                "reason": str(node.findtext("reason") or ""),
            })
    summarize_requests = root.find("summarize_requests")
    if summarize_requests is not None:
        for summary in summarize_requests.findall("summary"):
            source_ids = [node_id.strip() for node_id in str(summary.attrib.get("source_node_ids") or "").split(",") if node_id.strip()]
            decision.summarize_requests.append({
                "source_node_ids": source_ids,
                "goal": str(summary.findtext("goal") or ""),
            })
    return decision


def _parse_agent_decision_xml_loose(xml_text: str) -> EvaluatorDecision:
    text = str(xml_text or "")
    decision = EvaluatorDecision(
        stop=_text_bool(_first_tag_text(text, "stop")),
        reason=_first_tag_text(text, "reason") or "",
    )
    selected_block = _first_tag_block(text, "selected_actions")
    if selected_block:
        for attrs in _iter_action_attrs(selected_block):
            decision.selected_actions.append(_selected_action_from_attrs(attrs))
    else:
        text_without_candidates = re.sub(
            r"<candidate_actions\b[^>]*>.*?</candidate_actions>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        for attrs in _iter_action_attrs(text_without_candidates):
            decision.selected_actions.append(_selected_action_from_attrs(attrs))
    query = _first_tag_text(text, "query")
    if query:
        decision.search_query = query
    return decision


def _selected_action_from_attrs(attrs_text: str) -> dict[str, Any]:
    attrs = {
        key: value
        for key, _quote, value in re.findall(r"([A-Za-z_][\w:-]*)\s*=\s*(['\"])(.*?)\2", attrs_text, flags=re.DOTALL)
    }
    return {
        "candidate_id": str(attrs.get("candidate_id") or attrs.get("id") or ""),
        "candidate_index": _optional_int(attrs.get("index") or attrs.get("candidate_index")),
        "utility": str(attrs.get("utility") or ""),
        "reason": "",
    }


def _iter_action_attrs(text: str):
    for match in re.finditer(r"<action\b([^>]*)/?>", text, flags=re.DOTALL | re.IGNORECASE):
        yield match.group(1)


def _first_tag_text(text: str, tag_name: str) -> str:
    block = _first_tag_block(text, tag_name)
    if not block:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", "", block)).strip()


def _first_tag_block(text: str, tag_name: str) -> str:
    match = re.search(
        rf"<{re.escape(tag_name)}\b[^>]*>(.*?)</{re.escape(tag_name)}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1) if match else ""


def _text_bool(text: str | None) -> bool:
    return str(text or "").strip().lower() == "true"


def _optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _truncate(value: Any, char_limit: int) -> str:
    text = str(value or "")
    limit = max(0, int(char_limit))
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[:limit] + "..."


def _compact_recent_trace(trace_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in trace_items:
        action = str(item.get("action") or "")
        summary: dict[str, Any] = {
            "iteration": item.get("iteration"),
            "action": action,
        }
        if "ok" in item:
            summary["ok"] = bool(item.get("ok"))
        if item.get("message"):
            summary["message"] = _truncate(item.get("message"), 160)
        payload = item.get("payload")
        if isinstance(payload, dict):
            summary["payload"] = _compact_mapping(payload)
        decision = item.get("decision")
        if action == "EvaluatorDecision" and isinstance(decision, dict):
            summary["decision"] = {
                "stop": bool(decision.get("stop")),
                "selected_action_count": len(decision.get("selected_actions") or []),
                "has_search_query": bool(decision.get("search_query")),
                "prune_request_count": len(decision.get("prune_requests") or []),
                "summarize_request_count": len(decision.get("summarize_requests") or []),
                "reason": _truncate(decision.get("reason", ""), 160),
            }
        if item.get("candidate_id"):
            summary["candidate_id"] = _truncate(item.get("candidate_id"), 120)
        compact.append(summary)
    return compact


def _compact_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "node_id",
        "page_index",
        "source",
        "previous_state",
        "reactivated_from_pruned",
        "edge_id",
        "target_id",
        "query",
        "score",
    }
    compact = {}
    for key, value in payload.items():
        if key not in allowed_keys:
            continue
        compact[str(key)] = _truncate(value, 160) if isinstance(value, str) else value
    return compact


def _extract_agent_decision_xml(raw_xml: str) -> str:
    text = str(raw_xml).strip()
    fenced = re.search(r"```(?:xml)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("<agent_decision")
    end = text.rfind("</agent_decision>")
    if start >= 0 and end >= 0:
        return text[start:end + len("</agent_decision>")]
    return text


def _image_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"
