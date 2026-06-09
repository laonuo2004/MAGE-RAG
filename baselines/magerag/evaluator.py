from __future__ import annotations

import html
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from baselines.magerag.actions import CandidateAction
from baselines.magerag.state import EvidenceAgentState
from benchmarks.utils.document_preprocess import encode_image_file_to_base64
from utils.llm_utils import call_llm_messages, completion_content, xml_block

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorDecision:
    """
    LLM evaluator 的结构化决策结果。

    selected_actions 执行已有 ActivateNode 候选；open/search/prune 是开放式请求。
    stop=True 只有在没有任何增益动作时才会结束 online expansion。
    """

    stop: bool = False
    selected_actions: list[dict[str, Any]] = field(default_factory=list)
    open_requests: list[dict[str, str]] = field(default_factory=list)
    search_query: str | None = None
    prune_requests: list[dict[str, str]] = field(default_factory=list)


class XMLEvaluator:
    """
    Stage II 的 action evaluator。

    它把当前 evidence state 和 candidate actions 编成 XML，让 LLM 估计哪些动作有边际收益。
    设计重点是控制输出形状：模型只选 action index，避免复制长 node_id/edge_id 时出错。
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        retries: int = 1,
        raw_text_char_limit: int = 1200,
        include_images_for_opened_nodes: bool = False,
        candidate_preview_char_limit: int = 160,
        max_selected_actions_per_iteration: int = 4,
        prompt_style: str = "structured",
        include_few_shot_examples: bool = True,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.retries = int(retries)
        self.raw_text_char_limit = int(raw_text_char_limit)
        self.include_images_for_opened_nodes = bool(include_images_for_opened_nodes)
        self.candidate_preview_char_limit = int(candidate_preview_char_limit)
        self.max_selected_actions_per_iteration = max(1, int(max_selected_actions_per_iteration))
        self.prompt_style = str(prompt_style)
        self.include_few_shot_examples = bool(include_few_shot_examples)

    def build_context_xml(self, question: str, state: EvidenceAgentState, candidates: list[CandidateAction]) -> str:
        # XML context 是 evaluator 的“可观测状态”：按页面组织证据和候选动作。
        parts = ["<agent_step_context>", xml_block("question", question, escape=True, inline=True, indent=2)]
        parts.append("  <evidence_state>")
        for page_id in self._context_page_ids(state):
            if state.state_of(page_id) == "Pruned":
                continue
            page = state.graph.node(page_id)
            page_index = state.graph.node_page_index(page)
            parts.append(f'    <page id="{_esc(page_id)}" index="{page_index}">')
            parts.append("      " + xml_block("abstract", page.get("abstract", ""), escape=True, inline=True))
            for node in state.graph.nodes_on_page(page_index):
                node_id = str(node["id"])
                node_state = state.state_of(node_id)
                if node_state not in {"Active", "Opened", "Pruned"}:
                    continue
                parts.append(self._node_xml(state, node_id, node_state))
            parts.append("    </page>")
        parts.append("  </evidence_state>")
        parts.append(_recent_trace_xml(state.trace[-5:]))
        parts.append("  <candidate_actions>")
        activate_candidates = [candidate for candidate in candidates if candidate.action_type == "ActivateNode"]
        parts.append("    <ActivateNode>")
        for index, candidate in enumerate(activate_candidates, start=1):
            attrs_dict = _activate_candidate_xml_attrs(candidate)
            attrs = " ".join(f'{_esc(str(key))}="{_esc(str(value))}"' for key, value in attrs_dict.items())
            if attrs:
                attrs = " " + attrs
            parts.append(
                f'      <action index="{index}"{attrs}>'
                f'{xml_block("preview", _truncate(candidate.preview, self.candidate_preview_char_limit), escape=True, inline=True)}</action>'
            )
        if not activate_candidates:
            parts.append("      <none/>")
        parts.append("    </ActivateNode>")
        parts.append('    <OpenNode><template>&lt;open_requests&gt;&lt;node id="active non-page node id"/&gt;&lt;/open_requests&gt;</template></OpenNode>')
        parts.append('    <SearchEvidenceRequest><template>&lt;search_request&gt;&lt;query&gt;focused evidence query&lt;/query&gt;&lt;/search_request&gt;</template></SearchEvidenceRequest>')
        parts.append(
            '    <PruneNodeRequest><template>'
            '&lt;prune_requests&gt;&lt;page id="visible page id"&gt;&lt;reason&gt;why remove the whole page&lt;/reason&gt;&lt;/page&gt;&lt;/prune_requests&gt;\n'
            '&lt;prune_requests&gt;&lt;node id="visible element node id"&gt;&lt;reason&gt;why remove the element node&lt;/reason&gt;&lt;/node&gt;&lt;/prune_requests&gt;'
            '</template></PruneNodeRequest>'
        )
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
            log_prefix="MAGE-RAG evaluator",
            failure_value=lambda exc: f"<agent_decision><stop>true</stop><reason>Failed: {html.escape(str(exc))}</reason></agent_decision>",
        )
        raw = completion_content(completion)
        return parse_agent_decision_xml(raw), str(raw)

    def build_prompt(self, question: str, state: EvidenceAgentState, candidates: list[CandidateAction]) -> str:
        parts = [
            "<evaluator_prompt>",
            xml_block("role", "You are the MAGE-RAG online evidence controller.", escape=True, inline=True, indent=2),
            xml_block("objective", "Select evidence actions that most improve the final document-grounded answer.", escape=True, inline=True, indent=2),
            xml_block(
                "decision_policy",
                "Prefer actions that directly support, refute, complete, or disambiguate the question.\n"
                "Continue expanding while any available action can improve final answer evidence.\n"
                "Stop only when no ActivateNode, OpenNode, SearchEvidenceRequest, or PruneNodeRequest can improve the answer.\n"
                f"Select at most {self.max_selected_actions_per_iteration} numbered ActivateNode actions per round.\n"
                "The ActivateNode limit does not constrain OpenNode, SearchEvidenceRequest, or PruneNodeRequest.\n"
                "Prune active or opened evidence that is distracting, harmful, useless, or consuming context without helping the answer.",
                escape=True,
                indent=2,
            ),
            xml_block(
                "grounding_policy",
                "Base decisions only on the XML state, opened evidence, recent trace, and candidate actions.\n"
                "If useful numbered ActivateNode candidates exist, select their indexes.\n"
                "If useful active non-page nodes exist, request OpenNode.\n"
                "If useful evidence is missing from candidates, issue one focused SearchEvidenceRequest.",
                escape=True,
                indent=2,
            ),
            xml_block(
                "action_policy",
                "Numbered actions are only for ActivateNode candidates.\n"
                'To execute numbered ActivateNode candidates, write <action index="..."/> inside <selected_actions>.\n'
                "Use <open_requests> only for active non-page nodes visible in evidence_state.\n"
                "Page nodes cannot be opened.\n"
                "Page nodes and element nodes can be pruned.\n"
                "Pruning a page removes that page and its element nodes from future context and candidate expansion.\n"
                "Use one focused <search_request> when useful evidence is missing from candidates.\n"
                "Each pruned node must include a concise <reason> explaining why it should be removed from working evidence.",
                escape=True,
                indent=2,
            ),
            xml_block(
                "thinking_policy",
                "Before deciding, output one <think>...</think> block with detailed deliberation.\n"
                "Use think to compare the question, evidence_state, recent_trace, and candidate_actions.\n"
                "Every OpenNode or PruneNodeRequest target must be copied from an existing <page> or <node> element in evidence_state.",
                escape=True,
                indent=2,
            ),
            xml_block(
                "output_schema",
                "After <think>, return exactly one <agent_decision> XML document.\n"
                "Available children are <stop>, <selected_actions>, <open_requests>, <search_request>, and <prune_requests>.",
                escape=True,
                indent=2,
            ),
            xml_block(
                "self_check",
                "Before returning, verify that XML is parseable, selected indexes exist, open node ids are active non-page nodes, prune node ids exist in evidence_state, and stop is used only when no useful action remains.",
                escape=True,
                indent=2,
            ),
        ]
        if self.include_few_shot_examples:
            parts.append(self._few_shot_examples_xml())
        parts.extend([
            "</evaluator_prompt>",
            self.build_context_xml(question, state, candidates),
        ])
        return "\n".join(parts)

    def _few_shot_examples_xml(self) -> str:
        return "\n".join([
            "  <few_shot_examples>",
            "    <example>",
            "      <problem>When should you open an active element node?</problem>",
            "      <example_input><question>Which publication is named as helpful?</question><evidence_state><page id=\"doc:page:10\" index=\"10\"><abstract>Helpful publications are introduced.</abstract><node id=\"doc:page:10:block:2:paragraph\" type=\"paragraph\" state=\"active\"><content>Other publications that were helpful include:</content></node></page></evidence_state><candidate_actions><ActivateNode><none/></ActivateNode><OpenNode><template>&lt;open_requests&gt;&lt;node id=\"active non-page node id\"/&gt;&lt;/open_requests&gt;</template></OpenNode></candidate_actions></example_input>",
            "      <example_output><think>The active paragraph introduces the requested publications, but only its preview is visible. Opening it can expose the publication names.</think><agent_decision><open_requests><node id=\"doc:page:10:block:2:paragraph\"/></open_requests></agent_decision></example_output>",
            "    </example>",
            "    <example>",
            "      <problem>When should you perform a search?</problem>",
            "      <example_input><question>Which city hosted the follow-up workshop?</question><evidence_state><page id=\"doc:page:2\" index=\"2\"><abstract>Initial workshop agenda.</abstract></page></evidence_state><candidate_actions><ActivateNode><action index=\"1\" source=\"doc:page:2\" relation=\"reading_order\"><preview>Budget notes.</preview></action></ActivateNode></candidate_actions></example_input>",
            "      <example_output><think>The question asks for a follow-up workshop city. The available candidate is budget noise, so a focused search is more useful.</think><agent_decision><search_request><query>follow-up workshop hosted city</query></search_request></agent_decision></example_output>",
            "    </example>",
            "    <example>",
            "      <problem>How should you prune an existing element node?</problem>",
            "      <example_input><question>What medication dose is recommended for adults?</question><evidence_state><page id=\"doc:page:8\" index=\"8\"><abstract>Pediatric dosing table.</abstract><node id=\"doc:page:8:block:1:table\" type=\"table\" state=\"opened\"><content>Pediatric dose: 5 mg daily.</content></node></page></evidence_state><candidate_actions><ActivateNode><action index=\"1\" source=\"doc:page:9\" relation=\"reading_order\"><preview>Adult dosing table: recommended dose...</preview></action></ActivateNode></candidate_actions></example_input>",
            "      <example_output><think>The opened table is pediatric and can mislead the adult-dose answer. Candidate 1 points to adult dosing, and the pediatric table should be removed.</think><agent_decision><selected_actions><action index=\"1\"/></selected_actions><prune_requests><node id=\"doc:page:8:block:1:table\"><reason>Pediatric dosing does not answer the adult-dose question.</reason></node></prune_requests></agent_decision></example_output>",
            "    </example>",
            "    <example>",
            "      <problem>How should you prune an existing page node?</problem>",
            "      <example_input><question>Which lab reported the final accuracy?</question><evidence_state><page id=\"doc:page:3\" index=\"3\"><abstract>Table of contents and publication credits.</abstract></page><page id=\"doc:page:7\" index=\"7\"><abstract>Experiment results mention the final accuracy table.</abstract></page></evidence_state><candidate_actions><ActivateNode><action index=\"1\" source=\"doc:page:7\" relation=\"containment\"><preview>Final accuracy by lab...</preview></action></ActivateNode><PruneNodeRequest><template>&lt;prune_requests&gt;&lt;page id=\"visible page id\"&gt;&lt;reason&gt;why remove the whole page&lt;/reason&gt;&lt;/page&gt;&lt;/prune_requests&gt;</template></PruneNodeRequest></candidate_actions></example_input>",
            "      <example_output><think>Page 3 is visible in evidence_state but unrelated to the accuracy question. Candidate 1 can add relevant result evidence.</think><agent_decision><selected_actions><action index=\"1\"/></selected_actions><prune_requests><page id=\"doc:page:3\"><reason>Table of contents and credits do not help answer the accuracy question.</reason></page></prune_requests></agent_decision></example_output>",
            "    </example>",
            "    <example>",
            "      <problem>When should you stop?</problem>",
            "      <example_input><question>What is the reported total revenue?</question><evidence_state><page id=\"doc:page:12\" index=\"12\"><abstract>Total revenue table.</abstract><node id=\"doc:page:12:block:3:table\" type=\"table\" state=\"opened\"><content>Total revenue: $4.2 million.</content></node></page></evidence_state><candidate_actions><ActivateNode><action index=\"1\" source=\"doc:page:12\" relation=\"layout\"><preview>Page number footer.</preview></action></ActivateNode></candidate_actions></example_input>",
            "      <example_output><think>The opened table directly contains the requested total revenue. The remaining candidate is a footer and cannot improve the answer.</think><agent_decision><stop>true</stop></agent_decision></example_output>",
            "    </example>",
            "  </few_shot_examples>",
        ])

    def trace_input(self, question: str, state: EvidenceAgentState, candidates: list[CandidateAction]) -> dict[str, Any]:
        return {
            "prompt_text": self.build_prompt(question, state, candidates),
            "context_xml": self.build_context_xml(question, state, candidates),
            "candidate_actions": [
                {
                    "index": index,
                    "id": candidate.id,
                    "action_type": candidate.action_type,
                    "payload": candidate.payload,
                    "preview": _truncate(candidate.preview, self.candidate_preview_char_limit),
                }
                for index, candidate in enumerate(candidates, start=1)
            ],
            "opened_image_refs": self._opened_node_image_refs(state),
        }

    def _context_page_ids(self, state: EvidenceAgentState) -> list[str]:
        page_ids = set()
        for node_id in state.active_node_ids() + state.opened_node_ids() + state.pruned_node_ids():
            if node_id not in state.graph.nodes:
                continue
            node = state.graph.node(node_id)
            page_id = node_id if state.graph.is_page_node(node) else state.graph.parent_page_node_id(node_id)
            page_ids.add(page_id)
        return sorted(page_ids, key=lambda page_id: (state.graph.node_page_index(page_id), page_id))

    def _node_xml(self, state: EvidenceAgentState, node_id: str, node_state: str) -> str:
        node = state.graph.node(node_id)
        attrs = (
            f'id="{_esc(node_id)}" type="{_esc(node.get("type", ""))}" '
            f'state="{_esc(node_state.lower())}"'
        )
        if node_state == "Pruned":
            body = [xml_block("content", state.prune_reasons.get(node_id, ""), escape=True, inline=True)]
        elif node_state == "Opened":
            text = state.graph.node_text(node_id)
            truncated = len(text) > self.raw_text_char_limit
            body = [
                xml_block(
                    "content",
                    text[:self.raw_text_char_limit],
                    attributes={"truncated": truncated},
                    escape=True,
                    inline=True,
                )
            ]
            if node.get("bbox") is not None:
                body.append(xml_block("bbox", str(node.get("bbox")), escape=True, inline=True))
            if self._node_image_path(node):
                body.append('<image ref="image_0"/>')
        else:
            body = [xml_block("content", node.get("abstract", ""), escape=True, inline=True)]
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
    """
    解析 evaluator 返回的 XML。

    严格解析优先；如果模型输出缺少闭合标签或混入说明文字，loose parser 尽量恢复可执行决策，
    让一次格式瑕疵不至于直接终止整轮 online expansion。
    """

    xml_text = _extract_agent_decision_xml(raw_xml)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return _parse_agent_decision_xml_loose(xml_text)
    if root.tag != "agent_decision":
        raise ValueError("Evaluator XML root must be <agent_decision>.")
    decision = EvaluatorDecision(stop=_text_bool(root.findtext("stop")))
    selected = root.find("selected_actions")
    if selected is not None:
        for action in selected.findall("action"):
            decision.selected_actions.append({
                "candidate_id": str(action.attrib.get("candidate_id") or ""),
                "candidate_index": _optional_int(action.attrib.get("index") or action.attrib.get("candidate_index")),
                "utility": str(action.attrib.get("utility") or ""),
            })
    open_requests = root.find("open_requests")
    if open_requests is not None:
        for node in open_requests.findall("node"):
            decision.open_requests.append({"node_id": str(node.attrib.get("id") or "")})
    search_request = root.find("search_request")
    if search_request is not None and search_request.findtext("query") is not None:
        decision.search_query = str(search_request.findtext("query") or "")
    prune_requests = root.find("prune_requests")
    if prune_requests is not None:
        for item in list(prune_requests):
            if item.tag not in {"page", "node"}:
                continue
            decision.prune_requests.append({
                "node_id": str(item.attrib.get("id") or ""),
                "reason": str(item.findtext("reason") or ""),
            })
    return decision


def _parse_agent_decision_xml_loose(xml_text: str) -> EvaluatorDecision:
    text = str(xml_text or "")
    decision = EvaluatorDecision(
        stop=_text_bool(_first_tag_text(text, "stop")),
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
    open_block = _first_tag_block(text, "open_requests")
    if open_block:
        for attrs in _iter_node_attrs(open_block):
            decision.open_requests.append({"node_id": _attr_value(attrs, "id")})
    prune_block = _first_tag_block(text, "prune_requests")
    if prune_block:
        for match in re.finditer(r"<(node|page)\b([^>]*)>(.*?)</\1>", prune_block, flags=re.DOTALL | re.IGNORECASE):
            decision.prune_requests.append({
                "node_id": _attr_value(match.group(2), "id"),
                "reason": _first_tag_text(match.group(3), "reason"),
            })
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
    }


def _iter_action_attrs(text: str):
    for match in re.finditer(r"<action\b([^>]*)/?>", text, flags=re.DOTALL | re.IGNORECASE):
        yield match.group(1)


def _iter_node_attrs(text: str):
    for match in re.finditer(r"<node\b([^>]*)/?>", text, flags=re.DOTALL | re.IGNORECASE):
        yield match.group(1)


def _attr_value(attrs_text: str, name: str) -> str:
    attrs = {
        key: value
        for key, _quote, value in re.findall(r"([A-Za-z_][\w:-]*)\s*=\s*(['\"])(.*?)\2", attrs_text, flags=re.DOTALL)
    }
    return str(attrs.get(name) or "")


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


def _recent_trace_xml(trace_items: list[dict[str, Any]]) -> str:
    lines = ["  <recent_trace>"]
    for item in trace_items:
        action = str(item.get("action") or "")
        attrs: dict[str, Any] = {
            "iteration": item.get("iteration"),
            "action": action,
        }
        payload = item.get("payload")
        if isinstance(payload, dict):
            target = payload.get("node_id") or payload.get("target_id") or payload.get("page_index") or payload.get("query")
            if target is not None:
                attrs["target"] = _truncate(target, 160)
        decision = item.get("decision")
        if action == "EvaluatorDecision" and isinstance(decision, dict):
            attrs["selected_actions"] = len(decision.get("selected_actions") or [])
            attrs["open_requests"] = len(decision.get("open_requests") or [])
            attrs["prune_requests"] = len(decision.get("prune_requests") or [])
            if decision.get("search_query"):
                attrs["query"] = _truncate(decision.get("search_query"), 160)
        selection = item.get("selection")
        if isinstance(selection, dict) and selection.get("candidate_index") is not None:
            attrs["selection"] = selection.get("candidate_index")
        attr_text = " ".join(f'{_esc(key)}="{_esc(value)}"' for key, value in attrs.items() if value is not None)
        lines.append(f"    <step {attr_text}/>")
    lines.append("  </recent_trace>")
    return "\n".join(lines)


def _activate_candidate_xml_attrs(candidate: CandidateAction) -> dict[str, Any]:
    payload = candidate.payload
    attrs: dict[str, Any] = {}
    if payload.get("source_node_id"):
        attrs["source"] = payload["source_node_id"]
    for key in ("edge_type", "relation", "traversal_hint"):
        if payload.get(key):
            attrs[key] = payload[key]
    return attrs


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
