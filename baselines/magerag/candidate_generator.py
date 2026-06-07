from __future__ import annotations

import re

from baselines.magerag.actions import CandidateAction
from baselines.magerag.graph_store import EvidenceGraphStore
from baselines.magerag.state import ACTIVE, INACTIVE, PRUNED, EvidenceAgentState


class CandidateGenerator:
    """
    根据当前证据状态枚举 evaluator 可以选择的下一步。

    它不判断答案是否充分，只负责把“可能扩展证据图的动作”列出来；
    marginal utility 的判断交给 XMLEvaluator。
    """

    def __init__(self, graph: EvidenceGraphStore):
        self.graph = graph

    def generate(self, state: EvidenceAgentState) -> list[CandidateAction]:
        candidates: list[CandidateAction] = []
        seen = set()

        def add(action_type, payload, preview=""):
            key = (action_type, tuple(sorted(payload.items())))
            if key in seen:
                return
            seen.add(key)
            candidates.append(CandidateAction(
                id=_stable_candidate_id(action_type, payload),
                action_type=action_type,
                payload=payload,
                preview=preview,
            ))

        # 1. 已激活页面上的未读元素：对应 Open Element 前的 ActivateNode。
        for page_id in state.active_node_ids():
            page = self.graph.node(page_id)
            if not self.graph.is_page_node(page):
                continue
            page_index = self.graph.node_page_index(page)
            for node in self.graph.nodes_on_page(page_index):
                node_id = str(node["id"])
                node_state = state.state_of(node_id)
                if node_state in {INACTIVE, PRUNED}:
                    add("ActivateNode", {"node_id": node_id}, self.graph.preview_node(node_id))

        # 2. 已激活元素本身可以被打开，也可以沿结构/语义边继续扩展。
        for node_id in state.active_node_ids():
            if self.graph.is_page_node(self.graph.node(node_id)):
                continue
            add("OpenNode", {"node_id": node_id}, self.graph.preview_node(node_id))
            for edge in self.graph.out_edges.get(node_id, []):
                if str(edge["id"]) in state.active_edges:
                    continue
                target_id = str(edge["target"])
                target_page = self.graph.node_page_index(target_id)
                if self.graph.is_page_allowed(target_page):
                    add("FollowRelation", {"edge_id": str(edge["id"])}, self.graph.preview_node(target_id))

        # 3. 已 follow 的边在下一轮转化为目标页面/节点激活候选。
        for edge in state.active_edges.values():
            target_id = str(edge["target"])
            target_page = self.graph.node_page_index(target_id)
            if not self.graph.is_page_allowed(target_page):
                continue
            page_id = self.graph.parent_page_node_id(target_id)
            if state.state_of(page_id) == INACTIVE:
                add(
                    "ActivatePage",
                    {"page_index": target_page, "source": "relation_target"},
                    self.graph.preview_node(target_id),
                )
            elif state.state_of(target_id) in {INACTIVE, PRUNED}:
                add("ActivateNode", {"node_id": target_id}, self.graph.preview_node(target_id))

        # 4. Jump/SearchEvidence 的搜索结果同样先变成页面或节点激活候选。
        for item in state.search_results:
            node = item["node"]
            node_id = str(node["id"])
            page_index = self.graph.node_page_index(node)
            page_id = self.graph.parent_page_node_id(node_id)
            if state.state_of(page_id) == INACTIVE:
                add("ActivatePage", {"page_index": page_index, "source": "search"}, item.get("preview", ""))
            elif state.state_of(node_id) in {INACTIVE, PRUNED}:
                add("ActivateNode", {"node_id": node_id}, item.get("preview", ""))

        return candidates


def _stable_candidate_id(action_type: str, payload: dict) -> str:
    # ID 必须独立于候选列表顺序；LLM 偶尔会引用旧 ID，稳定 ID 能降低解析失败率。
    if "node_id" in payload:
        target = str(payload["node_id"])
    elif "edge_id" in payload:
        target = str(payload["edge_id"])
    elif "page_index" in payload:
        target = f'{payload.get("source", "page")}:{payload["page_index"]}'
    else:
        target = "|".join(f"{key}={payload[key]}" for key in sorted(payload))
    return f"act:{action_type}:{_xml_id_token(target)}"


def _xml_id_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).strip("_") or "target"
