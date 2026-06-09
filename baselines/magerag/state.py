from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from baselines.magerag.actions import (
    ActionResult,
    ActivateNode,
    ActivatePage,
    OpenNode,
    PruneNode,
    SearchEvidence,
)
from baselines.magerag.graph_store import EvidenceGraphStore


INACTIVE = "Inactive"
ACTIVE = "Active"
OPENED = "Opened"
PRUNED = "Pruned"


@dataclass
class EvidenceAgentState:
    """
    Online evidence graph 的可变状态。

    状态机刻意把节点分成 Active/Open/Pruned：Active 表示进入工作记忆，
    Open 才把详细内容暴露给 evaluator/reader，Pruned 则保留“曾看过但暂不使用”的记录。
    """

    graph: EvidenceGraphStore
    node_states: dict[str, str] = field(default_factory=dict)
    prune_reasons: dict[str, str] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    search_results: list[dict[str, Any]] = field(default_factory=list)

    def state_of(self, node_id: str) -> str:
        return self.node_states.get(str(node_id), INACTIVE)

    def execute(self, action, iteration: int | None = None) -> ActionResult:
        # 所有状态修改都走这个 dispatcher，保证 trace 中能复盘每一步 agent 行为。
        if isinstance(action, ActivatePage):
            result = self.activate_page(action.page_index, action.source)
        elif isinstance(action, ActivateNode):
            result = self.activate_node(action.node_id)
        elif isinstance(action, OpenNode):
            result = self.open_node(action.node_id)
        elif isinstance(action, SearchEvidence):
            result = self.search_evidence(action.query)
        elif isinstance(action, PruneNode):
            result = self.prune_node(action.node_id, action.reason)
        else:
            result = ActionResult(False, type(action).__name__, f"Unsupported action: {action!r}")
        self._record_result(result, action, iteration)
        return result

    def activate_page(self, page_index: int, source: str) -> ActionResult:
        page_index = int(page_index)
        if not self.graph.is_page_allowed(page_index):
            return self._validation("ActivatePage", f"page_index {page_index} is outside allowed_pages")
        node_id = self.graph.page_node_id(page_index)
        previous_state = self.state_of(node_id)
        self.node_states[node_id] = ACTIVE
        return ActionResult(True, "ActivatePage", payload={
            "node_id": node_id,
            "page_index": page_index,
            "source": source,
            "previous_state": previous_state,
        })

    def activate_node(self, node_id: str) -> ActionResult:
        node_id = str(node_id)
        node = self.graph.node(node_id)
        if not self.graph.is_page_allowed(self.graph.node_page_index(node)):
            return self._validation(
                "ActivateNode",
                f"node {node_id} is on page {self.graph.node_page_index(node)}, outside allowed_pages",
            )
        if self.graph.is_page_node(node):
            return self.activate_page(self.graph.node_page_index(node), "relation_target")
        page_node_id = self.graph.parent_page_node_id(node_id)
        if self.state_of(page_node_id) == PRUNED:
            return self._validation("ActivateNode", f"Parent page is Pruned for node {node_id}.")
        if self.state_of(page_node_id) == INACTIVE:
            self.activate_page(self.graph.node_page_index(node), "relation_target")
        previous_state = self.state_of(node_id)
        reactivated = previous_state == PRUNED
        self.node_states[node_id] = ACTIVE
        return ActionResult(True, "ActivateNode", payload={
            "node_id": node_id,
            "previous_state": previous_state,
            "reactivated_from_pruned": reactivated,
        })

    def open_node(self, node_id: str) -> ActionResult:
        node_id = str(node_id)
        state = self.state_of(node_id)
        if state == OPENED:
            return ActionResult(False, "OpenNode", payload={
                "node_id": node_id,
                "previous_state": state,
                "already_opened": True,
            })
        if state != ACTIVE:
            return self._validation("OpenNode", f"OpenNode requires Active node; {node_id} is {state}. ActivateNode first.")
        if self.graph.is_page_node(self.graph.node(node_id)):
            return self._validation("OpenNode", f"Page nodes cannot be opened: {node_id}.")
        self.node_states[node_id] = OPENED
        return ActionResult(True, "OpenNode", payload={"node_id": node_id, "previous_state": state})

    def prune_node(self, node_id: str, reason: str) -> ActionResult:
        node_id = str(node_id)
        state = self.state_of(node_id)
        if state not in {ACTIVE, OPENED, INACTIVE}:
            return self._validation("PruneNode", f"PruneNode requires Active, Opened, or Inactive node; {node_id} is {state}.")
        self.node_states[node_id] = PRUNED
        self.prune_reasons[node_id] = str(reason or "")
        return ActionResult(True, "PruneNode", payload={"node_id": node_id, "previous_state": state, "reason": reason})

    def _relation_traversal_target(self, edge: dict[str, Any]) -> str:
        source_id = str(edge["source"])
        target_id = str(edge["target"])
        source_state = self.state_of(source_id)
        target_state = self.state_of(target_id)
        if source_state == ACTIVE and target_state != ACTIVE:
            return target_id
        if (
            target_state == ACTIVE
            and source_state != ACTIVE
            and self.graph.is_logically_bidirectional_edge(edge)
        ):
            return source_id
        if source_state == ACTIVE:
            return target_id
        if target_state == ACTIVE and self.graph.is_logically_bidirectional_edge(edge):
            return source_id
        return target_id

    def search_evidence(self, query: str) -> ActionResult:
        self.search_results = []
        return ActionResult(True, "SearchEvidence", payload={
            "query": str(query),
        })

    def active_node_ids(self) -> list[str]:
        return sorted(node_id for node_id, state in self.node_states.items() if state == ACTIVE)

    def opened_node_ids(self) -> list[str]:
        return sorted(node_id for node_id, state in self.node_states.items() if state == OPENED)

    def pruned_node_ids(self) -> list[str]:
        return sorted(node_id for node_id, state in self.node_states.items() if state == PRUNED)

    def final_node_states(self) -> dict[str, str]:
        return dict(sorted(self.node_states.items()))

    def snapshot(self) -> dict[str, Any]:
        # snapshot 是给 evaluator trace 和分析插件消费的紧凑状态，不应放入大段原文或图片数据。
        active_node_ids = self.active_node_ids()
        opened_node_ids = self.opened_node_ids()
        pruned_node_ids = self.pruned_node_ids()
        active_page_indices = sorted(
            self.graph.node_page_index(node_id)
            for node_id in active_node_ids + opened_node_ids
            if self.graph.is_page_node(self.graph.node(node_id))
        )
        return {
            "active_node_ids": active_node_ids,
            "opened_node_ids": opened_node_ids,
            "pruned_node_ids": pruned_node_ids,
            "active_page_indices": active_page_indices,
            "active_count": len(active_node_ids),
            "opened_count": len(opened_node_ids),
            "pruned_count": len(pruned_node_ids),
        }

    def _validation(self, action_type: str, message: str) -> ActionResult:
        error = {"action_type": action_type, "message": message}
        self.validation_errors.append(error)
        return ActionResult(False, action_type, message)

    def _record_result(self, result: ActionResult, action, iteration: int | None):
        self.trace.append({
            "iteration": iteration,
            "action": type(action).__name__,
            "ok": result.ok,
            "message": result.message,
            "payload": result.payload,
            "state_snapshot_after": self.snapshot(),
        })
