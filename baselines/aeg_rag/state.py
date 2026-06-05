from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from baselines.aeg_rag.actions import (
    ActionResult,
    ActivateNode,
    ActivatePage,
    FollowRelation,
    OpenNode,
    PruneNode,
    SearchEvidence,
    SummarizeNodes,
)
from baselines.aeg_rag.graph_store import EvidenceGraphStore


INACTIVE = "Inactive"
ACTIVE = "Active"
OPENED = "Opened"
PRUNED = "Pruned"


@dataclass
class EvidenceAgentState:
    graph: EvidenceGraphStore
    graph_escape: bool = False
    node_states: dict[str, str] = field(default_factory=dict)
    prune_reasons: dict[str, str] = field(default_factory=dict)
    active_edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    search_results: list[dict[str, Any]] = field(default_factory=list)

    def state_of(self, node_id: str) -> str:
        return self.node_states.get(str(node_id), INACTIVE)

    def execute(self, action, iteration: int | None = None) -> ActionResult:
        if isinstance(action, ActivatePage):
            result = self.activate_page(action.page_index, action.source)
        elif isinstance(action, ActivateNode):
            result = self.activate_node(action.node_id)
        elif isinstance(action, OpenNode):
            result = self.open_node(action.node_id)
        elif isinstance(action, FollowRelation):
            result = self.follow_relation(action.edge_id)
        elif isinstance(action, SearchEvidence):
            result = self.search_evidence(action.query)
        elif isinstance(action, PruneNode):
            result = self.prune_node(action.node_id, action.reason)
        elif isinstance(action, SummarizeNodes):
            result = self.summarize_nodes(action.node_ids, action.goal, iteration=iteration)
        else:
            result = ActionResult(False, type(action).__name__, f"Unsupported action: {action!r}")
        self._record_result(result, action, iteration)
        return result

    def activate_page(self, page_index: int, source: str) -> ActionResult:
        page_index = int(page_index)
        if not self.graph.is_page_allowed(page_index, self.graph_escape):
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
        if not self.graph.is_page_allowed(self.graph.node_page_index(node), self.graph_escape):
            return self._validation(
                "ActivateNode",
                f"node {node_id} is on page {self.graph.node_page_index(node)}, outside allowed_pages",
            )
        if self.graph.is_page_node(node):
            return self.activate_page(self.graph.node_page_index(node), "relation_target")
        page_node_id = self.graph.parent_page_node_id(node_id)
        if self.state_of(page_node_id) != ACTIVE:
            return self._validation(
                "ActivateNode",
                f"Parent page is not Active for node {node_id}; activate page {self.graph.node_page_index(node)} first.",
            )
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
        self.node_states[node_id] = OPENED
        return ActionResult(True, "OpenNode", payload={"node_id": node_id, "previous_state": state})

    def prune_node(self, node_id: str, reason: str) -> ActionResult:
        node_id = str(node_id)
        state = self.state_of(node_id)
        if state not in {ACTIVE, OPENED}:
            return self._validation("PruneNode", f"PruneNode requires Active or Opened node; {node_id} is {state}.")
        self.node_states[node_id] = PRUNED
        self.prune_reasons[node_id] = str(reason or "")
        return ActionResult(True, "PruneNode", payload={"node_id": node_id, "previous_state": state, "reason": reason})

    def follow_relation(self, edge_id: str) -> ActionResult:
        edge = self.graph.edge(edge_id)
        target_id = str(edge["target"])
        target = self.graph.node(target_id)
        target_page = self.graph.node_page_index(target)
        if not self.graph.is_page_allowed(target_page, self.graph_escape):
            return self._validation("FollowRelation", f"target page {target_page} is outside allowed_pages")
        self.active_edges[str(edge_id)] = edge
        candidates = []
        page_id = self.graph.parent_page_node_id(target_id)
        if self.state_of(page_id) == INACTIVE:
            candidates.append({"type": "ActivatePage", "page_index": target_page})
        if self.state_of(target_id) in {INACTIVE, PRUNED} and self.state_of(page_id) == ACTIVE:
            candidates.append({"type": "ActivateNode", "node_id": target_id})
        return ActionResult(True, "FollowRelation", payload={
            "edge_id": str(edge_id),
            "target_id": target_id,
            "target_preview": self.graph.preview_node(target_id),
            "candidate_actions": candidates,
        })

    def search_evidence(self, query: str) -> ActionResult:
        results = self.graph.search(query, graph_escape=self.graph_escape)
        self.search_results = results
        return ActionResult(True, "SearchEvidence", payload={
            "query": str(query),
            "results": [
                {
                    "node_id": str(item["node"]["id"]),
                    "page_index": self.graph.node_page_index(item["node"]),
                    "score": item["score"],
                    "preview": item["preview"],
                }
                for item in results
            ],
        })

    def summarize_nodes(self, node_ids: list[str], goal: str, iteration: int | None = None) -> ActionResult:
        normalized_ids = [str(node_id) for node_id in node_ids]
        invalid = [node_id for node_id in normalized_ids if self.state_of(node_id) not in {ACTIVE, OPENED, PRUNED}]
        if invalid:
            return self._validation("SummarizeNodes", f"Cannot summarize unknown/inactive nodes: {invalid}")
        summary_id = f"summary:iter{iteration or 0}:{len(self.summaries)}"
        snippets = [self.graph.preview_node(node_id, char_limit=160) for node_id in normalized_ids]
        summary = {
            "summary_id": summary_id,
            "source_node_ids": normalized_ids,
            "goal": str(goal or ""),
            "text": " ".join(snippet for snippet in snippets if snippet),
        }
        self.summaries.append(summary)
        return ActionResult(True, "SummarizeNodes", payload=summary)

    def active_node_ids(self) -> list[str]:
        return sorted(node_id for node_id, state in self.node_states.items() if state == ACTIVE)

    def opened_node_ids(self) -> list[str]:
        return sorted(node_id for node_id, state in self.node_states.items() if state == OPENED)

    def pruned_node_ids(self) -> list[str]:
        return sorted(node_id for node_id, state in self.node_states.items() if state == PRUNED)

    def final_node_states(self) -> dict[str, str]:
        return dict(sorted(self.node_states.items()))

    def snapshot(self) -> dict[str, Any]:
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
            "active_edge_ids": sorted(self.active_edges),
            "summary_ids": [str(summary.get("summary_id")) for summary in self.summaries],
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
