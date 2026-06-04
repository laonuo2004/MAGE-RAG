from dataclasses import dataclass, field
from typing import Any


ALLOWED_ACTIVATE_PAGE_SOURCES = {"initial_retrieval", "adjacent", "relation_target", "search"}


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    action_type: str
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateAction:
    id: str
    action_type: str
    payload: dict[str, Any]
    preview: str = ""


@dataclass(frozen=True)
class ActivatePage:
    page_index: int
    source: str

    def __post_init__(self):
        if self.source not in ALLOWED_ACTIVATE_PAGE_SOURCES:
            raise ValueError(f"Invalid ActivatePage source: {self.source}")


@dataclass(frozen=True)
class ActivateNode:
    node_id: str


@dataclass(frozen=True)
class OpenNode:
    node_id: str


@dataclass(frozen=True)
class FollowRelation:
    edge_id: str


@dataclass(frozen=True)
class SearchEvidence:
    query: str


@dataclass(frozen=True)
class PruneNode:
    node_id: str
    reason: str


@dataclass(frozen=True)
class SummarizeNodes:
    node_ids: list[str]
    goal: str


def action_from_candidate(candidate: CandidateAction):
    payload = candidate.payload
    if candidate.action_type == "ActivatePage":
        return ActivatePage(page_index=int(payload["page_index"]), source=str(payload.get("source") or "search"))
    if candidate.action_type == "ActivateNode":
        return ActivateNode(node_id=str(payload["node_id"]))
    if candidate.action_type == "OpenNode":
        return OpenNode(node_id=str(payload["node_id"]))
    if candidate.action_type == "FollowRelation":
        return FollowRelation(edge_id=str(payload["edge_id"]))
    if candidate.action_type == "PruneNode":
        return PruneNode(node_id=str(payload["node_id"]), reason=str(payload.get("reason") or ""))
    raise ValueError(f"Unsupported candidate action type: {candidate.action_type}")
