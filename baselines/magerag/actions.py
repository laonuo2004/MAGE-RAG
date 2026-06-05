from dataclasses import dataclass, field
from typing import Any


# MAGE-RAG 的 online 阶段只通过这些小动作修改 evidence graph state。
# 这里保持动作为纯数据结构，方便 evaluator 输出、trace 记录和状态机执行解耦。
ALLOWED_ACTIVATE_PAGE_SOURCES = {"initial_retrieval", "adjacent", "relation_target", "search", "question_page_scope"}


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    action_type: str
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateAction:
    """
    Evaluator 看到的是候选动作，而不是直接可执行动作对象。

    CandidateAction 带 preview，用来评估 marginal utility；真正执行前再由
    action_from_candidate 转成状态机认识的 action dataclass。
    """

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
    """
    把 evaluator 选择的候选动作还原成状态机动作。

    SearchEvidence/SummarizeNodes 通常由 evaluator 的专门 XML 字段产生，
    不走 candidate list，因此这里故意只覆盖 graph expansion 候选。
    """

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
