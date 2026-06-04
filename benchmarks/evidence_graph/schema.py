from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceNode:
    id: str
    type: str
    doc_id: str
    page_index: int
    index: int | None = None
    bbox: list[int] | None = None
    abstract: str = ""
    embedding_path: str | None = None
    in_edges: list[str] = field(default_factory=list)
    out_edges: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "type": self.type,
            "doc_id": self.doc_id,
            "page_index": self.page_index,
            "abstract": self.abstract,
            "in_edges": self.in_edges,
            "out_edges": self.out_edges,
        }
        if self.index is not None:
            payload["index"] = self.index
        if self.bbox is not None:
            payload["bbox"] = self.bbox
        if self.embedding_path is not None:
            payload["embedding_path"] = self.embedding_path
        if self.metadata:
            payload["metadata"] = self.metadata
        payload.update(self.fields)
        return payload


@dataclass
class EvidenceEdge:
    id: str
    source: str
    target: str
    type: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "relation": self.relation,
            "weight": float(self.weight),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class EvidenceGraph:
    metadata: dict[str, Any]
    nodes: list[EvidenceNode]
    edges: list[EvidenceEdge]


@dataclass(frozen=True)
class BuildResult:
    doc_key: str
    graph_dir: str
    node_count: int
    edge_count: int
    status: str

