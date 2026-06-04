import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.evidence_graph.schema import EvidenceGraph


@dataclass
class LoadedGraphArtifacts:
    metadata: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


def write_graph_artifacts(graph: EvidenceGraph, graph_dir: Path) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    _write_json(graph_dir / "graph.json", graph.metadata)
    _write_jsonl(graph_dir / "nodes.jsonl", [node.to_dict() for node in graph.nodes])
    _write_jsonl(graph_dir / "edges.jsonl", [edge.to_dict() for edge in graph.edges])


def load_graph_artifacts(graph_dir: Path) -> LoadedGraphArtifacts:
    return LoadedGraphArtifacts(
        metadata=json.loads((graph_dir / "graph.json").read_text(encoding="utf-8")),
        nodes=_read_jsonl(graph_dir / "nodes.jsonl"),
        edges=_read_jsonl(graph_dir / "edges.jsonl"),
    )


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

