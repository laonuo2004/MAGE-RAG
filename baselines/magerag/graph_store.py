from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.evidence_graph.writer import load_graph_artifacts


PAGE_NODE_TYPE = "page"
BIDIRECTIONAL_EDGE_TYPES = {"reading_order", "section_hierarchy"}
BIDIRECTIONAL_LAYOUT_RELATIONS = {"left_of", "right_of"}
GRAPH_MODE_EDGE_TYPES = {
    "full_graph": None,
    "containment_only": {"containment"},
    "structural_graph": {"containment", "section_hierarchy", "hierarchy", "reading_order"},
    "semantic_graph": {"containment", "semantic"},
    "layout_graph": {"containment", "layout"},
}


class EvidenceGraphStore:
    """
    只读的文档证据图访问层。

    Offline 阶段已经把 PDF 解析成 page/text/table/figure 等节点和关系边；
    online agent 只通过这个 store 查询节点、边、页面归属和轻量检索结果。
    """

    def __init__(
        self,
        graph_dir: str | Path,
        allowed_pages: list[int] | set[int] | None = None,
        *,
        graph_mode: str = "full_graph",
        enabled_edge_types: list[str] | set[str] | tuple[str, ...] | None = None,
        disabled_edge_types: list[str] | set[str] | tuple[str, ...] | None = None,
        max_edges_per_node: int | None = None,
    ):
        self.graph_dir = Path(graph_dir)
        artifacts = load_graph_artifacts(self.graph_dir)
        self.metadata = artifacts.metadata
        self.allowed_pages = set(int(page) for page in allowed_pages) if allowed_pages is not None else None
        self.graph_mode = str(graph_mode or "full_graph")
        self.enabled_edge_types = _normalize_edge_types(enabled_edge_types)
        self.disabled_edge_types = _normalize_edge_types(disabled_edge_types) or set()
        self.max_edges_per_node = None if max_edges_per_node is None else max(0, int(max_edges_per_node))
        self.nodes = self._filter_nodes({str(node["id"]): dict(node) for node in artifacts.nodes})
        self.edges = self._filter_edges([dict(edge) for edge in artifacts.edges])
        self.out_edges: dict[str, list[dict[str, Any]]] = {}
        self.in_edges: dict[str, list[dict[str, Any]]] = {}
        for edge in self.edges.values():
            self.out_edges.setdefault(str(edge["source"]), []).append(edge)
            self.in_edges.setdefault(str(edge["target"]), []).append(edge)
        self.page_nodes: dict[int, str] = {}
        for node_id, node in list(self.nodes.items()):
            page_index = self.node_page_index(node)
            if self.is_page_node(node):
                self.page_nodes[page_index] = node_id

    def _filter_nodes(self, nodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if self.graph_mode != "page_only":
            return nodes
        return {
            node_id: node
            for node_id, node in nodes.items()
            if self.is_page_node(node)
        }

    def _filter_edges(self, edges: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        allowed_types = self._allowed_edge_types_for_mode()
        kept: list[dict[str, Any]] = []
        edge_counts_by_source: dict[str, int] = {}
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if source not in self.nodes or target not in self.nodes:
                continue
            edge_type = str(edge.get("type") or "").lower()
            if allowed_types is not None and edge_type not in allowed_types:
                continue
            if self.enabled_edge_types is not None and edge_type not in self.enabled_edge_types:
                continue
            if edge_type in self.disabled_edge_types:
                continue
            if self.max_edges_per_node is not None:
                current_count = edge_counts_by_source.get(source, 0)
                if current_count >= self.max_edges_per_node:
                    continue
                edge_counts_by_source[source] = current_count + 1
            kept.append(edge)
        return {str(edge["id"]): edge for edge in kept}

    def _allowed_edge_types_for_mode(self) -> set[str] | None:
        if self.graph_mode == "page_only":
            return set()
        if self.graph_mode not in GRAPH_MODE_EDGE_TYPES:
            raise ValueError(f"Unsupported MAGE-RAG graph.mode: {self.graph_mode}")
        return GRAPH_MODE_EDGE_TYPES[self.graph_mode]

    def is_page_allowed(self, page_index: int) -> bool:
        return bool(self.allowed_pages is None or int(page_index) in self.allowed_pages)

    def is_page_node(self, node: dict[str, Any]) -> bool:
        return str(node.get("type") or "").lower() == PAGE_NODE_TYPE or str(node.get("id") or "").startswith("page:")

    def page_node_id(self, page_index: int) -> str:
        return self.page_nodes[int(page_index)]

    def node(self, node_id: str) -> dict[str, Any]:
        try:
            return self.nodes[str(node_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown evidence node: {node_id}") from exc

    def edge(self, edge_id: str) -> dict[str, Any]:
        try:
            return self.edges[str(edge_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown evidence edge: {edge_id}") from exc

    def is_logically_bidirectional_edge(self, edge: dict[str, Any]) -> bool:
        edge_type = str(edge.get("type") or "").lower()
        relation = str(edge.get("relation") or "").lower()
        if edge_type in BIDIRECTIONAL_EDGE_TYPES:
            return True
        return edge_type == "layout" and relation in BIDIRECTIONAL_LAYOUT_RELATIONS

    def node_page_index(self, node: dict[str, Any] | str) -> int:
        if isinstance(node, str):
            node = self.node(node)
        return int(node.get("page_index", node.get("page_no", 0)))

    def parent_page_node_id(self, node_id: str) -> str:
        node = self.node(node_id)
        return self.page_node_id(self.node_page_index(node))

    def nodes_on_page(self, page_index: int, include_page: bool = False) -> list[dict[str, Any]]:
        page_index = int(page_index)
        nodes = [
            node for node in self.nodes.values()
            if self.node_page_index(node) == page_index and (include_page or not self.is_page_node(node))
        ]
        return sorted(nodes, key=lambda node: (str(node.get("type") or ""), str(node.get("id") or "")))

    def preview_node(self, node_id: str, char_limit: int = 240) -> str:
        node = self.node(node_id)
        text = str(
            node.get("abstract")
            or node.get("text")
            or node.get("content")
            or node.get("caption")
            or node.get("html")
            or ""
        ).strip()
        return text[:char_limit] + ("..." if len(text) > char_limit else "")

    def node_text(self, node_id: str) -> str:
        node = self.node(node_id)
        parts = []
        for key in ("abstract", "text", "content", "html", "caption", "markdown", "md_content"):
            value = node.get(key)
            if value:
                parts.append(str(value))
        fields = node.get("fields")
        if isinstance(fields, dict):
            for key in ("text", "content", "html", "caption", "markdown", "md_content"):
                value = fields.get(key)
                if value:
                    parts.append(str(value))
        return "\n".join(dict.fromkeys(parts))

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        # 当前 search 是 lightweight lexical fallback，不替代 Stage I 的 ColPali 检索。
        # 它服务于 evaluator 的 Jump/SearchEvidence，用来在已有 graph artifact 内找新的入口点。
        terms = [term.lower() for term in str(query).split() if term.strip()]
        scored = []
        for node_id, node in self.nodes.items():
            page_index = self.node_page_index(node)
            if not self.is_page_allowed(page_index):
                continue
            haystack = f"{node.get('type', '')} {self.node_text(node_id)}".lower()
            score = sum(haystack.count(term) for term in terms) if terms else 0
            if score > 0:
                scored.append((score, node_id, node))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"node": node, "score": score, "preview": self.preview_node(node_id)}
            for score, node_id, node in scored[:limit]
        ]


def _normalize_edge_types(edge_types) -> set[str] | None:
    if edge_types is None:
        return None
    return {str(edge_type).lower() for edge_type in edge_types}
