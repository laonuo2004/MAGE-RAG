from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.evidence_graph.writer import load_graph_artifacts


PAGE_NODE_TYPE = "page"


class EvidenceGraphStore:
    """
    只读的文档证据图访问层。

    Offline 阶段已经把 PDF 解析成 page/text/table/figure 等节点和关系边；
    online agent 只通过这个 store 查询节点、边、页面归属和轻量检索结果。
    """

    def __init__(self, graph_dir: str | Path, allowed_pages: list[int] | set[int] | None = None):
        self.graph_dir = Path(graph_dir)
        artifacts = load_graph_artifacts(self.graph_dir)
        self.metadata = artifacts.metadata
        self.allowed_pages = set(int(page) for page in allowed_pages) if allowed_pages is not None else None
        self.nodes = {str(node["id"]): dict(node) for node in artifacts.nodes}
        self.edges = {str(edge["id"]): dict(edge) for edge in artifacts.edges}
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
        self._ensure_page_nodes()

    def _ensure_page_nodes(self):
        # 有些解析产物只包含页面内元素，没有显式 page 节点。
        # Online 阶段必须先激活 page 再激活元素，所以这里补 synthetic page node。
        page_indices = {self.node_page_index(node) for node in self.nodes.values()}
        if self.allowed_pages is not None:
            page_indices |= self.allowed_pages
        for page_index in sorted(page_indices):
            if page_index in self.page_nodes:
                continue
            node_id = f"page:{page_index}"
            self.nodes[node_id] = {
                "id": node_id,
                "type": PAGE_NODE_TYPE,
                "doc_id": str(self.metadata.get("doc_id") or self.metadata.get("doc_key") or ""),
                "page_index": page_index,
                "abstract": f"Document page {page_index + 1}",
                "metadata": {"synthetic": True},
            }
            self.page_nodes[page_index] = node_id

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
