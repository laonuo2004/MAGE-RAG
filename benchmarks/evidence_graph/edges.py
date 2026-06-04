import math
import re
from collections import defaultdict

from benchmarks.evidence_graph.schema import EvidenceEdge, EvidenceNode


def build_structural_edges(nodes: list[EvidenceNode], *, layout_k: int = 2) -> list[EvidenceEdge]:
    builder = EdgeBuilder()
    by_page = defaultdict(list)
    pages = []
    for node in nodes:
        if node.type == "page":
            pages.append(node)
        else:
            by_page[node.page_index].append(node)

    for page in pages:
        for node in by_page.get(page.page_index, []):
            builder.add(page.id, node.id, "containment", "contains", _bbox_area_weight(node))

    pages.sort(key=lambda node: node.page_index)
    for source, target in zip(pages, pages[1:]):
        builder.add(source.id, target.id, "reading_order", "next", 1.0)

    readable = []
    for page_index in sorted(by_page):
        readable.extend(sorted(by_page[page_index], key=lambda node: node.index if node.index is not None else 0))
    for source, target in zip(readable, readable[1:]):
        builder.add(source.id, target.id, "reading_order", "next", 1.0)

    for page_nodes in by_page.values():
        _add_layout_edges(builder, page_nodes, layout_k)
        _add_section_edges(builder, page_nodes)
    return builder.edges


class EdgeBuilder:
    def __init__(self) -> None:
        self.edges: list[EvidenceEdge] = []
        self.counts = defaultdict(int)

    def add(self, source: str, target: str, edge_type: str, relation: str, weight: float, metadata=None) -> None:
        key = (edge_type, relation, source, target)
        ordinal = self.counts[key]
        self.counts[key] += 1
        edge_id = f"edge:{edge_type}:{relation}:{_slug(source)}:{_slug(target)}:{ordinal}"
        self.edges.append(EvidenceEdge(edge_id, source, target, edge_type, relation, weight, metadata or {}))


def attach_edge_indexes(nodes: list[EvidenceNode], edges: list[EvidenceEdge]) -> None:
    by_id = {node.id: node for node in nodes}
    for edge in edges:
        if edge.source in by_id:
            by_id[edge.source].out_edges.append(edge.id)
        if edge.target in by_id:
            by_id[edge.target].in_edges.append(edge.id)


def _bbox_area_weight(node: EvidenceNode) -> float:
    if not node.bbox or len(node.bbox) != 4:
        return 1.0
    x0, y0, x1, y1 = node.bbox
    return float(max(1, x1 - x0) * max(1, y1 - y0))


def _add_layout_edges(builder: EdgeBuilder, nodes: list[EvidenceNode], layout_k: int) -> None:
    boxed = [node for node in nodes if node.bbox and len(node.bbox) == 4]
    for source in boxed:
        candidates = []
        sx0, sy0, sx1, sy1 = source.bbox
        scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
        for target in boxed:
            if target.id == source.id:
                continue
            tx0, ty0, tx1, ty1 = target.bbox
            relation = _horizontal_relation((sx0, sy0, sx1, sy1), (tx0, ty0, tx1, ty1))
            if not relation:
                continue
            tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
            dx, dy = tcx - scx, tcy - scy
            distance = math.sqrt(dx * dx + dy * dy)
            candidates.append((distance, relation, target))
        for distance, relation, target in sorted(candidates, key=lambda item: item[0])[:layout_k]:
            builder.add(source.id, target.id, "layout", relation, 1.0 / (1.0 + distance), {"distance": distance})


def _horizontal_relation(source_bbox: tuple[float, float, float, float], target_bbox: tuple[float, float, float, float]) -> str | None:
    sx0, sy0, sx1, sy1 = source_bbox
    tx0, ty0, tx1, ty1 = target_bbox
    vertical_overlap = max(0.0, min(sy1, ty1) - max(sy0, ty0))
    if vertical_overlap <= 0:
        return None
    if tx0 >= sx1:
        return "right_of"
    if tx1 <= sx0:
        return "left_of"
    return None


def _add_section_edges(builder: EdgeBuilder, nodes: list[EvidenceNode]) -> None:
    stack: list[EvidenceNode] = []
    for node in sorted(nodes, key=lambda item: item.index if item.index is not None else 0):
        if node.type == "title":
            level = int(node.fields.get("level") or 1)
            while stack and int(stack[-1].fields.get("level") or 1) >= level:
                stack.pop()
            if stack:
                builder.add(stack[-1].id, node.id, "section_hierarchy", "has_subsection", 1.0)
            stack.append(node)
        elif stack:
            builder.add(stack[-1].id, node.id, "section_hierarchy", "contains_block", 1.0)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)[-120:]

