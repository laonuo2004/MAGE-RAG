from dataclasses import dataclass
import logging
from time import time

from benchmarks.evidence_graph.edges import attach_edge_indexes, build_structural_edges
from benchmarks.evidence_graph.embeddings import materialize_node_embeddings
from benchmarks.evidence_graph.mineru_loader import load_mineru_document
from benchmarks.evidence_graph.nodes import build_nodes
from benchmarks.evidence_graph.paths import GraphPathContext
from benchmarks.evidence_graph.schema import BuildResult, EvidenceGraph
from benchmarks.evidence_graph.semantic import build_semantic_edges
from benchmarks.evidence_graph.summaries import fill_llm_abstracts
from benchmarks.evidence_graph.writer import load_graph_artifacts, write_graph_artifacts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceGraphBuildConfig:
    semantic_k: int = 5
    layout_k: int = 2
    skip_llm: bool = False
    skip_embeddings: bool = False
    overwrite: bool = False
    preserve_existing_abstracts: bool = True
    overwrite_llm_abstracts: bool = False
    model: str = "Qwen3-VL-8B-Instruct"
    vllm_url: str = "http://localhost:8020"
    litellm_base_url: str = "http://localhost:4000/v1"
    semantic_device: str = "auto"
    max_pages: int | None = None
    abstract_processor_path: str | None = None
    abstract_context_window: int | None = None
    abstract_output_tokens: int = 4096
    abstract_safety_margin: int = 2048


def build_document_graph(context: GraphPathContext, config: EvidenceGraphBuildConfig) -> BuildResult:
    if context.graph_dir.joinpath("graph.json").exists() and not config.overwrite:
        return BuildResult(context.doc_key, str(context.graph_dir), 0, 0, "skipped")

    pages, content_pages, content_path = load_mineru_document(context.mineru_dir)
    mineru_page_count = len(pages)
    content_page_count = len(content_pages)
    pages, content_pages = _apply_max_pages(pages, content_pages, context=context, max_pages=config.max_pages)
    nodes = build_nodes(
        doc_id=context.doc_id,
        doc_key=context.doc_key,
        pages=pages,
        content_pages=content_pages,
        page_image_paths=context.page_image_paths,
        page_embedding_path=context.page_embedding_path,
        node_embedding_dir=context.node_embedding_dir,
        mineru_dir=context.mineru_dir,
        skip_llm=config.skip_llm,
    )
    protected_abstract_node_ids = set()
    if config.preserve_existing_abstracts and not config.overwrite_llm_abstracts:
        protected_abstract_node_ids = _restore_existing_abstracts(nodes, context.graph_dir)
    fill_llm_abstracts(
        nodes,
        model=config.model,
        litellm_base_url=config.litellm_base_url,
        skip_llm=config.skip_llm,
        protected_node_ids=protected_abstract_node_ids,
        abstract_processor_path=config.abstract_processor_path,
        abstract_context_window=config.abstract_context_window,
        abstract_output_tokens=config.abstract_output_tokens,
        abstract_safety_margin=config.abstract_safety_margin,
    )
    edges = build_structural_edges(nodes, layout_k=config.layout_k)
    semantic_index_path = context.graph_dir / "semantic_index.safetensors"
    if config.overwrite and semantic_index_path.exists():
        semantic_index_path.unlink()
    if not config.skip_embeddings:
        semantic_vectors = materialize_node_embeddings(
            nodes,
            model="colpali-v1.3",
            vllm_url=config.vllm_url,
            skip_embeddings=config.skip_embeddings,
            overwrite=config.overwrite,
        )
        edges.extend(build_semantic_edges(nodes, semantic_vectors, semantic_k=config.semantic_k, semantic_device=config.semantic_device))
    attach_edge_indexes(nodes, edges)

    metadata = {
        "benchmark_name": context.benchmark_name,
        "doc_id": context.doc_id,
        "doc_key": context.doc_key,
        "built_at_unix": int(time()),
        "sources": {
            "mineru_dir": str(context.mineru_dir),
            "content_list_v2": str(content_path),
            "page_embedding_path": str(context.page_embedding_path),
            "mineru_page_count": mineru_page_count,
            "content_page_count": content_page_count,
        },
        "config": {
            "semantic_k": config.semantic_k,
            "layout_k": config.layout_k,
            "skip_llm": config.skip_llm,
            "skip_embeddings": config.skip_embeddings,
            "preserve_existing_abstracts": config.preserve_existing_abstracts,
            "overwrite_llm_abstracts": config.overwrite_llm_abstracts,
            "model": config.model,
            "vllm_url": config.vllm_url,
            "litellm_base_url": config.litellm_base_url,
            "semantic_device": config.semantic_device,
            "max_pages": config.max_pages,
            "abstract_processor_path": config.abstract_processor_path,
            "abstract_context_window": config.abstract_context_window,
            "abstract_output_tokens": config.abstract_output_tokens,
            "abstract_safety_margin": config.abstract_safety_margin,
        },
        "semantic_embeddings": {
            "enabled": not config.skip_embeddings,
            "page_embedding_path": str(context.page_embedding_path),
            "node_embedding_dir": str(context.node_embedding_dir),
            "node_ids": [node.id for node in nodes] if not config.skip_embeddings else [],
        },
        "counts": {"nodes": len(nodes), "edges": len(edges), "pages_built": len(pages)},
    }
    write_graph_artifacts(EvidenceGraph(metadata=metadata, nodes=nodes, edges=edges), context.graph_dir)
    return BuildResult(context.doc_key, str(context.graph_dir), len(nodes), len(edges), "generated")


def _apply_max_pages(pages, content_pages, *, context: GraphPathContext, max_pages: int | None):
    if max_pages is None:
        return pages, content_pages
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1 when configured")

    capped_pages = [page for page in pages if int(page.get("page_idx")) < max_pages]
    capped_content_pages = content_pages[:max_pages]
    if len(capped_pages) != len(pages) or len(capped_content_pages) != len(content_pages):
        logger.info(
            "Capping %s graph pages to max_pages=%s: mineru_pages=%s->%s content_pages=%s->%s",
            context.doc_key,
            max_pages,
            len(pages),
            len(capped_pages),
            len(content_pages),
            len(capped_content_pages),
        )
    return capped_pages, capped_content_pages


def _restore_existing_abstracts(nodes, graph_dir):
    nodes_path = graph_dir / "nodes.jsonl"
    if not nodes_path.exists():
        return set()
    artifacts = load_graph_artifacts(graph_dir)
    existing_abstracts = {
        row.get("id"): row.get("abstract")
        for row in artifacts.nodes
        if row.get("id") and row.get("abstract")
    }
    protected_node_ids = set()
    for node in nodes:
        abstract = existing_abstracts.get(node.id)
        if abstract:
            node.abstract = str(abstract)
            protected_node_ids.add(node.id)
    return protected_node_ids
