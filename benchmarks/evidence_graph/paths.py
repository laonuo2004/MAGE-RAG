from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.utils.data_utils import colpali_pdf_embeddings_path, mmlongbench_file_id


@dataclass(frozen=True)
class GraphPathContext:
    benchmark_name: str
    doc_id: str
    doc_key: str
    mineru_dir: Path
    page_image_paths: dict[int, Path]
    page_embedding_path: Path
    graph_dir: Path
    node_embedding_dir: Path


def default_graph_root(code_dir: Path, benchmark_name: str, shard: str = "4000-4999") -> Path:
    root = code_dir / "benchmarks" / benchmark_name / "data" / "processed" / "evidence_graphs"
    return root / shard if benchmark_name == "longdocurl" else root


def default_node_embedding_root(code_dir: Path, benchmark_name: str, shard: str = "4000-4999") -> Path:
    root = code_dir / "benchmarks" / benchmark_name / "data" / "cache" / "colpali" / "node_embeddings"
    return root / shard if benchmark_name == "longdocurl" else root


def mmlongbench_context(
    *,
    code_dir: Path,
    doc_id: Any,
    mineru_root: Path,
    png_root: Path,
    graph_root: Path | None = None,
    node_embedding_root: Path | None = None,
) -> GraphPathContext:
    doc_key = mmlongbench_file_id(doc_id)
    return GraphPathContext(
        benchmark_name="mmlongbench",
        doc_id=str(doc_id),
        doc_key=doc_key,
        mineru_dir=mineru_root / doc_key,
        page_image_paths=_mmlongbench_page_images(png_root / doc_key),
        page_embedding_path=colpali_pdf_embeddings_path("mmlongbench", doc_key),
        graph_dir=(graph_root or default_graph_root(code_dir, "mmlongbench")) / doc_key,
        node_embedding_dir=(node_embedding_root or default_node_embedding_root(code_dir, "mmlongbench")) / doc_key,
    )


def longdocurl_context(
    *,
    code_dir: Path,
    doc_no: Any,
    mineru_root: Path,
    png_root: Path,
    shard: str = "4000-4999",
    graph_root: Path | None = None,
    node_embedding_root: Path | None = None,
) -> GraphPathContext:
    doc_key = str(doc_no)
    return GraphPathContext(
        benchmark_name="longdocurl",
        doc_id=doc_key,
        doc_key=doc_key,
        mineru_dir=mineru_root / doc_key,
        page_image_paths=_longdocurl_page_images(png_root, doc_key),
        page_embedding_path=colpali_pdf_embeddings_path("longdocurl", doc_key, shard=shard),
        graph_dir=(graph_root or default_graph_root(code_dir, "longdocurl", shard)) / doc_key,
        node_embedding_dir=(node_embedding_root or default_node_embedding_root(code_dir, "longdocurl", shard)) / doc_key,
    )


def _mmlongbench_page_images(page_dir: Path) -> dict[int, Path]:
    paths = {}
    for path in page_dir.glob("page_*_dpi*.png"):
        try:
            page_index = int(path.name.split("_")[1]) - 1
        except (IndexError, ValueError):
            continue
        paths[page_index] = path
    return paths


def _longdocurl_page_images(png_root: Path, doc_key: str) -> dict[int, Path]:
    page_dir = png_root / doc_key[:4]
    paths = {}
    for path in page_dir.glob(f"{doc_key}_*.png"):
        try:
            page_index = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        paths[page_index] = path
    return paths

