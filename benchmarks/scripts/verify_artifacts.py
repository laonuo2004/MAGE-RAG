#!/usr/bin/env python3
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz
from safetensors import safe_open


CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from benchmarks.evidence_graph.mineru_loader import load_mineru_document
from benchmarks.evidence_graph.paths import default_graph_root, default_node_embedding_root
from benchmarks.evidence_graph.writer import load_graph_artifacts
from benchmarks.utils.data_utils import (
    colpali_pdf_embeddings_path,
    colpali_question_embeddings_path,
    load_longdocurl_samples,
    load_mmlongbench_samples,
    mmlongbench_file_id,
)


SHARD = "4000-4999"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify benchmark preprocessing artifacts.")
    parser.add_argument("--benchmark", choices=["longdocurl", "mmlongbench"], required=True)
    parser.add_argument(
        "--stage",
        choices=["png", "mineru", "colpali", "graph"],
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_expected(benchmark):
    if benchmark == "longdocurl":
        samples = load_longdocurl_samples(
            CODE_DIR / "benchmarks/longdocurl/data/raw/LongDocURL.jsonl"
        )
        doc_keys = sorted({str(sample["doc_no"]) for sample in samples})
        questions = sorted({str(sample["question_id"]) for sample in samples})
        return doc_keys, questions

    samples = load_mmlongbench_samples(
        CODE_DIR / "benchmarks/mmlongbench/data/raw/samples.json"
    )
    doc_keys = sorted({mmlongbench_file_id(sample["doc_id"]) for sample in samples})
    questions = sorted({str(sample["question_id"]) for sample in samples})
    return doc_keys, questions


def benchmark_root(benchmark, kind):
    root = CODE_DIR / "benchmarks" / benchmark / "data" / kind
    if benchmark == "longdocurl":
        return root / SHARD
    return root


def tensor_shape(path, key):
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as tensors:
        if key not in tensors.keys():
            raise KeyError(f"{path} does not contain {key!r}")
        shape = tuple(tensors.get_slice(key).get_shape())
    if not shape or any(size < 1 for size in shape):
        raise ValueError(f"Invalid tensor shape in {path}: {shape}")
    return shape


def page_count(benchmark, doc_key):
    if benchmark == "longdocurl":
        root = (
            CODE_DIR
            / "benchmarks/longdocurl/data/processed/pdf_pngs"
            / SHARD
            / doc_key[:4]
        )
        return len(list(root.glob(f"{doc_key}_*.png")))
    root = CODE_DIR / "benchmarks/mmlongbench/data/processed/pdf_pngs" / doc_key
    return len(list(root.glob("page_*_dpi144.png")))


def expected_png_path(benchmark, doc_key, page_index):
    if benchmark == "longdocurl":
        return (
            CODE_DIR
            / "benchmarks/longdocurl/data/processed/pdf_pngs"
            / SHARD
            / doc_key[:4]
            / f"{doc_key}_{page_index}.png"
        )
    return (
        CODE_DIR
        / "benchmarks/mmlongbench/data/processed/pdf_pngs"
        / doc_key
        / f"page_{page_index + 1:04d}_dpi144.png"
    )


def source_pdf_path(benchmark, doc_key):
    if benchmark == "longdocurl":
        return (
            CODE_DIR
            / "benchmarks/longdocurl/data/raw/pdfs"
            / SHARD
            / f"{doc_key}.pdf"
        )
    return CODE_DIR / "benchmarks/mmlongbench/data/raw/documents" / f"{doc_key}.pdf"


def verify_document_pngs(benchmark, doc_key):
    pdf_path = source_pdf_path(benchmark, doc_key)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    with fitz.open(pdf_path) as pdf:
        expected_pages = len(pdf) if benchmark == "longdocurl" else min(len(pdf), 120)
        for page_index in range(expected_pages):
            png_path = expected_png_path(benchmark, doc_key, page_index)
            if not png_path.is_file():
                raise FileNotFoundError(png_path)
            pixmap = fitz.Pixmap(png_path)
            page = pdf[page_index]
            pixel_rect = (page.rect * fitz.Matrix(2, 2)).irect
            expected_size = (pixel_rect.width, pixel_rect.height)
            actual_size = (pixmap.width, pixmap.height)
            if actual_size != expected_size:
                raise ValueError(
                    f"Unexpected PNG size for {png_path}: "
                    f"expected={expected_size}, actual={actual_size}"
                )
        return expected_pages


def verify_pngs(benchmark, doc_keys, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        page_counts = list(
            executor.map(
                lambda doc_key: verify_document_pngs(benchmark, doc_key),
                doc_keys,
            )
        )
    return sum(page_counts)


def verify_mineru(benchmark, doc_keys):
    root = benchmark_root(benchmark, "processed/pdfs_mineru")
    for doc_key in doc_keys:
        doc_dir = root / doc_key
        pages, content_pages, _ = load_mineru_document(doc_dir)
        if not pages or not content_pages:
            raise ValueError(f"Empty MinerU output: {doc_dir}")


def verify_colpali(benchmark, doc_keys, question_ids):
    for doc_key in doc_keys:
        path = colpali_pdf_embeddings_path(benchmark, doc_key, shard=SHARD)
        shape = tensor_shape(path, "embeddings")
        expected_pages = page_count(benchmark, doc_key)
        if len(shape) != 3 or shape[0] != expected_pages:
            raise ValueError(
                f"PDF embedding/page mismatch for {doc_key}: "
                f"embedding_shape={shape}, PNG_pages={expected_pages}"
            )

    for question_id in question_ids:
        path = colpali_question_embeddings_path(benchmark, question_id)
        shape = tensor_shape(path, "query_embedding")
        if len(shape) != 2:
            raise ValueError(f"Invalid question embedding shape in {path}: {shape}")


def verify_graph(benchmark, doc_keys):
    graph_root = default_graph_root(CODE_DIR, benchmark, SHARD)
    node_embedding_root = default_node_embedding_root(CODE_DIR, benchmark, SHARD)
    for doc_key in doc_keys:
        graph_dir = graph_root / doc_key
        artifacts = load_graph_artifacts(graph_dir)
        if not artifacts.nodes:
            raise ValueError(f"Graph contains no nodes: {graph_dir}")

        node_ids = {node["id"] for node in artifacts.nodes}
        edge_ids = {edge["id"] for edge in artifacts.edges}
        if len(node_ids) != len(artifacts.nodes) or len(edge_ids) != len(artifacts.edges):
            raise ValueError(f"Duplicate node or edge ids: {graph_dir}")
        for edge in artifacts.edges:
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                raise ValueError(f"Edge references an unknown node in {graph_dir}: {edge['id']}")

        expected_element_embeddings = 0
        for node in artifacts.nodes:
            if node["type"] == "page":
                continue
            expected_element_embeddings += 1
            embedding_path = Path(node["embedding_path"])
            if embedding_path.parent != node_embedding_root / doc_key:
                raise ValueError(f"Unexpected node embedding path: {embedding_path}")
            shape = tensor_shape(embedding_path, "embedding")
            if len(shape) != 2:
                raise ValueError(f"Invalid node embedding shape in {embedding_path}: {shape}")

        actual_element_embeddings = len(
            list((node_embedding_root / doc_key).glob("*.safetensors"))
        )
        if actual_element_embeddings != expected_element_embeddings:
            raise ValueError(
                f"Node embedding count mismatch for {doc_key}: "
                f"expected={expected_element_embeddings}, actual={actual_element_embeddings}"
            )


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    doc_keys, question_ids = load_expected(args.benchmark)
    if args.stage == "png":
        page_total = verify_pngs(args.benchmark, doc_keys, args.workers)
        count = len(doc_keys)
        detail = f"documents ({page_total} pages)"
    elif args.stage == "mineru":
        verify_mineru(args.benchmark, doc_keys)
        count = len(doc_keys)
        detail = "documents"
    elif args.stage == "colpali":
        verify_colpali(args.benchmark, doc_keys, question_ids)
        count = len(doc_keys) + len(question_ids)
        detail = f"artifacts ({len(doc_keys)} documents, {len(question_ids)} questions)"
    else:
        verify_graph(args.benchmark, doc_keys)
        count = len(doc_keys)
        detail = "graphs"
    print(f"Verified {args.benchmark} {args.stage}: {count} {detail}.")


if __name__ == "__main__":
    main()
