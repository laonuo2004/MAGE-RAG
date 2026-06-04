#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))

from benchmarks.evidence_graph.builder import EvidenceGraphBuildConfig, build_document_graph
from benchmarks.evidence_graph.paths import longdocurl_context, mmlongbench_context
from benchmarks.evidence_graph.summaries import warm_abstract_processor
from benchmarks.utils.data_utils import load_longdocurl_samples, load_mmlongbench_samples, mmlongbench_file_id

logger = logging.getLogger("build_evidence_graphs")


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-document Evidence Graph artifacts from MinerU outputs.")
    parser.add_argument("--benchmark", choices=["longdocurl", "mmlongbench"], required=True)
    parser.add_argument("--doc-id", action="append", default=None, help="Document id/doc_no to process. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default="4000-4999")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--semantic-k", type=int, default=3)
    parser.add_argument("--layout-k", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Number of documents to build concurrently.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging, including semantic edge timing.")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum source pages to include per graph. Defaults to 120 for mmlongbench and unlimited for longdocurl.",
    )
    parser.add_argument(
        "--overwrite-llm-abstracts",
        action="store_true",
        help="Regenerate LLM abstracts even when existing graph nodes already contain abstracts.",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8020")
    parser.add_argument("--litellm-base-url", default="http://localhost:4000/v1")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--abstract-processor-path",
        default=os.environ.get("EVIDENCE_GRAPH_ABSTRACT_PROCESSOR_PATH"),
        help="Local AutoProcessor path used to estimate abstract request tokens before each LLM call.",
    )
    parser.add_argument(
        "--abstract-context-window",
        type=int,
        default=None,
        help="Maximum context window for abstract generation. Enables preflight truncation when set with --abstract-processor-path.",
    )
    parser.add_argument(
        "--abstract-output-tokens",
        type=int,
        default=4096,
        help="max_tokens for abstract generation and the output budget reserved during preflight.",
    )
    parser.add_argument(
        "--abstract-safety-margin",
        type=int,
        default=2048,
        help="Extra token margin reserved below the configured abstract context window.",
    )
    parser.add_argument(
        "--semantic-device",
        default="auto",
        help="Device preference for semantic MaxSim, e.g. auto, cpu, cuda:1, or cuda:1,cuda:0,cpu.",
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--mineru-root", default=None)
    parser.add_argument("--png-root", default=None)
    parser.add_argument("--graph-root", default=None)
    parser.add_argument("--node-embedding-root", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    max_pages = args.max_pages
    if max_pages is None and args.benchmark == "mmlongbench":
        max_pages = 120
    config = EvidenceGraphBuildConfig(
        semantic_k=args.semantic_k,
        layout_k=args.layout_k,
        skip_llm=args.skip_llm,
        skip_embeddings=args.skip_embeddings,
        overwrite=args.overwrite,
        overwrite_llm_abstracts=args.overwrite_llm_abstracts,
        model=args.model,
        vllm_url=args.vllm_url,
        litellm_base_url=args.litellm_base_url,
        semantic_device=args.semantic_device,
        max_pages=max_pages,
        abstract_processor_path=args.abstract_processor_path,
        abstract_context_window=args.abstract_context_window,
        abstract_output_tokens=args.abstract_output_tokens,
        abstract_safety_margin=args.abstract_safety_margin,
    )
    contexts = list(_contexts(args))
    if args.limit is not None:
        contexts = contexts[: args.limit]
    manifest = build_many(contexts, config, workers=args.workers)
    for record in manifest:
        print(json.dumps(record, ensure_ascii=False))


def build_many(contexts, config, workers=1):
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    warm_abstract_processor(config.abstract_processor_path if not config.skip_llm else None)
    if workers == 1 or len(contexts) <= 1:
        return [_build_one(context, config) for context in contexts]

    records = [None] * len(contexts)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_build_one, context, config): index for index, context in enumerate(contexts)}
        for future in as_completed(futures):
            index = futures[future]
            records[index] = future.result()
    return records


def _build_one(context, config):
    try:
        result = build_document_graph(context, config)
        record = result.__dict__
        logger.info("%s %s nodes=%s edges=%s", result.status, result.doc_key, result.node_count, result.edge_count)
        return record
    except Exception as exc:
        logger.exception("Failed building %s", context.doc_key)
        return {"doc_key": context.doc_key, "graph_dir": str(context.graph_dir), "status": "failed", "error": str(exc)}


def _contexts(args):
    explicit = set(args.doc_id or [])
    graph_root = Path(args.graph_root) if args.graph_root else None
    node_embedding_root = Path(args.node_embedding_root) if args.node_embedding_root else None
    if args.benchmark == "mmlongbench":
        input_path = Path(args.input_path or CODE_DIR / "benchmarks/mmlongbench/data/raw/samples.json")
        mineru_root = Path(args.mineru_root or CODE_DIR / "benchmarks/mmlongbench/data/processed/pdfs_mineru")
        png_root = Path(args.png_root or CODE_DIR / "benchmarks/mmlongbench/data/processed/pdf_pngs")
        doc_ids = sorted({sample["doc_id"] for sample in load_mmlongbench_samples(input_path)})
        for doc_id in doc_ids:
            doc_key = mmlongbench_file_id(doc_id)
            if explicit and str(doc_id) not in explicit and doc_key not in explicit:
                continue
            yield mmlongbench_context(
                code_dir=CODE_DIR,
                doc_id=doc_id,
                mineru_root=mineru_root,
                png_root=png_root,
                graph_root=graph_root,
                node_embedding_root=node_embedding_root,
            )
    else:
        input_path = Path(args.input_path or CODE_DIR / "benchmarks/longdocurl/data/raw/LongDocURL.jsonl")
        mineru_root = Path(args.mineru_root or CODE_DIR / f"benchmarks/longdocurl/data/processed/pdfs_mineru/{args.shard}")
        png_root = Path(args.png_root or CODE_DIR / f"benchmarks/longdocurl/data/processed/pdf_pngs/{args.shard}")
        doc_ids = sorted({sample["doc_no"] for sample in load_longdocurl_samples(input_path)})
        for doc_no in doc_ids:
            if explicit and str(doc_no) not in explicit:
                continue
            yield longdocurl_context(
                code_dir=CODE_DIR,
                doc_no=doc_no,
                mineru_root=mineru_root,
                png_root=png_root,
                shard=args.shard,
                graph_root=graph_root,
                node_embedding_root=node_embedding_root,
            )


if __name__ == "__main__":
    main()
