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

from benchmarks.evidence_graph.paths import longdocurl_context, mmlongbench_context
from benchmarks.utils.data_utils import load_longdocurl_samples, load_mmlongbench_samples, mmlongbench_file_id
from benchmarks.utils.mineru_text import build_mineru_text_pages
from utils.llm_utils import build_openai_client

logger = logging.getLogger("build_mineru_text_pages")


def parse_args():
    parser = argparse.ArgumentParser(description="Build page-level text from MinerU outputs, with optional LLM descriptions for visuals.")
    parser.add_argument("--benchmark", choices=["longdocurl", "mmlongbench"], required=True)
    parser.add_argument("--doc-id", action="append", default=None, help="Document id/doc_no to process. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", default="4000-4999")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--mineru-root", default=None)
    parser.add_argument("--output-name", default="vlm_text_pages.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--litellm-base-url", default=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = None if args.skip_llm else build_openai_client({"litellm": {"api_key": args.api_key, "base_url": args.litellm_base_url}})
    contexts = list(_contexts(args))
    if args.limit is not None:
        contexts = contexts[: args.limit]
    records = build_many(
        contexts,
        output_name=args.output_name,
        client=client,
        model_name=None if args.skip_llm else args.model,
        overwrite=args.overwrite,
        max_tokens=args.max_tokens,
        workers=args.workers,
    )
    for record in records:
        print(json.dumps(record, ensure_ascii=False))


def build_many(contexts, *, output_name, client, model_name, overwrite, max_tokens, workers=1):
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if workers == 1 or len(contexts) <= 1:
        return [_build_one(context, output_name, client, model_name, overwrite, max_tokens) for context in contexts]

    records = [None] * len(contexts)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_build_one, context, output_name, client, model_name, overwrite, max_tokens): index
            for index, context in enumerate(contexts)
        }
        for future in as_completed(futures):
            records[futures[future]] = future.result()
    return records


def _build_one(context, output_name, client, model_name, overwrite, max_tokens):
    output_path = context.mineru_dir / output_name
    try:
        artifact = build_mineru_text_pages(
            mineru_dir=context.mineru_dir,
            output_path=output_path,
            client=client,
            model_name=model_name,
            overwrite=overwrite,
            max_tokens=max_tokens,
        )
        logger.info("generated %s pages=%s", context.doc_key, len(artifact.get("pages", [])))
        return {"doc_key": context.doc_key, "output_path": str(output_path), "status": "generated"}
    except Exception as exc:
        logger.exception("Failed building MinerU text pages for %s", context.doc_key)
        return {"doc_key": context.doc_key, "output_path": str(output_path), "status": "failed", "error": str(exc)}


def _contexts(args):
    explicit = set(args.doc_id or [])
    if args.benchmark == "mmlongbench":
        input_path = Path(args.input_path or CODE_DIR / "benchmarks/mmlongbench/data/raw/samples.json")
        mineru_root = Path(args.mineru_root or CODE_DIR / "benchmarks/mmlongbench/data/processed/pdfs_mineru")
        doc_ids = sorted({sample["doc_id"] for sample in load_mmlongbench_samples(input_path)})
        for doc_id in doc_ids:
            doc_key = mmlongbench_file_id(doc_id)
            if explicit and str(doc_id) not in explicit and doc_key not in explicit:
                continue
            yield mmlongbench_context(
                code_dir=CODE_DIR,
                doc_id=doc_id,
                mineru_root=mineru_root,
                png_root=CODE_DIR / "benchmarks/mmlongbench/data/processed/pdf_pngs",
            )
        return

    input_path = Path(args.input_path or CODE_DIR / "benchmarks/longdocurl/data/raw/LongDocURL.jsonl")
    mineru_root = Path(args.mineru_root or CODE_DIR / f"benchmarks/longdocurl/data/processed/pdfs_mineru/{args.shard}")
    doc_ids = sorted({sample["doc_no"] for sample in load_longdocurl_samples(input_path)})
    for doc_no in doc_ids:
        if explicit and str(doc_no) not in explicit:
            continue
        yield longdocurl_context(
            code_dir=CODE_DIR,
            doc_no=doc_no,
            mineru_root=mineru_root,
            png_root=CODE_DIR / f"benchmarks/longdocurl/data/processed/pdf_pngs/{args.shard}",
            shard=args.shard,
        )


if __name__ == "__main__":
    main()
