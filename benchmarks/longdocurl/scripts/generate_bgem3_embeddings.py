#!/usr/bin/env python3
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIR))

from benchmarks.utils.embedding_cache_cli import run_embedding_cache_cli

DEFAULTS = {
    "description": "Generate BGEM3 dense chunk/query embeddings for LongDocURL.",
    "input_path": CODE_DIR / "benchmarks" / "longdocurl" / "data" / "raw" / "LongDocURL.jsonl",
    "doc_output_dir": CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "bgem3" / "doc_embeddings",
    "query_output_dir": CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "bgem3" / "query_embeddings",
    "metadata_output_dir": CODE_DIR / "benchmarks" / "longdocurl" / "tmp" / "bgem3" / "chunk_metadata",
    "checkpoint": "/root/autodl-tmp/ylz/models/bge-m3",
    "ocr_json_dir": CODE_DIR / "benchmarks" / "longdocurl" / "data" / "pdf_jsons" / "4000-4999",
    "image_prefix": CODE_DIR / "benchmarks" / "longdocurl" / "data" / "processed" / "pdf_pngs" / "4000-4999",
    "mineru_dir": CODE_DIR / "benchmarks" / "longdocurl" / "data" / "processed" / "pdfs_mineru" / "4000-4999",
}


if __name__ == "__main__":
    run_embedding_cache_cli(baseline_name="bgem3", benchmark_name="longdocurl", defaults=DEFAULTS)
