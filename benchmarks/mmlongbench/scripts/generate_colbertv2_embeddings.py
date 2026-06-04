#!/usr/bin/env python3
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_DIR))

from benchmarks.utils.embedding_cache_cli import run_embedding_cache_cli

DEFAULTS = {
    "description": "Generate ColBERTv2 chunk/query embeddings for MMLongBench.",
    "input_path": CODE_DIR / "benchmarks" / "mmlongbench" / "data" / "raw" / "samples.json",
    "doc_output_dir": CODE_DIR / "benchmarks" / "mmlongbench" / "tmp" / "colbertv2" / "doc_embeddings",
    "query_output_dir": CODE_DIR / "benchmarks" / "mmlongbench" / "tmp" / "colbertv2" / "query_embeddings",
    "metadata_output_dir": CODE_DIR / "benchmarks" / "mmlongbench" / "tmp" / "colbertv2" / "chunk_metadata",
    "checkpoint": "/root/autodl-tmp/ylz/models/colbertv2.0",
    "tmp_dir": CODE_DIR / "benchmarks" / "mmlongbench" / "tmp",
    "ocr_json_dir": CODE_DIR / "benchmarks" / "mmlongbench" / "tmp" / "pdf_jsons",
    "max_pages": 120,
}


if __name__ == "__main__":
    run_embedding_cache_cli(baseline_name="colbertv2", benchmark_name="mmlongbench", defaults=DEFAULTS)
