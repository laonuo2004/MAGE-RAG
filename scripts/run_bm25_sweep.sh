#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${CODE_ROOT}"

python main.py --multirun \
  benchmarks=longdocurl,mmlongbench \
  baselines=bm25 \
  benchmarks.workers=128 \
  baselines.top_k=3,5 \
  baselines.chunk_size=150,200 \
  baselines.chunk_overlap=0,20,50 \
  baselines.max_chunks_per_page=null
