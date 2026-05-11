#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${CODE_ROOT}"

python main.py --multirun \
  benchmarks=longdocurl,mmlongbench \
  baselines=bm25 \
  benchmarks.workers=64 \
  baselines.params.top_k=1,2,3,4,5 \
  baselines.params.chunk_size=100,150,200,250,300 \
  baselines.params.chunk_overlap=0,10,20,30,40,50 \
  baselines.max_chunks_per_page=null
