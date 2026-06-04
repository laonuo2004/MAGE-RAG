#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${CODE_ROOT}"
exec "${PYTHON_BIN}" main.py --multirun baselines=aeg-rag \
    benchmarks=longdocurl,mmlongbench \
    baselines.agent.run_online=true \
    baselines.agent.initial_retrieval_top_k=5 \
    baselines.agent.initial_retrieval_top_k_longdocurl=5 \
    baselines.agent.initial_retrieval_top_k_mmlongbench=5 \
    benchmarks.workers=64
