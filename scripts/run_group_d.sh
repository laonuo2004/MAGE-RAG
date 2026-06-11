#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${CODE_ROOT}"
exec "${PYTHON_BIN}" main.py --multirun baselines=magerag \
    benchmarks=mmlongbench \
    baselines.params.top_k=3 \
    baselines.controller.watchdog_iterations=10 \
    baselines.evaluator.max_selected_actions_per_iteration=5 \
    benchmarks.correction_enabled=true \
    baselines.graph.mode=page_only,containment_only,structural_graph,semantic_graph,layout_graph \
    benchmarks.workers=256