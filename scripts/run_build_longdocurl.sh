#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${CODE_ROOT}"

ABSTRACT_PROCESSOR_PATH="${ABSTRACT_PROCESSOR_PATH:-/root/autodl-tmp/ylz/models/Qwen3-VL-8B-Instruct}"
ABSTRACT_CONTEXT_WINDOW="${ABSTRACT_CONTEXT_WINDOW:-131072}"
ABSTRACT_OUTPUT_TOKENS="${ABSTRACT_OUTPUT_TOKENS:-4096}"
ABSTRACT_SAFETY_MARGIN="${ABSTRACT_SAFETY_MARGIN:-2048}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python}"

"${PYTHON_BIN}" benchmarks/scripts/build_evidence_graphs.py \
  --benchmark longdocurl \
  --workers 16 \
  --abstract-processor-path "${ABSTRACT_PROCESSOR_PATH}" \
  --abstract-context-window "${ABSTRACT_CONTEXT_WINDOW}" \
  --abstract-output-tokens "${ABSTRACT_OUTPUT_TOKENS}" \
  --abstract-safety-margin "${ABSTRACT_SAFETY_MARGIN}"
