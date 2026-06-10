#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOP_K="${TOP_K:-3}"
MAX_IMAGES="${MAX_IMAGES:-3}"
PROMPT_MODE="${PROMPT_MODE:-evidence_grpo}"
LONGDOCURL_SHARD="${LONGDOCURL_SHARD:-4000-4999}"
WORKERS="${WORKERS:-64}"
OVERWRITE="${OVERWRITE:-false}"
LOGGING_LEVEL="${LOGGING_LEVEL:-INFO}"

cd "${CODE_ROOT}"
exec "${PYTHON_BIN}" main.py --multirun \
  benchmarks=longdocurl,mmlongbench \
  baselines=evisrag \
  "benchmarks.workers=${WORKERS}" \
  "baselines.params.top_k=${TOP_K}" \
  "baselines.params.max_images=${MAX_IMAGES}" \
  "baselines.params.prompt_mode=${PROMPT_MODE}" \
  "baselines.params.longdocurl_shard=${LONGDOCURL_SHARD}" \
  "overwrite=${OVERWRITE}" \
  "logging.level=${LOGGING_LEVEL}"
