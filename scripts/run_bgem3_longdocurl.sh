#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-full}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/ylz/models/bge-m3}"
TOKENIZER_NAME="${TOKENIZER_NAME:-/root/autodl-tmp/ylz/models/bge-m3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOP_K="${TOP_K:-5}"
CHUNK_SIZE="${CHUNK_SIZE:-200}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-20}"
ALLOW_CROSS_PAGE="${ALLOW_CROSS_PAGE:-true}"
MAX_CROSS_PAGES="${MAX_CROSS_PAGES:-null}"
PROCESS_MODE="${PROCESS_MODE:-parallel}"
WORKERS="${WORKERS:-64}"
LOGGING_LEVEL="${LOGGING_LEVEL:-INFO}"
MODE_NAME="${MODE_NAME:-dense}"
TEXT_SOURCE="${TEXT_SOURCE:-ocr}"
USE_FP16="${USE_FP16:-true}"
MINERU_DIR="${MINERU_DIR:-/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/pdfs_mineru/4000-4999}"

cd "${CODE_ROOT}"

INPUT_PATH="benchmarks/longdocurl/data/LongDocURL.jsonl"
if [[ "${MODE}" == "debug20" ]]; then
  INPUT_PATH="benchmarks/longdocurl/tmp/bgem3/debug_inputs/longdocurl_bgem3_debug20.jsonl"
  mkdir -p "$(dirname "${INPUT_PATH}")"
  python - <<'PY'
src = "benchmarks/longdocurl/data/LongDocURL.jsonl"
dst = "benchmarks/longdocurl/tmp/bgem3/debug_inputs/longdocurl_bgem3_debug20.jsonl"
with open(src, "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(lines[:20])
print(dst)
PY
fi

GEN_ARGS=(
  --mode both
  --input-path "${INPUT_PATH}"
  --checkpoint "${CHECKPOINT}"
  --tokenizer-name "${TOKENIZER_NAME}"
  --batch-size "${BATCH_SIZE}"
  --chunk-size "${CHUNK_SIZE}"
  --chunk-overlap "${CHUNK_OVERLAP}"
  --mode-name "${MODE_NAME}"
  --text-source "${TEXT_SOURCE}"
  --overwrite
)

if [[ "${ALLOW_CROSS_PAGE}" == "true" ]]; then
  GEN_ARGS+=(--allow-cross-page)
else
  GEN_ARGS+=(--no-allow-cross-page)
fi

if [[ "${MAX_CROSS_PAGES}" != "null" && "${MAX_CROSS_PAGES}" != "None" && -n "${MAX_CROSS_PAGES}" ]]; then
  GEN_ARGS+=(--max-cross-pages "${MAX_CROSS_PAGES}")
fi

if [[ "${USE_FP16}" == "true" ]]; then
  GEN_ARGS+=(--use-fp16)
else
  GEN_ARGS+=(--no-use-fp16)
fi

python benchmarks/longdocurl/scripts/generate_bgem3_embeddings.py "${GEN_ARGS[@]}"

python main.py \
  benchmarks=longdocurl \
  baselines=bgem3 \
  "benchmarks.qa_file=${INPUT_PATH}" \
  "benchmarks.mineru_dir=${MINERU_DIR}" \
  "benchmarks.process_mode=${PROCESS_MODE}" \
  "benchmarks.workers=${WORKERS}" \
  "logging.level=${LOGGING_LEVEL}" \
  "baselines.params.mode=${MODE_NAME}" \
  "baselines.params.text_source=${TEXT_SOURCE}" \
  "baselines.params.top_k=${TOP_K}" \
  "baselines.params.chunk_size=${CHUNK_SIZE}" \
  "baselines.params.chunk_overlap=${CHUNK_OVERLAP}" \
  "baselines.params.allow_cross_page=${ALLOW_CROSS_PAGE}" \
  "baselines.params.max_cross_pages=${MAX_CROSS_PAGES}" \
  "baselines.checkpoint=${CHECKPOINT}" \
  "baselines.tokenizer_name=${TOKENIZER_NAME}" \
  "baselines.batch_size=${BATCH_SIZE}" \
  "baselines.use_fp16=${USE_FP16}"
