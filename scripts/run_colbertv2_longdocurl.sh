#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-full}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/ylz/models/colbertv2.0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOP_K="${TOP_K:-5}"
CHUNK_SIZE="${CHUNK_SIZE:-200}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-20}"
ALLOW_CROSS_PAGE="${ALLOW_CROSS_PAGE:-true}"
MAX_CROSS_PAGES="${MAX_CROSS_PAGES:-null}"
PROCESS_MODE="${PROCESS_MODE:-parallel}"
WORKERS="${WORKERS:-64}"
LOGGING_LEVEL="${LOGGING_LEVEL:-INFO}"

cd "${CODE_ROOT}"

INPUT_PATH="benchmarks/longdocurl/data/LongDocURL.jsonl"
if [[ "${MODE}" == "debug20" ]]; then
  INPUT_PATH="benchmarks/longdocurl/tmp/colbertv2/debug_inputs/longdocurl_colbert_debug20.jsonl"
  mkdir -p "$(dirname "${INPUT_PATH}")"
  python - <<'PY'
src = "benchmarks/longdocurl/data/LongDocURL.jsonl"
dst = "benchmarks/longdocurl/tmp/colbertv2/debug_inputs/longdocurl_colbert_debug20.jsonl"
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
  --batch-size "${BATCH_SIZE}"
  --checkpoint "${CHECKPOINT}"
  --chunk-size "${CHUNK_SIZE}"
  --chunk-overlap "${CHUNK_OVERLAP}"
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

python benchmarks/longdocurl/scripts/generate_colbertv2_embeddings.py "${GEN_ARGS[@]}"

python scripts/verify_colbertv2_cache.py \
  --benchmark longdocurl \
  --input-path "${INPUT_PATH}" \
  --doc-embeddings-root benchmarks/longdocurl/tmp/colbertv2/doc_embeddings \
  --query-embeddings-root benchmarks/longdocurl/tmp/colbertv2/query_embeddings \
  --chunk-metadata-root benchmarks/longdocurl/tmp/colbertv2/chunk_metadata \
  --checkpoint "${CHECKPOINT}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-overlap "${CHUNK_OVERLAP}" \
  --allow-cross-page "${ALLOW_CROSS_PAGE}" \
  --max-cross-pages "${MAX_CROSS_PAGES}"

MAIN_ARGS=(
  benchmarks=longdocurl
  baselines=colbertv2
  "benchmarks.qa_file=${INPUT_PATH}"
  "benchmarks.process_mode=${PROCESS_MODE}"
  "benchmarks.workers=${WORKERS}"
  "logging.level=${LOGGING_LEVEL}"
  "baselines.params.top_k=${TOP_K}"
  "baselines.params.chunk_size=${CHUNK_SIZE}"
  "baselines.params.chunk_overlap=${CHUNK_OVERLAP}"
  "baselines.params.allow_cross_page=${ALLOW_CROSS_PAGE}"
)

if [[ "${MAX_CROSS_PAGES}" == "null" || "${MAX_CROSS_PAGES}" == "None" || -z "${MAX_CROSS_PAGES}" ]]; then
  MAIN_ARGS+=("baselines.params.max_cross_pages=null")
else
  MAIN_ARGS+=("baselines.params.max_cross_pages=${MAX_CROSS_PAGES}")
fi

python main.py "${MAIN_ARGS[@]}"
