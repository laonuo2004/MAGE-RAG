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

INPUT_PATH="benchmarks/mmlongbench/data/raw/samples.json"
if [[ "${MODE}" == "debug20" ]]; then
  INPUT_PATH="benchmarks/mmlongbench/data/cache/colbertv2/debug_inputs/mmlongbench_colbert_debug20.json"
  mkdir -p "$(dirname "${INPUT_PATH}")"
  python - <<'PY'
import json
src = "benchmarks/mmlongbench/data/raw/samples.json"
dst = "benchmarks/mmlongbench/data/cache/colbertv2/debug_inputs/mmlongbench_colbert_debug20.json"
with open(src, "r", encoding="utf-8") as f:
    samples = json.load(f)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(samples[:20], f, ensure_ascii=False, indent=2)
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

python benchmarks/scripts/generate_colbertv2_embeddings.py --benchmark mmlongbench "${GEN_ARGS[@]}"

python scripts/verify_colbertv2_cache.py \
  --benchmark mmlongbench \
  --input-path "${INPUT_PATH}" \
  --doc-embeddings-root benchmarks/mmlongbench/data/cache/colbertv2/doc_embeddings \
  --query-embeddings-root benchmarks/mmlongbench/data/cache/colbertv2/query_embeddings \
  --chunk-metadata-root benchmarks/mmlongbench/data/cache/colbertv2/chunk_metadata \
  --checkpoint "${CHECKPOINT}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-overlap "${CHUNK_OVERLAP}" \
  --allow-cross-page "${ALLOW_CROSS_PAGE}" \
  --max-cross-pages "${MAX_CROSS_PAGES}"

MAIN_ARGS=(
  benchmarks=mmlongbench
  baselines=colbertv2
  "benchmarks.input_path=${INPUT_PATH}"
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
