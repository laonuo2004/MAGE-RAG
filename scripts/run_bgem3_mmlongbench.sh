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

cd "${CODE_ROOT}"

INPUT_PATH="benchmarks/mmlongbench/data/samples.json"
if [[ "${MODE}" == "debug20" ]]; then
  INPUT_PATH="benchmarks/mmlongbench/tmp/bgem3/debug_inputs/mmlongbench_bgem3_debug20.json"
  mkdir -p "$(dirname "${INPUT_PATH}")"
  python - <<'PY'
import json
src = "benchmarks/mmlongbench/data/samples.json"
dst = "benchmarks/mmlongbench/tmp/bgem3/debug_inputs/mmlongbench_bgem3_debug20.json"
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

python benchmarks/mmlongbench/scripts/generate_bgem3_embeddings.py "${GEN_ARGS[@]}"

python main.py \
  benchmarks=mmlongbench \
  baselines=bgem3 \
  "benchmarks.input_path=${INPUT_PATH}" \
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
