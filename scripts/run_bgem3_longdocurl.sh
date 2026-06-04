#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-full}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/ylz/models/bge-m3}"
TOKENIZER_NAME="${TOKENIZER_NAME:-/root/autodl-tmp/ylz/models/bge-m3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
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
MINERU_DIR="${MINERU_DIR:-/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/processed/pdfs_mineru/4000-4999}"

cd "${CODE_ROOT}"

INPUT_PATH="benchmarks/longdocurl/data/raw/LongDocURL.jsonl"
if [[ "${MODE}" == "debug20" ]]; then
  INPUT_PATH="benchmarks/longdocurl/data/cache/bgem3/debug_inputs/longdocurl_bgem3_debug20.jsonl"
  mkdir -p "$(dirname "${INPUT_PATH}")"
  python - <<'PY'
import os
src = "benchmarks/longdocurl/data/raw/LongDocURL.jsonl"
dst = "benchmarks/longdocurl/data/cache/bgem3/debug_inputs/longdocurl_bgem3_debug20.jsonl"
text_source = os.environ.get("TEXT_SOURCE", "ocr")
mineru_dir = os.environ.get(
    "MINERU_DIR",
    "/root/autodl-tmp/ylz/NeurIPS_2026/code/benchmarks/longdocurl/data/processed/pdfs_mineru/4000-4999",
)
selected = []
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        if text_source == "vlm_text":
            import json
            sample = json.loads(line)
            doc_dir = os.path.join(mineru_dir, str(sample["doc_no"]))
            if not (
                os.path.exists(os.path.join(doc_dir, "content_list_v2.json"))
                or os.path.exists(os.path.join(doc_dir, "full.md"))
            ):
                continue
        selected.append(line)
        if len(selected) >= 20:
            break
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(selected)
print(dst)
PY
fi

GEN_ARGS=(
  --mode both
  --input-path "${INPUT_PATH}"
  --checkpoint "${CHECKPOINT}"
  --tokenizer-name "${TOKENIZER_NAME}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
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

python benchmarks/scripts/generate_bgem3_embeddings.py --benchmark longdocurl "${GEN_ARGS[@]}"

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
  "baselines.max_length=${MAX_LENGTH}" \
  "baselines.use_fp16=${USE_FP16}"
