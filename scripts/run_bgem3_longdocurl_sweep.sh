#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-full}"
TOP_K_LIST="${TOP_K_LIST:-3 5}"
CHUNK_SIZE_LIST="${CHUNK_SIZE_LIST:-150 200}"
CHUNK_OVERLAP_LIST="${CHUNK_OVERLAP_LIST:-0 20}"
ALLOW_CROSS_PAGE_LIST="${ALLOW_CROSS_PAGE_LIST:-true}"
MAX_CROSS_PAGES_LIST="${MAX_CROSS_PAGES_LIST:-null}"

cd "${CODE_ROOT}"

for TOP_K in ${TOP_K_LIST}; do
  for CHUNK_SIZE in ${CHUNK_SIZE_LIST}; do
    for CHUNK_OVERLAP in ${CHUNK_OVERLAP_LIST}; do
      for ALLOW_CROSS_PAGE in ${ALLOW_CROSS_PAGE_LIST}; do
        for MAX_CROSS_PAGES in ${MAX_CROSS_PAGES_LIST}; do
          echo "=== LongDocURL BGEM3 sweep: top_k=${TOP_K} chunk_size=${CHUNK_SIZE} chunk_overlap=${CHUNK_OVERLAP} allow_cross_page=${ALLOW_CROSS_PAGE} max_cross_pages=${MAX_CROSS_PAGES} ==="
          TOP_K="${TOP_K}" \
          CHUNK_SIZE="${CHUNK_SIZE}" \
          CHUNK_OVERLAP="${CHUNK_OVERLAP}" \
          ALLOW_CROSS_PAGE="${ALLOW_CROSS_PAGE}" \
          MAX_CROSS_PAGES="${MAX_CROSS_PAGES}" \
          bash scripts/run_bgem3_longdocurl.sh "${MODE}"
        done
      done
    done
  done
done
