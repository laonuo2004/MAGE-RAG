#!/usr/bin/env bash
set -euo pipefail

LOCAL_CODE_DIR="${LOCAL_CODE_DIR:-${PYTHONPATH:?PYTHONPATH is required}}"
REMOTE_CODE_DEST="${REMOTE_CODE_DEST:-ai4s:/root/autodl-tmp/ylz/NeurIPS_2026/code/}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-10}"

RSYNC_EXCLUDES=(
  "--exclude=__pycache__/"
  "--exclude=.pytest_cache/"
  "--exclude=outputs/"
  "--exclude=results/"
  "--exclude=logs/"
  "--exclude=analysis_cache/"
  "--exclude=postgresql/"
  "--exclude=benchmarks/longdocurl/data/"
  "--exclude=benchmarks/mmlongbench/data/"
  "--exclude=flash_attn-*.whl"
  "--exclude=*.pyc"
)

while true; do
  echo "[$(date -Is)] syncing code/config to ${REMOTE_CODE_DEST}"
  rsync -az --delete --human-readable --itemize-changes \
    "${RSYNC_EXCLUDES[@]}" \
    "${LOCAL_CODE_DIR%/}/" \
    "${REMOTE_CODE_DEST}"
  sleep "${SYNC_INTERVAL_SECONDS}"
done
