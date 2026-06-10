#!/usr/bin/env bash
set -euo pipefail

LOCAL_CODE_DIR="${LOCAL_CODE_DIR:-/root/autodl-tmp/ylz/NeurIPS_2026/code}"
REMOTE_CODE_DEST="${REMOTE_CODE_DEST:-MM:/root/autodl-tmp/ylz/NeurIPS_2026/code}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-10}"

while true; do
  echo "[$(date -Is)] syncing ai4s results/outputs to ${REMOTE_CODE_DEST}"
  mkdir -p \
    "${LOCAL_CODE_DIR%/}/results/longdocurl/magerag" \
    "${LOCAL_CODE_DIR%/}/outputs"
  ssh MM "mkdir -p '${REMOTE_CODE_DEST#MM:}/results/longdocurl/magerag' '${REMOTE_CODE_DEST#MM:}/outputs'"
  rsync -az --human-readable --itemize-changes \
    "${LOCAL_CODE_DIR%/}/results/longdocurl/magerag/" \
    "${REMOTE_CODE_DEST%/}/results/longdocurl/magerag/"
  rsync -az --human-readable --itemize-changes \
    "${LOCAL_CODE_DIR%/}/outputs/" \
    "${REMOTE_CODE_DEST%/}/outputs/"
  sleep "${SYNC_INTERVAL_SECONDS}"
done
