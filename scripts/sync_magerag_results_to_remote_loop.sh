#!/usr/bin/env bash
set -euo pipefail

LOCAL_CODE_DIR="${LOCAL_CODE_DIR:-${PYTHONPATH:?PYTHONPATH is required}}"
REMOTE_CODE_DEST="${REMOTE_CODE_DEST:?REMOTE_CODE_DEST is required, for example ai4s:/root/autodl-tmp/ylz/NeurIPS_2026/code}"
REMOTE_SSH_HOST="${REMOTE_SSH_HOST:?REMOTE_SSH_HOST is required, for example ai4s or MM}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-10}"
RESULTS_RELATIVE_DIR="${RESULTS_RELATIVE_DIR:-results/longdocurl/magerag}"
RSYNC_TEMP_DIR="${RSYNC_TEMP_DIR:-/root/autodl-tmp/.tmp/rsync-magerag-results}"

local_results_dir="${LOCAL_CODE_DIR%/}/${RESULTS_RELATIVE_DIR}"
remote_code_dir="${REMOTE_CODE_DEST#*:}"
remote_results_dir="${remote_code_dir%/}/${RESULTS_RELATIVE_DIR}"

while true; do
  echo "[$(date -Is)] syncing ${local_results_dir}/ to ${REMOTE_CODE_DEST%/}/${RESULTS_RELATIVE_DIR}/"
  mkdir -p "${local_results_dir}"
  ssh "${REMOTE_SSH_HOST}" "mkdir -p '${remote_results_dir}' '${RSYNC_TEMP_DIR}'"
  rsync -az --update --temp-dir="${RSYNC_TEMP_DIR}" --exclude='.res_*' --human-readable --itemize-changes \
    "${local_results_dir}/" \
    "${REMOTE_CODE_DEST%/}/${RESULTS_RELATIVE_DIR}/"
  sleep "${SYNC_INTERVAL_SECONDS}"
done
