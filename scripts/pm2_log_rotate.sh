#!/usr/bin/env bash

set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/root/autodl-tmp/ylz/NeurIPS_2026/code}"
LOG_DIR="${LOG_DIR:-${CODE_ROOT}/logs}"
MAX_SIZE_MB="${MAX_SIZE_MB:-100}"
KEEP="${KEEP:-5}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

max_size_bytes=$((MAX_SIZE_MB * 1024 * 1024))

rotate_one() {
  local log_file="$1"
  [[ -f "${log_file}" ]] || return 0

  local size
  size="$(stat -c '%s' "${log_file}")"
  (( size >= max_size_bytes )) || return 0

  local ts rotated
  ts="$(date +%Y%m%d-%H%M%S)"
  rotated="${log_file}.${ts}"

  mv "${log_file}" "${rotated}"
  : > "${log_file}"
  pm2 reloadLogs >/dev/null 2>&1 || true
  gzip -f "${rotated}" || true

  local base
  base="$(basename "${log_file}")"
  find "${LOG_DIR}" -maxdepth 1 -type f -name "${base}.*.gz" \
    | sort -r \
    | awk -v keep="${KEEP}" 'NR > keep { print }' \
    | xargs -r rm -f

  printf '%s rotated %s (%s bytes)\n' "$(date -Is)" "${log_file}" "${size}"
}

run_once() {
  mkdir -p "${LOG_DIR}"
  shopt -s nullglob
  local log_file
  for log_file in "${LOG_DIR}"/pm2-*.log; do
    rotate_one "${log_file}"
  done
}

if [[ "${1:-}" == "--once" ]]; then
  run_once
  exit 0
fi

while true; do
  run_once
  sleep "${INTERVAL_SECONDS}"
done
