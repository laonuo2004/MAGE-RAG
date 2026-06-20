#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-${PYTHONPATH:?PYTHONPATH is required}}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python}"
WORKERS="${WORKERS:-16}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-30}"
RESULTS_FILE="${RESULTS_FILE:-${CODE_ROOT}/results/longdocurl/g2_reader/full_adjust0_corr_relaxed.jsonl}"
METRICS_FILE="${RESULTS_FILE%.jsonl}.metrics.json"

log() {
  echo "[g2-longdocurl-pm2] $*" >&2
}

cleanup_stale_g2_children() {
  local pids
  pids="$(pgrep -f "python -m baselines.g2reader run-sample.*benchmarks/longdocurl/data/cache/g2_reader" || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  log "stopping stale G2 run-sample children: ${pids//$'\n'/ }"
  # shellcheck disable=SC2086
  kill ${pids} 2>/dev/null || true
  sleep 5
  pids="$(pgrep -f "python -m baselines.g2reader run-sample.*benchmarks/longdocurl/data/cache/g2_reader" || true)"
  if [[ -n "${pids}" ]]; then
    log "force-stopping stale G2 run-sample children: ${pids//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
  fi
}

completed_count() {
  if [[ ! -f "${METRICS_FILE}" ]]; then
    echo 0
    return 0
  fi
  "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_count", 0))' "${METRICS_FILE}"
}

failed_count() {
  if [[ ! -f "${METRICS_FILE}" ]]; then
    echo 1
    return 0
  fi
  "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("failed_count", 1))' "${METRICS_FILE}"
}

trap cleanup_stale_g2_children EXIT INT TERM

mkdir -p "$(dirname "${RESULTS_FILE}")"
cd "${CODE_ROOT}"

cleanup_stale_g2_children

log "starting LongDocURL G2-Reader resume run workers=${WORKERS}"
"${PYTHON_BIN}" main.py \
  benchmarks=longdocurl \
  baselines=g2-reader \
  overwrite=false \
  "benchmarks.workers=${WORKERS}" \
  baselines.params.max_adjust_rounds=0 \
  benchmarks.results_file="${RESULTS_FILE}"

completed="$(completed_count)"
failed="$(failed_count)"
log "run finished completed=${completed} failed=${failed} metrics=${METRICS_FILE}"

if [[ "${failed}" != "0" ]]; then
  log "not complete; sleeping ${RESTART_DELAY_SECONDS}s then exiting non-zero for PM2 restart"
  sleep "${RESTART_DELAY_SECONDS}"
  exit 1
fi

log "all samples complete"
