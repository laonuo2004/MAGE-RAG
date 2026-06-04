#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-throughput}"
SERVE_SCRIPT="${VLLM_SERVE_SCRIPT:-scripts/serve_qwen3_vl_vllm.sh}"
STOP_GRACE_SECONDS="${VLLM_STOP_GRACE_SECONDS:-20}"
CLEAN_STALE_ON_START="${CLEAN_STALE_ON_START:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

VLLM_PID=""

log() {
  echo "[pm2-vllm-wrapper] $*" >&2
}

cleanup_process_group() {
  local pid="${1:-}"

  if [[ -z "${pid}" ]]; then
    return 0
  fi

  log "stopping vLLM process group pgid=${pid}"
  kill -TERM -"${pid}" 2>/dev/null || true
  sleep "${STOP_GRACE_SECONDS}"
  kill -KILL -"${pid}" 2>/dev/null || true
}

cleanup_pid() {
  local pid="${1:-}"

  if [[ -z "${pid}" ]]; then
    return 0
  fi

  log "stopping stale vLLM pid=${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  sleep "${STOP_GRACE_SECONDS}"
  kill -KILL "${pid}" 2>/dev/null || true
}

cleanup() {
  cleanup_process_group "${VLLM_PID}"
}

cleanup_stale_vllm() {
  local port="${PORT:-8000}"
  local model_name="${MODEL_NAME:-Qwen3-VL-8B-Instruct}"
  local pattern
  local pid
  local pgid
  local ppid

  pattern="vllm serve .*(${model_name}|--port ${port}|--port=${port})"
  log "checking stale vLLM processes with pattern: ${pattern}"
  pgrep -af "${pattern}" || true

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -n "${pgid}" ]]; then
      cleanup_process_group "${pgid}"
    else
      cleanup_pid "${pid}"
    fi
  done < <(pgrep -f "${pattern}" || true)

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    ppid="$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "${ppid}" == "1" ]]; then
      cleanup_pid "${pid}"
    fi
  done < <(pgrep -f "VLLM::EngineCore" || true)
}

trap cleanup EXIT INT TERM

if [[ "${CLEAN_STALE_ON_START}" == "1" ]]; then
  cleanup_stale_vllm
fi

log "starting ${SERVE_SCRIPT} profile=${PROFILE}"
setsid env \
  -u VLLM_SERVE_SCRIPT \
  -u VLLM_STOP_GRACE_SECONDS \
  -u CLEAN_STALE_ON_START \
  bash "${SERVE_SCRIPT}" "${PROFILE}" &
VLLM_PID=$!

wait "${VLLM_PID}"
