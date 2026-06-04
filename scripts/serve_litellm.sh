#!/usr/bin/env bash

set -euo pipefail

LITELLM_BIN="${LITELLM_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/litellm}"
CONFIG_PATH="${1:-configs/litellm_config.yaml}"
PORT="${2:-4000}"
LITELLM_DEBUG="${LITELLM_DEBUG:-0}"
LITELLM_PROXY_URL="${LITELLM_PROXY_URL:-http://127.0.0.1:7890}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${CODE_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CODE_ROOT}/.env"
  set +a
fi

export HTTP_PROXY="${LITELLM_PROXY_URL}"
export HTTPS_PROXY="${LITELLM_PROXY_URL}"
export http_proxy="${LITELLM_PROXY_URL}"
export https_proxy="${LITELLM_PROXY_URL}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,117.156.131.215"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,117.156.131.215"

cd "${CODE_ROOT}"
ARGS=(--config "${CONFIG_PATH}" --port "${PORT}")

if [[ "${LITELLM_DEBUG}" == "1" ]]; then
  ARGS+=(--debug)
fi

exec "${LITELLM_BIN}" "${ARGS[@]}"
