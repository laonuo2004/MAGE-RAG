#!/usr/bin/env bash

set -euo pipefail

LITELLM_BIN="${LITELLM_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/litellm}"
CONFIG_PATH="${1:-configs/litellm_config.yaml}"
PORT="${2:-4000}"
LITELLM_DEBUG="${LITELLM_DEBUG:-0}"
LITELLM_PROXY_URL="${LITELLM_PROXY_URL:-http://127.0.0.1:7890}"
LITELLM_SCHEMA_PATH="${LITELLM_SCHEMA_PATH:-/root/autodl-tmp/conda/envs/logma-rag-py12/lib/python3.12/site-packages/litellm/proxy/schema.prisma}"
PRISMA_QUERY_ENGINE_BINARY="${PRISMA_QUERY_ENGINE_BINARY:-/root/.cache/prisma-python/binaries/5.17.0/393aa359c9ad4a4bb28630fb5613f9c281cde053/node_modules/@prisma/engines/query-engine-debian-openssl-3.0.x}"

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
export DISABLE_SCHEMA_UPDATE="${DISABLE_SCHEMA_UPDATE:-true}"
export PRISMA_QUERY_ENGINE_BINARY

cd "${CODE_ROOT}"

if [[ -f "${LITELLM_SCHEMA_PATH}" && ! -e "${CODE_ROOT}/schema.prisma" ]]; then
  ln -s "${LITELLM_SCHEMA_PATH}" "${CODE_ROOT}/schema.prisma"
fi

ARGS=(--config "${CONFIG_PATH}" --port "${PORT}")

if [[ "${LITELLM_DEBUG}" == "1" ]]; then
  ARGS+=(--debug)
fi

exec "${LITELLM_BIN}" "${ARGS[@]}"
