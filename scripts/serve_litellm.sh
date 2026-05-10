#!/usr/bin/env bash

set -euo pipefail

LITELLM_BIN="${LITELLM_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/litellm}"
CONFIG_PATH="${1:-configs/litellm_config.yaml}"
PORT="${2:-4000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${CODE_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CODE_ROOT}/.env"
  set +a
fi

cd "${CODE_ROOT}"
exec "${LITELLM_BIN}" --config "${CONFIG_PATH}" --port "${PORT}" --debug
