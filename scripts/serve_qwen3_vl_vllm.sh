#!/usr/bin/env bash

# ### Serving Qwen3-VL-8B with vLLM

# For local OpenAI-compatible serving on a single GPU, this repo provides three presets for `Qwen/Qwen3-VL-8B-Instruct`:

# ```bash
# # Throughput-oriented default profile.
# bash scripts/serve_qwen3_vl_vllm.sh throughput

# # Long-context profile for 256k token requests.
# bash scripts/serve_qwen3_vl_vllm.sh longctx

# # Maximum-context profile for up to 1M token requests.
# # Note: 1M context requires massive VRAM or specific optimizations.
# bash scripts/serve_qwen3_vl_vllm.sh maxctx
# ```

# Notes:
# - `throughput` uses `--max-model-len 32768`.
# - `longctx` uses `--max-model-len 262144` (256K).
# - `maxctx` uses `--max-model-len 1048576` (1M).
# - All presets enable chunked prefill.
# - Override GPU or port when needed, for example: `CUDA_VISIBLE_DEVICES=1 PORT=8001 bash scripts/serve_qwen3_vl_vllm.sh longctx`.

set -euo pipefail

PROFILE="${1:-throughput}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-1}"
VLLM_BIN="${VLLM_BIN:-/root/autodl-tmp/conda/envs/logma-rag/bin/vllm}"
MODEL_NAME="${MODEL_NAME:-/root/autodl-tmp/ylz/models/Qwen3-VL-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
LIMIT_MM_PER_PROMPT_VIDEO="${LIMIT_MM_PER_PROMPT_VIDEO:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

COMMON_ARGS=(
  serve
  "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
)

case "${PROFILE}" in
  throughput)
    DEFAULT_GPU_MEMORY_UTILIZATION="0.6"
    DEFAULT_MAX_MODEL_LEN="16384"
    DEFAULT_MAX_NUM_SEQS="64"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="32768"
    ;;
  longctx)
    DEFAULT_GPU_MEMORY_UTILIZATION="0.6"
    DEFAULT_MAX_MODEL_LEN="131072"
    DEFAULT_MAX_NUM_SEQS="32"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="16384"
    ;;
  maxctx)
    DEFAULT_GPU_MEMORY_UTILIZATION="0.8"
    DEFAULT_MAX_MODEL_LEN="262144"
    DEFAULT_MAX_NUM_SEQS="16"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="16384"
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    echo "Usage: $0 [throughput|longctx|maxctx]" >&2
    exit 1
    ;;
esac

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEMORY_UTILIZATION}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${DEFAULT_MAX_NUM_BATCHED_TOKENS}}"

PROFILE_ARGS=(
  --served-model-name Qwen3-VL-8B-Instruct
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}"
  --limit-mm-per-prompt.video "${LIMIT_MM_PER_PROMPT_VIDEO}"
)

if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  PROFILE_ARGS+=(--enable-chunked-prefill)
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  PROFILE_ARGS+=(--trust-remote-code)
fi

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  PROFILE_ARGS+=(--enable-prefix-caching)
fi

if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
  PROFILE_ARGS+=(--async-scheduling)
fi

if [[ -n "${LIMIT_MM_PER_PROMPT}" ]]; then
  PROFILE_ARGS+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}")
fi

if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=( ${EXTRA_VLLM_ARGS} )
  PROFILE_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Starting vLLM with profile=${PROFILE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}, GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}, port=${PORT}" >&2
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  "${VLLM_BIN}" \
  "${COMMON_ARGS[@]}" \
  "${PROFILE_ARGS[@]}"
