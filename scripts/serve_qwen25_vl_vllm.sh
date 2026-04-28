#!/usr/bin/env bash

# ### Serving Qwen2.5-VL-7B with vLLM

# For local OpenAI-compatible serving on a single GPU, this repo provides three presets for `Qwen/Qwen2.5-VL-7B-Instruct`:

# ```bash
# # Throughput-oriented default profile.
# bash scripts/serve_qwen25_vl_vllm.sh throughput

# # Long-context profile for occasional 40k-60k token requests.
# bash scripts/serve_qwen25_vl_vllm.sh longctx

# # Maximum-context profile for 128k-token reruns.
# bash scripts/serve_qwen25_vl_vllm.sh maxctx
# ```

# Notes:
# - `throughput` uses `--max-model-len 32768`, `--max-num-seqs 12`, and `--max-num-batched-tokens 24576`. This is the recommended default for mixed workloads.
# - `longctx` uses `--max-model-len 65536`, `--max-num-seqs 4`, and `--max-num-batched-tokens 16384`. Use it when you need to rerun very long document requests.
# - `maxctx` uses `--max-model-len 128000`, `--max-num-seqs 2`, and `--max-num-batched-tokens 8192`. Use it only for maximum-context reruns where throughput is secondary.
# - All presets enable chunked prefill.
# - Override GPU or port when needed, for example: `CUDA_VISIBLE_DEVICES=1 PORT=8001 bash scripts/serve_qwen25_vl_vllm.sh throughput`.

set -euo pipefail

PROFILE="${1:-throughput}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-1}"
VLLM_BIN="${VLLM_BIN:-/root/autodl-tmp/conda/envs/logma-rag/bin/vllm}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

COMMON_ARGS=(
  serve
  "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
)

case "${PROFILE}" in
  throughput)
    DEFAULT_GPU_MEMORY_UTILIZATION="0.9"
    DEFAULT_MAX_MODEL_LEN="32768"
    DEFAULT_MAX_NUM_SEQS="16"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="24576"
    ;;
  longctx)
    DEFAULT_GPU_MEMORY_UTILIZATION="0.9"
    DEFAULT_MAX_MODEL_LEN="65536"
    DEFAULT_MAX_NUM_SEQS="12"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="16384"
    ;;
  maxctx)
    DEFAULT_GPU_MEMORY_UTILIZATION="0.9"
    DEFAULT_MAX_MODEL_LEN="128000"
    DEFAULT_MAX_NUM_SEQS="8"
    DEFAULT_MAX_NUM_BATCHED_TOKENS="8192"
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
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)

if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  PROFILE_ARGS+=(--enable-chunked-prefill)
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
