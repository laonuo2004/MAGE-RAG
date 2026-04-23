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
# - `throughput` and `longctx` cap multimodal input to `4` images per prompt by default.
# - `maxctx` caps multimodal input more aggressively at `2` images per prompt to preserve memory headroom.
# - All presets enable chunked prefill.
# - Override GPU or port when needed, for example: `CUDA_VISIBLE_DEVICES=1 PORT=8001 bash scripts/serve_qwen25_vl_vllm.sh throughput`.

set -euo pipefail

PROFILE="${1:-throughput}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-1}"
VLLM_BIN="${VLLM_BIN:-/root/autodl-tmp/conda/envs/logma-rag/bin/vllm}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

COMMON_ARGS=(
  serve
  "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
)

case "${PROFILE}" in
  throughput)
    PROFILE_ARGS=(
      --gpu-memory-utilization 0.92
      --max-model-len 32768
      --max-num-seqs 12
      --max-num-batched-tokens 24576
      --enable-chunked-prefill
      --limit-mm-per-prompt '{"image":4,"video":0}'
    )
    ;;
  longctx)
    PROFILE_ARGS=(
      --gpu-memory-utilization 0.95
      --max-model-len 65536
      --max-num-seqs 4
      --max-num-batched-tokens 16384
      --enable-chunked-prefill
      --limit-mm-per-prompt '{"image":4,"video":0}'
    )
    ;;
  maxctx)
    PROFILE_ARGS=(
      --gpu-memory-utilization 0.95
      --max-model-len 128000
      --max-num-seqs 2
      --max-num-batched-tokens 8192
      --enable-chunked-prefill
      --limit-mm-per-prompt '{"image":2,"video":0}'
    )
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    echo "Usage: $0 [throughput|longctx|maxctx]" >&2
    exit 1
    ;;
esac

echo "Starting vLLM with profile=${PROFILE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}, port=${PORT}" >&2
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  "${VLLM_BIN}" \
  "${COMMON_ARGS[@]}" \
  "${PROFILE_ARGS[@]}"
