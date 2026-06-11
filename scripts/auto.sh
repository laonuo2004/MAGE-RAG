#!/usr/bin/env bash
set -euo pipefail

# =========================
# Basic config
# =========================
WORKDIR="/vllm"
DISKDIR="/root/autodl-tmp"
VENV_DIR="${WORKDIR}/.venv"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
MODEL_DIR="${DISKDIR}/Qwen3-VL-8B-Instruct"
SERVED_MODEL_NAME="Qwen3-VL-8B-Instruct"
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

MAX_MODEL_LEN=131072
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=32768
PORT=6006
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=1

export VLLM_VERSION=0.19.1
export CUDA_VERSION=128
export CPU_ARCH=x86_64

# =========================
# System / Python tools
# =========================
cd /
mkdir -p "${WORKDIR}"

python3 -m pip install -U pip -i "${PIP_INDEX_URL}"
python3 -m pip install -U uv -i "${PIP_INDEX_URL}"
python -m pip install --upgrade pip setuptools wheel -i "${PIP_INDEX_URL}"

# =========================
# Create virtual environment
# =========================
cd "${WORKDIR}"

if [ ! -d "${VENV_DIR}" ]; then
  uv venv
fi

source "${VENV_DIR}/bin/activate"

# =========================
# Install Python packages
# =========================

uv pip install -U modelscope -i "${PIP_INDEX_URL}"

# =========================
# Download model
# =========================
mkdir -p "${MODEL_DIR}"

# 如果目录为空，才下载，避免重复下载
if [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
  modelscope download \
    --model "${MODEL_ID}" \
    --local_dir "${MODEL_DIR}"
else
  echo "[INFO] Model directory already exists and is not empty: ${MODEL_DIR}"
fi

# =========================
# Install Python packages
# =========================

uv pip install "vllm==${VLLM_VERSION}" --torch-backend=cu${CUDA_VERSION} -i "${PIP_INDEX_URL}"

# =========================
# Create serve script
# =========================
cat > "${WORKDIR}/serve.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

source "${VENV_DIR}/bin/activate"

vllm serve "${MODEL_DIR}" \\
  --served-model-name "${SERVED_MODEL_NAME}" \\
  --host 0.0.0.0 \\
  --port ${PORT} \\
  --trust-remote-code \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --gpu-memory-utilization 0.8 \\
  --max-num-seqs ${MAX_NUM_SEQS} \\
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \\
  --limit-mm-per-prompt.video 0 \\
  --mm-processor-cache-gb 0 \\
  --data-parallel-size ${DATA_PARALLEL_SIZE} \\
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \\
  --async-scheduling \\
  --aggregate-engine-logging
EOF

chmod +x "${WORKDIR}/serve.sh"

echo "[INFO] Setup finished."
echo "[INFO] Start server with:"
echo "bash ${WORKDIR}/serve.sh"

echo "set -g mouse on" >> ~/.tmux.conf
tmux new-session -d -s vllm "bash ${WORKDIR}/serve.sh"
tmux attach -t vllm