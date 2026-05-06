#!/usr/bin/env bash
set -euo pipefail

# =========================
# Basic config
# =========================
WORKDIR="/vllm"
VENV_DIR="${WORKDIR}/.venv"
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
MODEL_DIR="/hy-tmp/Qwen3-VL-8B-Instruct"
SERVED_MODEL_NAME="Qwen3-VL-8B-Instruct"
PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"

MAX_MODEL_LEN=65536
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=2048
PORT=8080

export VLLM_VERSION=0.19.1
export CUDA_VERSION=128
export CPU_ARCH=x86_64

# =========================
# System / Python tools
# =========================
cd /
mkdir -p "${WORKDIR}"

python3 -m pip install -U pip
python3 -m pip install -U uv
python -m pip install --upgrade pip setuptools wheel

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

uv pip install -U modelscope

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

vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port ${PORT} \
  --trust-remote-code \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --limit-mm-per-prompt.video 0 \
  --mm-processor-cache-gb 0 \
  --async-scheduling 
EOF

chmod +x "${WORKDIR}/serve.sh"

echo "[INFO] Setup finished."
echo "[INFO] Start server with:"
echo "bash ${WORKDIR}/serve.sh"

tmux new-session -d -s vllm "bash /vllm/serve.sh"