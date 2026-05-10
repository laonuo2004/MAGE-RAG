PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/logma-rag-py12/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${CODE_ROOT}"
exec "${PYTHON_BIN}" main.py --multirun baselines=m3docrag \
    baselines.top_k=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    benchmarks=longdocurl,mmlongbench \
    benchmarks.workers=128