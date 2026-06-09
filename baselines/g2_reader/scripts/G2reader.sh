#!/usr/bin/env bash
set -euo pipefail
subsets=(feta_tab paper_tab spiqa slidevqa scigraphvqa)
BASE_SAVE_DIR="results/"
for subset in "${subsets[@]}"; do
  DATA_PATH="/data/new/processed_${subset}.jsonl"
  SAVE_DIR="${BASE_SAVE_DIR}/${subset}"
  mkdir -p "${SAVE_DIR}"
  echo "Running ${subset}..." 
  echo "Data path: ${DATA_PATH}"
  python -m test.test_rag --data_path "${DATA_PATH}" --save_dir "${SAVE_DIR}" --model <YOUR_MODEL_NAME> --use_dag 
done