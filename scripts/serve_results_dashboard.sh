#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${RESULTS_DASHBOARD_PORT:-8501}"
conda run -n logma-rag-py12 streamlit run results_dashboard.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.headless true
