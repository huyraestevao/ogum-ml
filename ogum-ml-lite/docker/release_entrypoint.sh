#!/usr/bin/env bash
set -euo pipefail

STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
STREAMLIT_ADDRESS="${STREAMLIT_ADDRESS:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"

export STREAMLIT_SERVER_PORT="${STREAMLIT_PORT}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_ADDRESS}"
export STREAMLIT_THEME_BASE="light"

uvicorn server.api_main:app --host 0.0.0.0 --port "${API_PORT}" &
UVICORN_PID=$!

cleanup() {
  kill "${UVICORN_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec streamlit run app/streamlit_app.py \
  --server.port="${STREAMLIT_PORT}" \
  --server.address="${STREAMLIT_ADDRESS}"
