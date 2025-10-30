#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
PORT=${PORT:-7860}

export PORT
exec "$PYTHON_BIN" -m src.gradio_stream


