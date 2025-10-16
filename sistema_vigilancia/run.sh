#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
SRC=${1:-0}

exec "$PYTHON_BIN" -m src.main --source "$SRC"





