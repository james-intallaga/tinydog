#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_COMMAND=${PYTHON_COMMAND:-python3}

exec "$PYTHON_COMMAND" "$PROJECT_DIR/scripts/tinydog.py" "$@"
