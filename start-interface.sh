#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Compatibility name for the explicit older arithmetic/MLX research path.
# The shared launcher reads .env as data and owns cleanup for every process.
if [ "$#" -eq 0 ]; then
  exec "$PROJECT_DIR/tinydog.sh" start --legacy-mlx
fi
if [ "$#" -le 2 ]; then
  WEIGHTS=$1
  CONFIG=${2:-configs/300m.json}
  exec "$PROJECT_DIR/tinydog.sh" start --legacy-mlx \
    --weights "$WEIGHTS" --config "$CONFIG"
fi
echo "Usage: ./start-interface.sh [weights [config]]" >&2
exit 2
