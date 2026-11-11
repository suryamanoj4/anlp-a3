#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  RUN="uv run python"
else
  RUN="python"
fi

LIMIT="${LIMIT:-100}"

for name in transformer deltanet mamba hybrid; do
  $RUN -m src.part1.evaluate \
    --config "config/part1/${name}.yaml" \
    --split test \
    --limit "$LIMIT" \
    --wandb
done
