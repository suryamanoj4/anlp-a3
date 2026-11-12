#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  RUN="uv run python"
else
  RUN="python"
fi

$RUN -m src.part2.run --config config/part2/base.yaml "$@"
