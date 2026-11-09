#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  RUN="uv run python"
else
  RUN="python"
fi

TARGET="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$TARGET" == "all" ]]; then
  for name in transformer mamba deltanet hybrid; do
    $RUN -m src.part1.train --config "config/part1/${name}.yaml" "$@"
  done
else
  $RUN -m src.part1.train --config "$TARGET" "$@"
fi
