#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  RUN="uv run python"
else
  RUN="python"
fi

ROLL="${ROLL_NUMBER:-rollnumber}"
ARCH="${BEST_ARCH:-transformer}"

if [[ "${1:-}" == "--heldout" ]]; then
  $RUN -m src.part1.infer \
    --config "config/part1/${ARCH}.yaml" \
    --input data/part1/heldout \
    --output "output/part1/heldout_result_${ROLL}.jsonl"
  exit 0
fi

for name in transformer mamba deltanet hybrid; do
  $RUN -m src.part1.infer \
    --config "config/part1/${name}.yaml" \
    --split test \
    --output "output/part1/test_decrypt_${name}.jsonl"
done
