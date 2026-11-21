#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash training/evaluate_buyer_operations.sh training/runs/<completed-run> [diagnostic options]'
  exit 2
fi
BUYER_RUN="$1"
shift
exec "$PYTHON_BIN" -u "$TASK_ROOT/training/buyer_diagnostics.py" \
  --versions operations --adapter "$BUYER_RUN/final_adapter" --compare-base \
  --cases-file "$TASK_ROOT/training/data/buyer_operations_v1/test_cases.json" "$@"
