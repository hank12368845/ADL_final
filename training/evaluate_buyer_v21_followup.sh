#!/usr/bin/env bash
# User-started inference only: final cash probe, then intermediate legacy probes.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/hank/.cache/adl_final_venv/bin/python}"
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash training/evaluate_buyer_v21_followup.sh <v2.1-run> [--gpus 0,1] [--port 8000]\n'
  exit 2
fi
BUYER_RUN="$1"
shift
BUYER_EVAL_OPTIONS=(--gpus 0,1)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus|--port)
      if [[ $# -lt 2 ]]; then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
      BUYER_EVAL_OPTIONS+=("$1" "$2")
      shift 2
      ;;
    *) printf 'Unsupported option: %s (use --gpus or --port)\n' "$1" >&2; exit 2 ;;
  esac
done
printf 'Inference only: 3 evaluations, 768 calls total; each phase displays progress/ETA and saves its result folder.\n'
printf 'Reference: v2 step8. Candidate checkpoints: the supplied v2.1 run.\n'
printf 'Phase 1/3 | completed prior calls 0/768 | final cash controls: 384 calls\n'
"$PYTHON_BIN" -u "$TASK_ROOT/training/evaluate_buyer_operations_v2_1.py" \
  "$BUYER_RUN" "${BUYER_EVAL_OPTIONS[@]}" --suite cash --checkpoint final
printf 'Phase 2/3 | completed prior calls 384/768 | step_0008 legacy controls: 192 calls\n'
"$PYTHON_BIN" -u "$TASK_ROOT/training/evaluate_buyer_operations_v2_1.py" \
  "$BUYER_RUN" "${BUYER_EVAL_OPTIONS[@]}" --suite legacy --checkpoint step_0008
printf 'Phase 3/3 | completed prior calls 576/768 | step_0012 legacy controls: 192 calls\n'
"$PYTHON_BIN" -u "$TASK_ROOT/training/evaluate_buyer_operations_v2_1.py" \
  "$BUYER_RUN" "${BUYER_EVAL_OPTIONS[@]}" --suite legacy --checkpoint step_0012
printf 'Completed 3/3 evaluations, 768/768 calls. Compare offer capture and cash rejection before choosing a checkpoint.\n'
