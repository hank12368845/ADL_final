#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
print('Downloading pinned Qwen3.5 base + tokenizer; CPU/network only.', flush=True)
snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
print('Base model download complete.', flush=True)
PY
"$PYTHON_BIN" -u "$TASK_ROOT/training/fetch_official_data.py"
