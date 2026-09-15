#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/fourlang_translation"
PYTHON_BIN="${FOURLANG_STUDENT_PYTHON:-/root/autodl-tmp/venvs/small100_student/bin/python}"
ACTION="${1:-run-preview}"
shift || true

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/pipeline_v3/preview_m2m100_error_replay.py \
  "$ACTION" \
  --config configs/multilingual/fourlang_m2m100_exp4_preview_v1.toml \
  "$@"
