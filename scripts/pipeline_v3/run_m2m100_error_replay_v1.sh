#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/fourlang_translation"
PYTHON_BIN="${FOURLANG_STUDENT_PYTHON:-/root/autodl-tmp/venvs/small100_student/bin/python}"
CONFIG="configs/multilingual/fourlang_m2m100_error_replay_v1.toml"
ACTION="${1:-preflight}"

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FOURLANG_MODEL_ROOT=/root/autodl-tmp/models

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

case "$ACTION" in
  build)
    "$PYTHON_BIN" scripts/pipeline_v3/build_m2m100_error_replay_v1.py --config "$CONFIG"
    ;;
  preflight)
    "$PYTHON_BIN" scripts/pipeline_v3/fourlang_m2m100_student.py preflight-error-replay --config "$CONFIG"
    ;;
  train)
    "$PYTHON_BIN" scripts/pipeline_v3/fourlang_m2m100_student.py train-error-replay --config "$CONFIG"
    ;;
  evaluate)
    "$PYTHON_BIN" scripts/pipeline_v3/fourlang_m2m100_student.py eval-error-replay --config "$CONFIG"
    ;;
  compare)
    "$PYTHON_BIN" scripts/pipeline_v3/fourlang_m2m100_student.py compare-error-replay --config "$CONFIG"
    ;;
  run-all)
    "$PYTHON_BIN" scripts/pipeline_v3/build_m2m100_error_replay_v1.py --config "$CONFIG"
    "$PYTHON_BIN" scripts/pipeline_v3/fourlang_m2m100_student.py run-error-replay --config "$CONFIG"
    ;;
  *)
    echo "Usage: $0 {build|preflight|train|evaluate|compare|run-all}" >&2
    exit 2
    ;;
esac
