#!/usr/bin/env bash
set -euo pipefail

: "${QWEN3_MODEL:=Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
: "${QWEN3_VOICES_DIR:=/voices}"
: "${QWEN3_DEVICE:=auto}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

export QWEN3_MODEL QWEN3_VOICES_DIR QWEN3_DEVICE HOST PORT LOG_LEVEL

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
