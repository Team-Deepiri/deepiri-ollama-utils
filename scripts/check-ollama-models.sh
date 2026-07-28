#!/usr/bin/env bash
# Compatibility entry point replacing diri-cyrex/scripts/llm/check-ollama-models.sh.
set -euo pipefail

if ! command -v deepiri-ollama-utils >/dev/null 2>&1; then
    echo "deepiri-ollama-utils is not installed or not on PATH." >&2
    echo "Install this package and deepiri-gpu-utils, then retry." >&2
    exit 127
fi

exec deepiri-ollama-utils recommend-models "$@"
