#!/bin/bash
# Blocks Claude's Edit/Write tools from touching raw data, lockfiles or git internals.
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
FILE_PATH="${FILE_PATH//\\//}"

PROTECTED_PATTERNS=("data/raw/" "uv.lock" ".git/" "reports/results/")

for pattern in "${PROTECTED_PATTERNS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    echo "Blocked: $FILE_PATH matches protected pattern '$pattern'. Raw data is read-only, uv.lock changes via 'uv add', and results JSONs are written only by scripts." >&2
    exit 2
  fi
done
exit 0
