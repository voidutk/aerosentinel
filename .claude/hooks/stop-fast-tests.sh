#!/bin/bash
# Before Claude ends a turn, run lint + fast tests. Blocks the stop (exit 2) if they fail,
# so Claude keeps fixing. Skips if no Python project exists yet, if already re-triggered,
# or if no .py file changed since the last commit (staged, unstaged or untracked).
INPUT=$(cat)
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active')" = "true" ]; then
  exit 0
fi
cd "$CLAUDE_PROJECT_DIR" || exit 0
[ -f pyproject.toml ] || exit 0
[ -d tests ] || exit 0

if git rev-parse --verify -q HEAD >/dev/null 2>&1; then
  CHANGED_PY=$( { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } 2>/dev/null )
else
  # No commits yet: every .py file in the working tree counts as changed.
  CHANGED_PY=$(git ls-files --cached --others --exclude-standard -- '*.py' 2>/dev/null)
fi
[ -n "$CHANGED_PY" ] || exit 0

OUT=$(uv run ruff check . 2>&1 && uv run pytest -q -m "not slow" -x 2>&1)
if [ $? -ne 0 ]; then
  echo "Lint or fast tests are failing. Fix them before finishing:" >&2
  echo "$OUT" | tail -40 >&2
  exit 2
fi
exit 0
