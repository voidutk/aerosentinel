# AeroSentinel build kit — start here

This kit turns the "Jet Engine Failure Predictor" idea into a portfolio-grade project, and sets Claude Code up to build it in phases.

## What's inside
| File | What it is |
|---|---|
| `CLAUDE.md` | Short rules Claude Code reads every session (commands, leakage rules, definition of done) |
| `docs/SPEC.md` | The full A–Z specification: data, protocol, models, uncertainty, decision layer, serving, README plan, resume kit |
| `docs/PHASES.md` | 8 phases with copy-paste prompts and review steps; each phase ends in a shippable, tagged release |
| `docs/RESEARCH.md` | Every source the spec relies on, with links |
| `.claude/settings.json` | Permissions + hooks: protect raw data, auto-format Python, block "done" until lint + fast tests pass |
| `.claude/hooks/*.sh` | The hook scripts |
| `.claude/agents/ml-reviewer.md` | A read-only reviewer subagent that hunts leakage and unsupported claims |
| `.claude/skills/run-experiment/` | `/run-experiment <name>` — train over 3 seeds, evaluate, refresh tables |

## Setup (once)
1. Install: `uv` (https://docs.astral.sh/uv/), `jq`, `git`, `gh` (GitHub CLI), Claude Code.
2. Create an empty GitHub repo named `aerosentinel` and clone it.
3. Copy everything in this kit (including the hidden `.claude/` folder) into the repo root.
4. `chmod +x .claude/hooks/*.sh`
5. `cd aerosentinel && claude`, then run `/hooks` to confirm the three hooks loaded and `/context` to confirm CLAUDE.md loaded.

## Then
Open `docs/PHASES.md` and run Phase 0. One phase per fresh session (`/clear`), plan mode first, review with the subagent at the end, PR + tag.

## Two rules that keep the project honest
- Every number in the README comes from a results JSON a script wrote. If Claude types a metric by hand, reject it.
- If a C-MAPSS FD001 RMSE comes in suspiciously good (< 11), assume leakage until proven otherwise.
