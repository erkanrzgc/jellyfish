# Octopus agent instructions

These instructions are shared by Codex, Claude, and other coding agents working in this repository.

## Start here

1. Read `PROJECT_HANDOFF.md` for current decisions, latest verification, and the next task.
2. Check `git status --short` before editing. Preserve changes made by the user or another agent.
3. Read the relevant source and tests. Treat dated reports and old agent notes as historical evidence, not current runtime configuration.

`agent/config.py` owns runtime defaults. `README.md` is the public overview; `docs/DEMO_GUIDE.md` contains runnable demo instructions. Existing local `CLAUDE.md` points to this shared handoff and retains older Claude notes.

## Working agreements

- The product is Turkish-first; public repository documentation is English. Use ASCII paths (`octopus`); `Octópus` is the display brand.
- Keep v1.0 / `octopus-v10` as the evaluated default. v1.1 and v1.2 are experiments that were not promoted. Changing a default is not authorization to retrain or publish a model.
- Preserve ignored model files, checkpoints, data, and audit logs. Never reset or delete artifacts just to obtain a clean tree.
- Get explicit user approval before spending money on pods/cloud training. Keep credentials out of commands, logs, commits, and handoff documents.
- Run appropriate regression tests for behavior changes. Standard checks: `uv run pytest -q`, `uv run python -m agent.smoke`, and `git diff --check`.
- Clearly separate mock/runtime validation from live model and real-tool results. Do not claim complete integration coverage from catalog size or unit-test counts.
- The user alternates between agents. Avoid simultaneous edits to the same files; inspect the current diff and handoff before resuming.

## End each meaningful work session

Update `PROJECT_HANDOFF.md`: what changed, exact validation results, unresolved issues, and the next concrete step. Keep a single current handoff rather than separate competing Codex/Claude progress files. Leave historical experiment reports unchanged unless correcting a documented error.

The latest user instruction takes precedence over earlier notes. Do not require the user to repeat authorization already present in the conversation.
