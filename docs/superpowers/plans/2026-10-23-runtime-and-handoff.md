# Runtime defaults, demos, and shared handoff

**Goal:** Make the evaluated v1.0 runtime easy to launch and let Codex and Claude resume from the same project-local context.

**Approved scope:** The user accepted the preceding review's proposal and delegated implementation on 2026-10-23. Work stays in the existing checkout so switching agents sees the same files. No training, cloud spend, publication, or model promotion is part of this task.

**Architecture:** A small `agent/config.py` owns the evaluated runtime defaults. CLI and evaluation entry points consume those values; explicit model/tokenizer overrides remain available. An offline smoke runner exercises the existing parser/loop/registry and filesystem boundaries. `PROJECT_HANDOFF.md` holds current state; root agent instructions point both assistants to it.

**Tech stack:** Python, pytest, existing uv environment, Ollama for optional live inference.

## Checklist

- [x] Add regression coverage for v1.0 model/tokenizer selection and CLI overrides; run it and observe failures before implementation.
- [x] Centralize defaults: `jellyfish-v10`, `models/jellyfish-v8-tokenizer` (the template used in v1.0 evaluations), `models/Modelfile.v10`. Align CLI, backend, and active eval runners. Keep historical reports and conversion recipes unchanged.
- [x] Add an offline runner for three checks: allowed mock scan through the tool loop, rejected out-of-scope scan before execution, real temporary-workspace file round trip plus rejected path escape. Exit nonzero on a failed check. Cover detection of executor failures.
- [x] Rewrite README/model setup docs with accurate version status, executor coverage, prerequisites, expected smoke output, and evaluation limits. Separate historical measurements from current verification.
- [x] Add `AGENTS.md` and `PROJECT_HANDOFF.md`; prepend the shared entry point to existing local `CLAUDE.md` without removing historical notes.
- [x] Run focused tests, full pytest, smoke runner, CLI help, and whitespace/diff review. Record exact outcomes and pending live checks in the handoff.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests/agent/test_cli.py tests/agent/test_gguf_model.py tests/agent/test_smoke.py tests/eval/test_runtime_defaults.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m agent.smoke
.\.venv\Scripts\python.exe -m agent.cli --help
git diff --check
```

Live Ollama inference is conditional on an available local service; Docker execution requires a user-prepared lab. Missing services must be reported as unverified, never as successful model/tool validation.

## Follow-up beyond this task

Use the shared handoff's ordered next steps for live v1.0 verification, per-tool CLI adapters, and a larger evaluation covering parameters and completed tasks. Do not infer that the 117-tool catalog means 117 tested real integrations.
