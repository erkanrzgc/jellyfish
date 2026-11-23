# Jellyfish shared project handoff

> Read this first when switching between Codex and Claude. This file is the shared current state; it replaces reliance on chat history. Update it at the end of meaningful work.

**Last updated:** 2026-11-21 (Europe/Istanbul), Claude — committed the 2026-10-23 runtime/reliability work on `feat/cybersecurity-reliability` and opened a PR.

**2026-11-21 status (Claude):** re-verified before commit: `uv run pytest -q` → 436 passed; `uv run python -m agent.smoke` → 3/3. The count rose from 340 because the cybersecurity-reliability work packages (runtime failures, execution approval, 60-case eval + runner, conversation interface; see `docs/superpowers/plans/2026-10-23-cybersecurity-reliability.md`) landed after the section below was written. On 2026-11-07 a user-approved disk cleanup removed every local GGUF except `octopus-v10` (from `models/` and Ollama); v0.8.1/v0.9/v1.1 GGUFs remain on Hugging Face, v0.7/v0.8/v1.2 only as local adapters in `checkpoints_sft/`. Separate track: the EVREN (SSB) LLM API is free until 2026-12-01 and is the candidate teacher for preference data; its training GPUs are image-only, so any LLM training still needs another provider.

**2026-11-21 preference-data sprint (Claude, branch `feat/evren-preference-data`, commit `9c2b18c`, not yet pushed):** plan `docs/superpowers/plans/2026-11-21-evren-preference-data.md`; code `data/dpo/` (57 tests; suite 493 passed). Stages: `python -m data.dpo.prompts` (EVREN teacher, running) → `python -m data.dpo.rejected` (local Ollama `octopus-v10`, ~18 s/prompt, needs the PC on for hours) → `python -m data.dpo.chosen` (EVREN deepseek) → `python -m data.dpo.build`. All stages resume; outputs are in gitignored `data/dpo/out/`. The key is read from `EVREN_API_KEY`/`HKCU\Environment` and must never be printed. Deadline 2026-11-30 (free EVREN inference ends 2026-12-01). No training or spending is authorized in this sprint.
**2026-11-22 harness + base bake-off (Claude, autonomous, same branch, not pushed):** master plan and live
progress log: `docs/superpowers/plans/2026-11-22-harness-and-base-bakeoff.md` (read it first). Commits
`5a8ae6b` constrained call repair (`agent/constrained.py`, `GgufModel.constrain_calls`, default repairs only
catalog-named calls; invented-name repair behind `repair_unknown_names`), `65e43bf` router Turkish-letter fold,
`239cd32`/`44d1024` stock-model native-tool bake-off on the 212 DPO val cases (`eval/native_tool_bakeoff.py`),
`e891f48` DPO trainer (`train/dpo_bf16.py`, BOS pre-flight), `bda1047` pod recipe `cloud/pod_dpo_v13.sh`
(v1.0 SFT repro + DPO + Q4, watchdog, no HF token on pod). v1.0 SFT data reproduced exactly at `fb2e0c9`
(train=3231, sha e633b060…). Suite 532 passed. Key correction: v10 `catalog` failures are mostly invalid
parameter keys (118) rather than invented names (99). Results land in `data/dpo/out/repair_replay.log` and
`data/dpo/out/bakeoff_summary.json`. Spending rule: a RunPod run only on a clear winner, cap $3 total.
**Workspace:** `C:\Users\erkanrzgc\Desktop\Octopus`.
**User request:** improve runtime/demo reliability after the project review and maintain one Markdown handoff so either agent can continue without losing context.

## Current task

Runtime-default alignment, repeatable offline demos, and shared documentation are implemented. The full suite passed **340 tests**, and the smoke runner passed **3/3 offline checks**. Live model and Docker integration remain unverified; start with the ordered next steps below.

The user delegated routine implementation with "top sende". There is no authorization in this task for paid training, cloud spending, publishing, or promoting an experimental model. Work is in the existing checkout; no commit, push, or PR has been made.

## Project and decisions

- Turkish-first cybersecurity and Linux/server model with a separate agent runtime.
- Base: `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1`; training: bf16 LoRA via Transformers/PEFT/TRL. Earlier from-scratch and Qwen/Unsloth paths are historical experiments.
- **Default (2026-11-22, user decision):** stock `gemma4:e4b`, native tool calling (`agent/config.py`). **Legacy:** v1.0 / `octopus-v10`. v1.1 and v1.2 were not promoted after tool-selection regressions; see `eval/reports/v1.2_gapfill_retrain_2026-09-15.md`.
- **Template:** `models/jellyfish-v8-tokenizer`, intentionally matching the v1.0 evaluation setup. (Folder renamed from octopus-v8-tokenizer on 2026-11-23; contents unchanged.)
- **Single source for runtime defaults:** `agent/config.py`. CLI/backend and active toolcall/subagent eval runners import it. Explicit model and tokenizer overrides remain supported.
- The v1.0 Q4 GGUF exists locally. Its bf16 adapter was not retained, according to the August quality report. Do not assume a fresh clone includes ignored weights, tokenizer directories, or generated evaluation data.

## Changes in this session

| File(s) | Change |
|---|---|
| `agent/config.py` | Central model, evaluated tokenizer directory, and Modelfile defaults |
| `agent/backends/gguf_model.py` | Use v1.0 defaults; missing-default-model error points to `Modelfile.v10` |
| `agent/cli.py` | Consistent defaults; `--tokenizer-dir`; forward `--target` in GGUF mock-tool mode |
| `eval/run_toolcall_eval.py`, `eval/run_subagent_eval.py` | Share the same runtime defaults; historical reports remain unchanged |
| `agent/smoke.py` | Three offline acceptance checks with a nonzero exit code on failure |
| `tests/agent/test_cli.py`, `test_gguf_model.py`, `test_smoke.py`; `tests/eval/test_runtime_defaults.py` | Regression tests for defaults/overrides and detection of smoke backend failures |
| `README.md`, `models/README.md`, `docs/DEMO_GUIDE.md`, `pyproject.toml` | Accurate status, setup, project description, executor coverage, demo commands, and measurement limitations |
| `AGENTS.md`, this file, local `CLAUDE.md` | Shared entry points and continuity protocol |

Implementation checklist: `docs/superpowers/plans/2026-10-23-runtime-and-handoff.md`.

## Verification snapshot

Before the changes, `python -m pytest -q` passed **327 tests** in the existing `.venv`.

During implementation, targeted regressions reproduced the old v0.7 defaults and missing tokenizer override before the fix. Final commands and results:

| Command | Observed result |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest tests/agent/test_cli.py tests/agent/test_gguf_model.py tests/agent/test_smoke.py tests/eval/test_runtime_defaults.py -q` | 32 passed |
| `uv run pytest -q` | 340 passed in 9.04 seconds |
| `uv run python -m agent.smoke` | 3/3 offline checks passed; exit 0 |
| `uv run python -m agent.cli --help` | v1.0 model and v0.8 tokenizer defaults displayed; exit 0 |
| `git diff --check` | No whitespace errors; Git emitted its normal LF/CRLF conversion notices |

An independent read-only code review found no material issues and reproduced 32 focused test passes, 3/3 smoke checks, and a clean whitespace check. Local links in the changed documentation and the changed runtime Python syntax were also checked successfully.

Ollama's local `/api/tags` request timed out after four seconds. A subsequent request to `127.0.0.1:11434/api/version` was refused, with no listening socket found on port 11434. `ollama`, `uv`, `docker`, and `wsl` commands are installed, but executable availability does not establish service readiness. No live inference or Docker scan has been claimed or performed in this session. No process was started/killed and no model was retrained/downloaded.

## Resume commands

From the repository root:

```powershell
git status --short
uv run python -m agent.smoke
uv run pytest -q
```

Existing Windows environment fallback:

```powershell
.\.venv\Scripts\python.exe -m agent.smoke
.\.venv\Scripts\python.exe -m pytest -q
```

For a live model check, follow `models/README.md` and `docs/DEMO_GUIDE.md`; use the promoted model and evaluated tokenizer together. Review actual calls and tool results, not just the legacy CLI's exit code.

## Next steps, in order

1. **Live v1.0 verification:** check Ollama readiness, then run the real-model/mock-tool demo. Record call validity, chosen tool, target/parameters, and final interpretation. Do not silently promote v1.1/v1.2 to get a working demo.
2. **Real nmap integration:** inspect an existing authorized Docker lab, run the scripted-model/real-tool demo, then the combined real-model/real-tool path. The guide gives commands and expected behavior. Get approval before cloud spending; do not create an external scan target.
3. **Per-tool CLI adapters:** implement and lab-test a few useful tools before claiming broad execution support. The generic argv mapping is incomplete; Docker's built-in map contains only nmap.
4. **Evaluation depth:** extend the small held-out set to cover argument correctness, all emitted calls, task completion, and varied refusal probes. Keep comparable generation settings and preserve raw replies. Current toolcall scoring examines only the first parsed tool name; 24 prompts are not broad quality evidence.
5. **Artifact reproducibility:** any future training must preserve adapter, tokenizer, pinned environment, data manifest/split, and eval outputs before a pod is deleted.

## Known limitations to retain

- The three offline checks use a scripted model and simulated scanning. Only their temporary-workspace filesystem operations are real.
- The normal CLI still prints a transcript and may exit zero when that transcript contains a backend error. The smoke runner has an explicit failure status; do not confuse the two.
- Catalog breadth is not integration coverage. Free-form `secenekler` and tool-specific argument/target behavior need validation; `shell=False` does not establish complete policy enforcement.
- The historical safety report includes a refusal followed by a tool call in one probe. Small successful safety sets do not erase that observation.
- `serve/jellyfish_chat.py` is a legacy Qwen/Unsloth entry point, not the current v1.0 GGUF runtime.
- `CLAUDE.md` is currently Git-ignored and kept locally with its original history. `AGENTS.md` and this handoff are normal repository files; after a fresh clone, ask Claude to read them if no local entry point exists.

## Agent-switch prompt

> `AGENTS.md` ve `PROJECT_HANDOFF.md` dosyalarını oku, mevcut git diff'ini kontrol et ve sıradaki adımdan devam et. Yapılan işleri tekrarlama; yeni doğrulama sonuçlarını ortak devir dosyasına yaz.
