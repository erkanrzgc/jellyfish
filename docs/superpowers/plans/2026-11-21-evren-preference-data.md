# EVREN preference-data sprint

Approved direction: 2026-11-21 (user chose EVREN as teacher). Deadline: **2026-11-30**, one day before EVREN's free inference period ends (2026-12-01).

## Goal

Build an auditable set of `(prompt, rejected, chosen)` triples that target v1.0's observed **structural tool-use failures**: invented tool names, wrong first tool, missing required parameters, and tool calls where none is needed. `rejected` is v1.0's own reply (on-policy, local); `chosen` is a teacher-written reply that passes the same structural checks.

The output serves two later uses without committing to either: DPO preference pairs, and SFT "correction" examples for the cybersecurity-reliability plan. **No training and no spending happen in this plan.**

## Constraints

- **Teacher:** `deepseek-v4.1-flash` (MIT). Measured 2026-11-21: ~60 s per short streamed answer, correct Turkish. `glm-5.3` was saturated (503 / 214 s to first token); use it only as an optional second opinion and check its "Z.ai GLM" license before using its outputs.
- **EVREN terms v1, section 2:** no critical-infrastructure attack content. Offensive prompts must be framed as authorized lab, CTF or scoped pentest work; ICS/SCADA/energy/utility targets are excluded. Prompts are synthetic; no personal data is sent.
- **Quota:** 15M tokens/day, account concurrency 8 (`/v1/quota`). Honor `Retry-After` on 429; back off on 503. Reasoning models share `max_tokens` between thinking and answer, so use at least 4096.
- **Secrets:** the key is read from `EVREN_API_KEY` or `HKCU\Environment`; it is never printed, logged or written to output files.
- **No eval contamination:** drop any generated prompt that overlaps `eval/suites/reliability-dev-v1.json`, `eval/data/toolcall_eval_tr.jsonl`, `eval/data/quality_safety_tr.jsonl` or `eval/data/cassiopeia_eval_prompts_tr.txt` (normalized exact match, or token-set Jaccard >= 0.6). Eval prompts are never shown to the teacher.

## Pipeline

1. **Prompts** (`data/dpo/prompts.py`): the teacher writes cases in the reliability-suite schema (`category`, `family_id`, `prompt`, `expected_tools`, `expected_params`, `allow_tools`) for the six suite categories. The mix is weighted toward the dev60 weak spots: knowledge questions that tempt an unnecessary call, near-miss tool names, and calls with required parameters. Validation: `expected_tools` must be catalog names, `expected_params` must be catalog parameters, plus dedupe and the contamination filter. Target: 1500 cases.
2. **Rejected** (`data/dpo/rejected.py`): local `octopus-v10` via `agent.profiles.create_model` with the evaluated settings (context 4096, output 1024, temperature 0). Score with `eval.reliability_eval.structural_metrics`; keep only cases that fail at least one check, and record which. Resumable append-only JSONL, about 10 s per case.
3. **Chosen** (`data/dpo/chosen.py`): the teacher writes the reply in Octópus format (`dusunce` + `arac` blocks, Turkish) under `OCTOPUS_TOOL_SYSTEM_PROMPT`, given the catalog entries for the case and its expected behavior. It must pass every structural check that `rejected` failed and introduce no new failure. Up to two retries; otherwise the case is dropped with the reason recorded.
4. **Build** (`data/dpo/build.py`): writes TRL conversational preference records `{prompt: [system, user], chosen: [assistant], rejected: [assistant]}`, an SFT view, and a manifest (counts per category and failure type, teacher model, v1.0 GGUF SHA-256, timestamps, drop reasons). The split is grouped by `family_id` (90/10) so that variants of one scenario never cross splits.

Shared: `data/dpo/evren_client.py`, a minimal stdlib client (no new dependency) with bounded concurrency, retries and key handling.

## Tasks (TDD: test first, then implementation)

- [ ] `evren_client`: key loading (env, then registry; never in repr/logs), 429/503 retry with `Retry-After`, timeout, reasoning-field handling. Tests use a fake HTTP opener.
- [ ] `prompts`: schema and catalog validation, contamination filter, dedupe. Tests with fixed teacher output, including invented tool names, overlapping prompts and malformed JSON.
- [ ] `rejected`: failure classification and resume. Tests with a scripted model.
- [ ] `chosen`: validation gate and retry/drop policy. Tests with scripted teacher replies (valid, wrong tool, extra call, prose only).
- [ ] `build`: record format, grouped split, manifest counts. Tests on a small fixture.
- [ ] Smoke run: 20 prompts end to end, then a human spot-check before scaling.
- [ ] Full run: prompts, then the overnight rejected pass, then chosen, then build.
- [ ] Human spot-check of 20 random pairs; update `PROJECT_HANDOFF.md`; PR.

## Acceptance

- At least 500 validated pairs (stretch: 800). Every `chosen` passes all structural checks; every `rejected` fails at least one.
- Zero overlap with the eval sets listed above; per-category and per-failure-type counts reported.
- A 20-pair human spot-check is recorded before any training decision.
- Full test suite green; no key material in code, logs or outputs.

## Risks

- **Teacher latency or capacity:** concurrency 8, resumable stages, start the overnight run early.
- **Too few failures** (v1.0 failed 10/60 structurally on dev60, about 17%): if 1500 prompts yield fewer than 500 failures, add one temperature-0.6 sample per prompt or generate more weak-spot prompts.
- **Content-guard refusals** of offensive prompts: log and skip; never rephrase to get around them.
- **Structural checks are not semantic quality:** a teacher reply can pass and still be wrong. The spot-check and the reliability-plan human review remain required.
