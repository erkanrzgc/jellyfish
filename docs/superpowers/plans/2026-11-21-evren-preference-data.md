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

- [x] `evren_client`: key loading (env, then registry; never in repr/logs), 429/503 retry with `Retry-After`, timeout, reasoning-field handling. Tests use a fake HTTP opener.
- [x] `prompts`: schema and catalog validation, contamination filter, dedupe. Tests with fixed teacher output, including invented tool names, overlapping prompts and malformed JSON.
- [x] `rejected`: failure classification and resume. Tests with a scripted model.
- [x] `chosen`: validation gate and retry/drop policy. Tests with scripted teacher replies (valid, wrong tool, extra call, prose only).
- [x] `build`: record format, grouped split, manifest counts. Tests on a small fixture.
- [x] Smoke run: 20 prompts end to end, then a human spot-check before scaling.
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

## Smoke findings (2026-11-21)

- **Gateway limits:** when a model is busy EVREN allows only 2 concurrent requests per model per account (429 `fair_share_limit_exceeded`), not the 8 reported by `/v1/quota`. The client caps 2 per model; prompt generation alternates `deepseek-v4.1-flash` and `qwen3.8-flash-next` (Apache-2.0) to get 4 in flight. Chosen replies stay on deepseek.
- **Prompt quality:** the first teacher prompts named the expected tool, gave away no-tool answers, used real brands/IPs/handles and listed the privesc `shell` tool as a diagnostic alternative. The validator now rejects all of these, and fundamentals cases must be tool-free.
- **Labels:** v1.0 putting a URL in `hedef` instead of `url` is not a failure (the runtime treats target keys as one); `--relabel` fixed two false labels without a new model run.
- **Teacher artifact:** chosen replies justified authorization from address ranges ("RFC1918", "documentation range", "reserved domain") and wrapped `run_cmd` in `ssh`. Both are now teacher rules and automatic gate rejections; regenerated replies were clean on the first attempt.
- **Yield:** 35 prompts, 8 genuine v1.0 failures (about 23%), 8/8 accepted fixes. v1.0 takes about 18 s per prompt locally; deepseek about 85 s per chosen reply. The full run uses 300 prompt requests (`--total 2400`) with category weights shifted toward forensics and tooling, the highest-yield categories.
- **Label noise in the full run (2026-11-22):** early rows showed v1.0 marked wrong for picking an equivalent tool (nmap for a fast port scan where the case listed only rustscan; gobuster for ffuf) and for different flags in free-form `secenekler`. Such pairs would teach the near-miss tool preference that regressed v1.1/v1.2. `classify_failures` now accepts one-hop substitutes from tight functional groups (`SUBSTITUTE_GROUPS`) and compares only the targets/paths inside `secenekler`. `chosen` and `build` recompute labels from the raw reply, so the running `rejected` pass needs no restart. After the run: review the expected-vs-actual confusion table for `wrong_first_tool` (open question: `netexec` for SMB enumeration), extend groups, rebuild; weight the spot-check toward `in_catalog_wrong_tool`.
