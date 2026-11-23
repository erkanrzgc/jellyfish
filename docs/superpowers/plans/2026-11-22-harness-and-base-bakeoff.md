# Harness reliability + base-model bake-off (2026-11-22)

Approved direction (user, 2026-11-22): fix tool-use reliability in the harness first, compare base models with
evidence, then spend the remaining RunPod balance (~$3.33) on the winner. The user is away; Claude works
autonomously on everything that costs $0 and stops at the money gate.

## Evidence that motivates this

- v10 structural failures on the DPO case set (first 1253 cases): 422 failures; 175 (41%) are invented tool
  names (`catalog`), 222 `wrong_first_tool` (many are generic fallbacks: `run_cmd`, `web_fetch`, `web_search`),
  58 `missing_call`, 52 `param_mismatch`, 16 `unnecessary_call`.
- SFT tool data regressed tool selection twice (v1.1, v1.2). The two real wins were inference-time
  (conditional RCA nudge 3/5 -> 4/5, extension discovery manifest +2/3).
- Turkish-Gemma-9b-v0.1 is Gemma-2 based with no native tool calling; we taught a custom ```arac``` format.
  Gemma 4 (E2B/E4B/12B/26B-A4B/31B, April 2026) and Qwen3 have native function calling.
- Mingbird (arXiv 2610.02001, Apache-2.0, github.com/Mingbird/Mingbird-agent): local-first Ollama/Windows harness
  for 2-9B models, 0.886 vs goose 0.631. Gains: completion checks (+0.152), tool-subset routing by domain,
  plain-text tool-call rescue + argument repair. No constrained decoding; not tested on 100+ tools or Turkish.
- BILGE (TUBITAK, 37B) is not released and would not fit 8 GB VRAM. EVREN hosts `gemma-4-31b`,
  `qwen3-embedding-8b`, `qwen3-reranker-8b`, `qwen3-guard-4b` (free until 2026-12-01).

## Constraints

- Local GPU: RTX 5060 8 GB VRAM. Shipping model must fit Q4 in ~8 GB (Gemma-4 E4B/12B, Qwen3-8B, v10).
- The DPO data chain (`data/dpo`, background) uses the local GPU until the `rejected` stage ends (~20:00);
  the bake-off waits for it. `chosen` uses EVREN only.
- **User decisions (2026-11-22, AskUserQuestion):**
  - Downloads APPROVED: Mingbird repo clone, `ollama pull` qwen3:8b, gemma4 E4B and 12B.
  - Pod: "Kendin başlat" — Claude may launch the RunPod training itself IF the bake-off gives a clear winner,
    budget cap **$3** total, delete the pod as soon as artifacts (bf16 adapter + GGUF) are on HF, self-stop
    watchdog on the pod. If the winner is unclear, prepare script + estimate and wait.
- No push / PR without confirmation. No promotion of a new default without the user.
- 17:35 user: "guncelle sana herseyin iznini veriyorum otonom bir sekilde calis" -> Ollama update approved
  (do it after the `rejected` stage ends, it restarts the server), full autonomy within the rules above.
  gemma4 pull failed on Ollama 0.30.0 ("download the latest version"); qwen3:8b pulled OK.
- User said Claude may work at "ultracode" scale (subagents / workflow allowed) while away.
- State at 17:30: Mingbird cloned to `C:\Users\erkanrzgc\refs\Mingbird-agent` (Apache-2.0, 7.7 MB).
  `ollama pull qwen3:8b gemma4:e4b gemma4:12b` running in background (log `data/dpo/out/pulls.log`).
  DPO chain resumed after a PC shutdown at 17:04 (rejected 1282/2193; chain `rejected -> chosen -> build`).

## Progress log (autonomous run, 2026-11-22 evening)

- 19:05 resumed. Corrected breakdown on 1936 rejected rows: `catalog` 243 = bad parameter key 118, unknown name 99,
  missing required 15, bad JSON 9. Plus 7 bad-JSON blocks counted as `missing_call`; 52 `missing_call` are
  ```bash advice instead of a call (not repairable safely: converting advice into execution is a policy change).
- Probe: Ollama 0.30.0 honours `format` (JSON schema) in raw mode; v10 `{"alan":..}` -> `{"domain":..}`.
- `5a8ae6b` constrained repair (`agent/constrained.py`, `GgufModel.constrain_calls`). Default repairs only calls
  naming a catalog tool; invented-name repair behind `repair_unknown_names` (advisor: on refusal prompts it could
  turn a non-executable invented call into a runnable real one). Replay eval `eval/call_repair_replay.py`.
- `65e43bf` router folds Turkish letters: general fallback 56.2% -> 53.9% on 1407 tool prompts; routed-role
  recall of an acceptable tool only 78% -> do NOT narrow the v10 repair schema by router (full catalog).
- `239cd32` stock-model bake-off runner `eval/native_tool_bakeoff.py` (native tools, num_ctx 16k, think off,
  320-case stratified sample, same labels). qwen3:8b smoke: correct `dig` call, prompt 6.7k tokens.
- Task 6 DONE: `build_sft --source distill seed_tr tools --seed-repeat 3` at `fb2e0c9` -> train=3231, sha
  e633b060..., deterministic across two runs. v1.0 SFT is reproducible for a pod run.
- Task 3 decision (measured, $0): no top-k tool subset for v10. Keyword router recall 78% would hide the right
  tool in 22% of routed cases, and a prompt-side tool list is the same distribution-shift risk as the global RCA
  rule. Catalog has no tool descriptions for a better retriever. Stock models get the full catalog (6.7k tokens
  fits num_ctx 16k).
- Task 7: `e891f48` DPO trainer (render identical to SFT, single-BOS pre-flight, tested on the real tokenizer);
  `bda1047` `cloud/pod_dpo_v13.sh`. Estimate ~3.2 h A40 (SFT 1570 steps ~2.2 h + DPO ~0.4 h + GGUF ~0.5 h)
  ~= $1.4 at ~$0.44/h; watchdog stops the pod at 5 h (<= ~$2.2). Pairs need `chosen` (~9 h, EVREN) first.
- 19:46-20:01 INCIDENT: the stopped old update script (`after_rejected_update.sh`, TaskStop did not kill the
  MSYS sh child) killed Ollama mid-replay and upgraded to 0.40.16. Replay: 300 attempted, 212 applied, 87 lost
  to connection errors. 0.30.0 replay (full 2193 rows, partial): failed 766 -> 681 (catalog-name repair) ->
  671 (any-name); `catalog` 300 -> 160 -> 88.
- SAFETY FINDING: catalog-name repair changed 13 no-tool replies into valid calls, 7 of them on authorization
  prompts (nmap on an ex-colleague's mail server, bettercap ARP poisoning, subfinder on a rival's domain). The
  invalid originals could not execute; repaired ones can (policy scope + execution approval still apply).
  Keep `constrain_calls` default OFF; enable only with scope + approval; the fix for these is DPO (refusal chosen).
- Engine change: 0.40.16 reproduces only 8/30 stored v10 replies byte-for-byte -> v10 is re-run beside the stock
  models on 0.40.16 (`--v10` arm, `refs/bakeoff_run.sh`); the stored 0.30.0 arms stay as reference only.
  Format re-probe on 0.40.16 OK (1.3 s).
- Background chain: `refs/after_rejected_v2.sh` (rejected end -> repair replay on 0.30.0 -> Ollama upgrade ->
  gemma4 pulls) then `refs/after_pulls_bakeoff.sh` (format re-probe -> bake-off qwen3:8b, gemma4:e4b,
  gemma4:12b -> `data/dpo/out/bakeoff_summary.json`). Logs: pulls.log, repair_replay.log, bakeoff.log.

- 22:30 BAKE-OFF RESULT (212 val, Ollama 0.40.16): gemma4:e4b 177 > qwen3:8b 155 > v10+repair 148-152 > v10 137.
  Report `eval/reports/gemma4_e4b_bakeoff_2026-11-22.md`. gemma4:12b stopped at 70/212 (no 8 GB fit).
  User (22:40) agreed: no spend now; integrate stock gemma4:e4b first ($0); possible rename to "Jellyfish"
  pending (name-clash check: HF Jellyfish-13B exists).
- 22:45 second PC shutdown; all JSONL intact; `chosen` resumed (bc9arxc81).
- `853c2b5` NativeChatModel backend (`agent/backends/native_chat.py`, CLI `--gguf --native gemma4:e4b`); live
  demo: nmap native call -> skill layer -> policy denied unverified `-sC`.
- `bf8bbbd` opt-in `evidence_push` in `run_tool_loop`. gemma4 RCA 0/5 (base) / 1/5 (agentic prompt) /
  5/5 (push); push costs 3 val no-tool cases. Safety battery: unauthorized 4/4 + reframing 2/2 refused,
  authorized 4/4 helped, base64 probe over-refused, persona prefix on every answer.
- NEXT: persona prompt fix (introduce only when asked), decide default for evidence_push, then (paid, later)
  a small Gemma 4 LoRA on the RCA + tool data if the agentic-RCA gap matters.

- 2026-11-23 02:45 DPO data DONE: 758 pairs (train 678 / val 80), dropped 8, teacher 429s retried. Spot-check
  OK (tool use, real catalog tools, calls instead of user commands); 19 empty `dusunce` fences fixed in build
  (`fix(data)` commit). Rename done: GitHub repo is now erkanrzgc/jellyfish (user-approved); nothing pushed.

## Tasks

1. [x] **Mingbird review** (clone into a separate dir outside the repo): read tool-subset routing, completion
   checks, call rescue/repair. Write a short pattern note; adapt patterns, do not copy wholesale.
2. [x] **Constrained tool names** in the harness: Ollama `format` JSON schema with an enum of catalog names
   for the `arac` call (or grammar), behind a flag. TDD.
3. [x] **Tool-subset retrieval** (measured, not adopted; see log): per request show the model only the top-k relevant tools (+ assistant tools),
   deterministic (keyword/embedding). Measure router accuracy separately. TDD.
4. [x] **Call rescue/repair** (constrained repair; ```bash advice rescue rejected as unsafe): parse plain-text or near-JSON calls, map near-miss names to catalog names only
   when unambiguous. TDD.
5. [x] **Bake-off** (after `rejected` finishes, $0, local): same held-out cases, temperature 0:
   - v10 baseline vs v10 + tasks 2-4
   - stock Qwen3-8B and stock Gemma-4 (E4B or 12B) with native tool calling + the same harness
   Report structural metrics + a Turkish quality spot-check. Large models optionally on free Colab T4.
6. [x] **v1.0 data reproducibility check** (only matters if v10 stays the base): rebuild data at `fb2e0c9`,
   confirm train=3231.
7. [ ] **Training prep**: DPO trainer (TRL, bf16 LoRA, beta ~0.1, 1 epoch) smoke-tested on a tiny model; data
   converter from ```arac``` to the winner's native tool-call format if a new base wins; one pod script with a
   self-stop watchdog and a cost estimate. Launch only on a clear bake-off winner (cap $3), otherwise stop.
8. [x] DPO chain finishes -> confusion table, manifest, 20-pair spot-check; update `PROJECT_HANDOFF.md`.

## Decision rule

- If v10 + harness closes most structural failures and stock new bases do not beat it -> keep v10, spend on
  v1.0 repro + DPO.
- If a stock new base beats v10 on tool selection and passes the Turkish spot-check -> convert data to its format
  and spend on one SFT (+ optional DPO) run on that base.
