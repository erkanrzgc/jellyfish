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

## Tasks

1. [ ] **Mingbird review** (clone into a separate dir outside the repo): read tool-subset routing, completion
   checks, call rescue/repair. Write a short pattern note; adapt patterns, do not copy wholesale.
2. [ ] **Constrained tool names** in the harness: Ollama `format` JSON schema with an enum of catalog names
   for the `arac` call (or grammar), behind a flag. TDD.
3. [ ] **Tool-subset retrieval**: per request show the model only the top-k relevant tools (+ assistant tools),
   deterministic (keyword/embedding). Measure router accuracy separately. TDD.
4. [ ] **Call rescue/repair**: parse plain-text or near-JSON calls, map near-miss names to catalog names only
   when unambiguous. TDD.
5. [ ] **Bake-off** (after `rejected` finishes, $0, local): same held-out cases, temperature 0:
   - v10 baseline vs v10 + tasks 2-4
   - stock Qwen3-8B and stock Gemma-4 (E4B or 12B) with native tool calling + the same harness
   Report structural metrics + a Turkish quality spot-check. Large models optionally on free Colab T4.
6. [ ] **v1.0 data reproducibility check** (only matters if v10 stays the base): rebuild data at `fb2e0c9`,
   confirm train=3231.
7. [ ] **Training prep**: DPO trainer (TRL, bf16 LoRA, beta ~0.1, 1 epoch) smoke-tested on a tiny model; data
   converter from ```arac``` to the winner's native tool-call format if a new base wins; one pod script with a
   self-stop watchdog and a cost estimate. Launch only on a clear bake-off winner (cap $3), otherwise stop.
8. [ ] DPO chain finishes -> confusion table, manifest, 20-pair spot-check; update `PROJECT_HANDOFF.md`.

## Decision rule

- If v10 + harness closes most structural failures and stock new bases do not beat it -> keep v10, spend on
  v1.0 repro + DPO.
- If a stock new base beats v10 on tool selection and passes the Turkish spot-check -> convert data to its format
  and spend on one SFT (+ optional DPO) run on that base.
