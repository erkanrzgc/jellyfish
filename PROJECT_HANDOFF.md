# Jellyfish shared project handoff

> Read this first when switching between Codex and Claude. This file is the shared current state; it replaces reliance on chat history. Update it at the end of meaningful work.

**Last updated:** 2026-11-25 (Europe/Istanbul), Claude. gemma4 DPO stage 2 ran ($1.89): the 1-epoch DPO model scores 175/212 on the val cases, the same as the requant control and stock `gemma4:e4b`, so it is not promoted; stock `gemma4:e4b` stays the default.

**Workspace:** `C:\Users\erkanrzgc\Desktop\Jellyfish` (renamed from `Desktop\Octopus` on 2026-11-23; `.venv` rebuilt, Claude project memory moved).
**GitHub:** `erkanrzgc/jellyfish`, branch `main` at `f6bd6bd` (PR #19 `feat/gemma4-dpo-prep` merged); the stage-2 run is on branch `feat/gemma4-dpo-run`.

## Current state

- **Rename done everywhere:** local folder, `.venv`, Claude memory, GitHub repo, Hugging Face repos (`erkanrzgcc/jellyfish-v0.8.1`, `-v0.9`, `-v1.0`, `-v1.1`), Ollama tag (`jellyfish-v10`, same weights as the old `octopus-v10`), and code identifiers (`JELLYFISH_*` prompts, `jellyfish-lab`/`jellyfish-target` Docker defaults, `models/jellyfish-v8-tokenizer`).
- **Old names kept on purpose:** the legacy v1.0 training prompt text in `data/sft/persona.py` still says "Octópus" because v1.0 was trained on it. Dated eval reports under `eval/reports/` keep historical model names. Do not rewrite them.
- **Default runtime:** stock `gemma4:e4b` with native tool calling (`agent/config.py`, user decision 2026-11-22). v1.0 / `jellyfish-v10` remains the legacy fine-tuned path; v1.1 and v1.2 were not promoted.
- **Logo:** `assets/jellyfish.png` is the orange Flaticon jellyfish (sha256 prefix `981693fbaed8`), credited at the end of `README.md`. A pink icon was briefly merged in PR #11 by mistake and reverted in PR #12.
- **DPO data (EVREN teacher):** finished 2026-11-23 02:45: 758 pairs (train 678 / val 80) in gitignored `data/dpo/out/`. Pairs use the v1.0 ```` ```arac``` ```` format, not gemma4's native tool format.
- **On-policy gemma4 DPO pairs (2026-11-24, $0):** 339 pairs (train 305 / val 34) in gitignored `data/dpo/out/native_pairs/` (`pairs_train.jsonl`, `pairs_val.jsonl`, `tools.json`, `manifest.json`). `rejected` is gemma4:e4b's own failing reply under the current `NATIVE_SYSTEM_PROMPT` (native `tool_calls`); `chosen` is an EVREN/teacher answer converted to native `tool_calls` and re-checked by the same structural gate. Built by `data/dpo/native_pairs.py`.
  - **Keep `data/dpo/out/native_train/chosen_extra.jsonl`** (121 EVREN answers, gitignored, only on this disk). Everything else in `data/dpo/out/native_train/` and `native_pairs/` can be rebuilt locally for $0, but these answers cannot be regenerated for free after 2026-12-01. Back it up before any disk cleanup.
  - Reply files are not tied to the prompt automatically: if `NATIVE_SYSTEM_PROMPT` changes, regenerate both `gemma4_e4b_{train,val}.jsonl` before `build` (the manifest's `system_prompt_sha256` records which prompt the pairs assume).
- **gemma4 DPO trainer (stage-2 prep, 2026-11-24, $0):** `train/dpo_gemma4.py` + `cloud/pod_dpo_gemma4.sh` + `models/Modelfile.g4-{dpo,requant}`. Base `google/gemma-4-E4B-it` pinned at revision `4ecc2ba6…` (public, Apache-2.0, not gated); its chat template is vendored in `train/templates/gemma4_e4b_chat_template.jinja` and the trainer aborts if the tokenizer's template differs.
  - **Prompts match Ollama exactly.** HF template + `native_tools()` gave the same prompt token count as Ollama's `prompt_eval_count` on all 2193 DPO cases. Ollama's own renderer tests use the same template (`gemma4_e2b_chat_template.jinja2` in Ollama 0.40.16 differs only in one comment character), and `gemma4-small` = `gemma4` renderer flags.
  - **Completions are not rendered by the stock template.** The template (and Ollama's history renderer) writes an assistant turn as tool calls first, prose second; raw `/api/generate` shows gemma4 generates prose, a blank line, then `<|tool_call>…<tool_call|>`, and stops on `<|tool_response>` (id 50; `<turn|>` id 106 for plain answers). `patch_template` moves prose before calls with `\n\n` (586 of 784 replies) and ends the final model turn on its stop token without the trailing newline; prompts render unchanged (tested).
  - **Rows are tokenized by the trainer, not by TRL** (`_prepare_dataset` pass-through): TRL 1.18.2 appends `<eos>` to text completions, and Arrow would merge per-tool argument dicts into one struct with null-filled keys. Pre-flight checks one BOS, a stop token at the end of every completion, and length (prompts 6943–7016 tokens because of 130 tool declarations; longest row 8405 ≤ `MAX_LENGTH` 8704). TRL 1.18.2 computes log-probs with its fused LM head (no full 262k-vocab logits) and handles gemma4's logit softcap.
  - **LoRA only on the language model** (`TARGET_MODULES` regex): the vision and audio towers also have `*_proj` modules. 294 modules (42 layers × 7).
  - **Deployment path: merged full GGUF, not an adapter.** Ollama 0.40.16 rejects new LoRA adapters (`errAdaptersUnsupported`, "LoRA adapters are no longer supported", in `server/create.go`; old adapter layers are only carried over), and on Windows it cannot `ollama create` from safetensors (`MLX runtime is not available`; that path only offers int4/int8/nvfp4/mxfp4/mxfp8). It does import a gemma4 GGUF with `RENDERER gemma4` / `PARSER gemma4` (checked with the stock blob, text only). The stock `gemma4:e4b` blob is a llama.cpp GGUF (`blk.N.attn_q`…, imatrix Q4_K_M, separate projector and MTP draft). So the pod merges the LoRA into the full `Gemma4ForConditionalGeneration` and converts it with llama.cpp v1.0.0 (`convert_hf_to_gguf.py`, supports Gemma 4) to Q4_K_M without imatrix. To keep quantization out of the comparison, the stock weights go through the same convert+quantize (`CONTROL=1`): the DPO effect is `jellyfish-g4-dpo` vs `jellyfish-g4-requant` (`models/Modelfile.g4-{dpo,requant}`), with Ollama's `gemma4:e4b` (175/212) as reference. Ollama logs that adapters are deprecated, so do not plan on adapters with later Ollama versions either.
  - The stock HF weights (16 GB) are in gitignored `models/gemma-4-E4B-it-hf/`; the tokenizer/template files in `models/gemma-4-E4B-it-tokenizer/`. Both can be deleted and re-downloaded at the pinned revision.
- **gemma4 DPO stage 2 ran (2026-11-25, user-approved, $1.89 of the $3 cap; RunPod balance left $1.44):** A40 SECURE ($0.59/h), pod `30x8ugzb98by3d`, deleted at 01:38 UTC.
  - **1 epoch, not 2.** The smoke step measured 214 s/step (8 pairs of ~7k-token prompts, policy + reference passes, gradient checkpointing), so 2 epochs (78 steps) would have been ~$3.5 in total and exceeded the cap and the balance; the user chose 1 epoch (39 steps, 2 h 23 min). LR 2e-5, beta 0.1, LoRA r=32 unchanged. `train/dpo_gemma4.py` now defaults to 1 epoch and evaluates once after training (the epoch-end eval duplicated it).
  - **Result: not promoted.** 212 val cases: DPO 175/212 = requant control 175/212 = Ollama `gemma4:e4b` 175/212; DPO vs control +8 −8 (noise). Fewer missing calls, more wrong tools/parameters (see Verification snapshot and Next step 3). Default stays stock `gemma4:e4b`.
  - **Training:** loss 0.693 → 0.009–0.013 by step 30; final mean 0.208. On the 34 val pairs: eval loss 0.021, reward accuracy 1.00, margin 7.56, peak VRAM 43.4 GB. The pairs separate very easily (teacher answers vs the model's own failures), so the pair metrics say little about the 212-case behaviour.
  - **Artifacts:** adapter + `run_manifest.json` + tokenizer files + pod logs archived in private HF `erkanrzgcc/jellyfish-g4-dpo` (logs under `logs/`); local copies in gitignored `checkpoints_sft/jellyfish-g4-dpo-adapter/` and `checkpoints_sft/jellyfish-g4-dpo-logs/`. GGUFs (gitignored, not on HF): `models/jellyfish-g4-dpo-Q4_K_M.gguf` (sha256 `16f7113e…`, 5.30 GB) and `models/gemma4-e4b-requant-Q4_K_M.gguf` (control, `01dbed11…`, 5.34 GB). Ollama: `jellyfish-g4-dpo`, `jellyfish-g4-requant`.
  - **Merged GGUF has 666 tensors, the control 720.** transformers 5 does not build `k_proj`/`v_proj`/`k_norm` for the 18 KV-shared layers (24–41), so the merged save drops them (LoRA was on 258 modules, not 294). They are dead weights; llama.cpp v1.0.0 marks them `TENSOR_NOT_REQUIRED` for layers without their own KV, and Ollama 0.40.16 loads the DPO GGUF and parses its tool calls.
  - **Pod lessons:** the `runpod-torch-v240` template's torchvision 0.19.1 breaks `import transformers` on torch 2.6 (recipe now pins torchvision 0.21.0 / torchaudio 2.6.0); the pod's own `runpodctl` had no API key, so the in-pod watchdog cannot stop the pod; locally, `runpodctl` 2.5 `pod list/delete` fail (`rest.runpod.io` does not resolve) while the deprecated `get/create/remove pod` work (`create` needs `MSYS_NO_PATHCONV=1` in Git Bash). The stock control conversion ran on CPU during training. scp from the pod ran at 6–19 MB/s. Next time compare a community RTX A6000 price first.

## Changes in the 2026-11-23/24 sessions

| Item | Result |
|---|---|
| PR #11 `feat/evren-preference-data` → `main` | Merged as `92c496b`: gemma4:e4b default, native chat backend, constrained call repair, router fold, DPO data pipeline (`data/dpo/`), DPO trainer (`train/dpo_bf16.py`), pod recipe `cloud/pod_dpo_v13.sh`, base-model bake-off, rename commits |
| CI fix (`6a6f8b7`) | `.github/workflows/ci.yml` now installs `pyyaml` and `jinja2` besides pytest. `main` CI had been red since 2026-09 because `agent/skills.py` imports `yaml`. `tests/train/test_dpo_render.py` now skips the real-tokenizer test when `transformers` is absent |
| PR #12 `fix/logo-icon` | Merged as `68c4a06`; restores the orange logo |
| Branches | `feat/evren-preference-data` and `fix/logo-icon` deleted locally and on GitHub |
| Docker path without Kali (PR #15) | `DockerExecutor` and `_docker_container_ip` now call `docker` directly; optional `docker_prefix` (same pattern as `sandbox_executor`) can still route through a WSL distro. Kali WSL is no longer installed on this PC, so the old `wsl -d kali-linux` prefix failed with `WSL_E_DISTRO_NOT_FOUND`. `RealExecutor` (`--real`, tools run natively inside Kali) still requires Kali WSL |
| nmap skill aligned with policy (PR #17) | `agent/arguments.py` now also allows `-p-` (all ports). `agent/skills/tools/nmap.md` recommends only allowlisted flags (typical `-sV -Pn -p-`) and lists the rejected ones (`-sC`/`--script`, `-A`, `-O`, `-T<n>`, `-oN`/`-oX`). New `tests/agent/test_arguments.py` locks the skill examples to the allowlist |
| On-policy gemma4 pairs (branch `feat/gemma4-native-pairs`) | `eval/native_tool_bakeoff.py --split train` and `--split val --policy` write gemma4 replies to `data/dpo/out/native_train/gemma4_e4b_{train,val}.jsonl`, never to the bake-off file. `data/dpo/native_pairs.py` (`gaps`, `build`) pairs failures with teacher answers. A chosen answer is sent back to EVREN if it still says Octópus/Octopus or fails the gate after conversion. Pairs where both sides call `run_cmd` and only the command string differs (`free` vs `free -h`, quote style, `head` vs `head -n 10`; some labeled commands like `htop`/`watch` were worse than the rejected one) are dropped: 43 such pairs. A few real errors went with them (an IP passed to `df`/`iostat`, interactive `top -o %CPU`); a later rule could keep `run_cmd` pairs whose rejected command contains a host or IP. EVREN filled 121 missing answers (1 dropped, 1 unusable after retry). |
| PR #18 `feat/gemma4-native-pairs` | Merged as `8a35fcf`; branch deleted locally and on GitHub |
| gemma4 DPO prep (branch `feat/gemma4-dpo-prep`) | New `train/dpo_gemma4.py` (render, patch template, tokenize, pre-flight, LoRA DPO with TRL 1.18.2), vendored template, `cloud/pod_dpo_gemma4.sh` (pinned env, pre-flight, 2-step smoke with peak VRAM and s/step, train, merge, llama.cpp Q4_K_M for the DPO model and a stock control), `models/Modelfile.g4-{dpo,requant}`, 20 tests in `tests/train/test_dpo_gemma4.py` (CI renders with a jinja2 environment that mirrors transformers; one test uses the real tokenizer when present). `train/dpo_bf16.render_pair` takes optional template kwargs (v1.0 path unchanged). `*.jinja` pinned to LF in `.gitattributes`; `models/gemma-4-*/` ignored |
| PR #19 `feat/gemma4-dpo-prep` | Merged as `f6bd6bd`; branch deleted locally and on GitHub |
| gemma4 DPO run (branch `feat/gemma4-dpo-run`) | `train/dpo_gemma4.py`: 1 epoch default, single eval after training. `cloud/pod_dpo_gemma4.sh`: torchvision/torchaudio pins, watchdog caveat. Handoff results below |
| Docker lab renamed | Created network `jellyfish-lab` and container `jellyfish-target` (`bkimminich/juice-shop`, port 3000, no host port binding); removed old `octopus-target` and `octopus-lab`. `docs/DEMO_GUIDE.md` now shows the lab setup commands and `--port 3000` |

## Verification snapshot (2026-11-24/25)

| Command | Observed result |
|---|---|
| `uv run pytest -q` | 590 passed (on `feat/gemma4-dpo-prep`) |
| HF `google/gemma-4-E4B-it` template + `native_tools()` vs Ollama `prompt_eval_count` | 2193/2193 identical prompt token counts |
| `uv run python -m train.dpo_gemma4 --check-only` (local, transformers 5.12.1, no torch) | 305 train + 34 val pairs, 130 tools, longest row 8405 tokens; pre-flight passed |
| Raw `/api/generate` with the HF-rendered prompt vs `/api/chat` replies (temperature 0, seed 3407) | Same prompt token counts; generation order prose → `\n\n` → call → stop confirmed. Texts diverge mid-answer between the two endpoints (1/9 identical), so Ollama temperature-0 output is not reproducible across call paths; single-run eval differences of a few cases are noise |
| `ollama create` from the stock safetensors (`--quantize q4_K_M`, then `int4`) | Fails: q4_K_M unsupported on this path; int4 needs the MLX runtime, which is not available on Windows |
| Ollama 0.40.16 source (`server/create.go`, `llm/llama_server.go`) | `ollama create` with `ADAPTER` or an adapter GGUF returns "LoRA adapters are no longer supported"; the runner still passes `--lora` to llama-server only for adapter layers carried over from old models |
| `ollama create jellyfish-g4-textonly-test` from the stock `gemma4:e4b` model blob alone + `RENDERER gemma4` / `PARSER gemma4` | Created; `ollama show` lists renderer `gemma4-small`, parser `gemma4`, capabilities completion/tools/thinking (no vision: projector left out) |
| same model, 212 val cases (`--model jellyfish-g4-textonly-test --split val --policy`, 23 min) | **174/212** (tool 103/139, no-tool 71/73) vs `gemma4:e4b` 175/212 (106/139, 69/73). Same weights and temperature 0, yet only 100/212 replies are identical: the total moved by 1 but the categories by 2–3. That is the measured noise floor for one run; native tool parsing works on an imported GGUF |
| llama.cpp converter run locally | Not run: the local tool-permission check blocked running llama.cpp's Python converter on this PC; the pod script runs llama.cpp instead |
| gemma4:e4b on the 1981 DPO train cases, current prompt (`--split train`) | 1634 passed; tool cases 952/1268, no-tool 682/713; missing_call 135, wrong_first_tool 102, param_mismatch 74, unnecessary_call 31, catalog 7 |
| gemma4:e4b on the 212 val cases, current prompt (`--split val --policy`) | **175/212** (tool 106/139, no-tool 69/73). This is the pre-DPO baseline; the 177/212 in the 2026-11-22 bake-off was measured under an older prompt (every reply said "Octópus") |
| `python -m data.dpo.native_pairs build` | 339 pairs (305/34), 0 Octópus in chosen or rejected, 1 missing + 1 unusable chosen, 43 command quibbles dropped |
| Clean Python 3.12 venv with only `pytest`, `pyyaml`, `jinja2` (mirrors CI), `pytest -q` | 548 passed, 1 skipped |
| GitHub Actions `test` on `6a6f8b7` | pass |
| `uv run python -m agent.smoke` | 3/3 offline checks passed |
| `docker run --rm --network jellyfish-lab instrumentisto/nmap -Pn -p 3000 jellyfish-target` (executor argv without the `wsl` prefix) | 1 host up, `3000/tcp open` |
| `echo evet \| uv run python -m agent.cli --docker --target jellyfish-target --port 3000 --scope jellyfish-target` | approval prompt shown; real `docker run ... instrumentisto/nmap -Pn -sV -p3000 jellyfish-target`; `3000/tcp open`, host 172.25.0.2; `durum=completed` |
| `uv run python -m agent.cli --gguf --docker --target jellyfish-target` (gemma4:e4b native, Ollama 0.40.16), before PR #17 | IP resolved to 172.25.0.2; model called `nmap -sV -sC`, the nmap skill guide was injected, model re-called `-sV -sC -Pn -p-`; policy denied `-sC` (`agent/arguments.py`) before approval; `durum=denied`, 0 tools executed, 17 s. Persona line printed unprompted |
| same command, after PR #17 | model first called `nmap -sV -sC`, read the updated skill, re-called `-sV -Pn` (allowed); 2 approvals; real nmap ran twice with identical args; `3000/tcp open` on 172.25.0.2; model identified a Juice Shop-like web app and suggested `whatweb`/`nuclei`; `durum=completed`, 4 steps, 2 calls, 51 s |
| Pod [0]–[2] (`--check-only`, `--smoke-steps 2`, torch 2.6.0+cu124, transformers 5.12.1, TRL 1.18.2, PEFT 0.22.0, A40) | pre-flight passed on the pod; `Gemma4ForConditionalGeneration` trains with TRL's fused LM head; LoRA on 258 language-model modules, 69.8M trainable params; 2 steps: loss 0.6931, peak VRAM 39.3 GB, 214 s/step |
| Pod control arm: stock snapshot → llama.cpp v1.0.0 `convert_hf_to_gguf.py` (f16, transformers 5.12.1) → `llama-quantize Q4_K_M` | 720 tensors, 15.0 GB f16 → 5073 MiB Q4_K_M (5.66 BPW), no errors; ran on CPU while the GPU trained |
| Pod DPO arm: 1 epoch (39 steps) → merge on CPU → same convert + quantize | train 8597 s, final mean loss 0.208, val-pair eval loss 0.021 / reward accuracy 1.00 / margin 7.56, peak VRAM 43.4 GB; merged GGUF 666 tensors (KV-shared k/v dropped, see Current state), 5042 MiB Q4_K_M |
| `ollama create jellyfish-g4-requant` / `jellyfish-g4-dpo` (Modelfiles in `models/`) + one nmap prompt each | both import (renderer `gemma4-small`, parser `gemma4`, tools capability) and return a native `nmap` call (`hedef` 10.10.10.5, `-sV`) |
| `jellyfish-g4-requant`, 212 val cases (`--split val --policy`) | **175/212** (tool 105/139, no-tool 70/73), same total as Ollama `gemma4:e4b`. vs `gemma4:e4b`: +9 −9 cases, only 9/212 identical replies; so losing the imatrix costs nothing measurable here, and single-run case flips of ±9 are noise |
| `jellyfish-g4-dpo`, 212 val cases (`--split val --policy`) | **175/212** (tool 106/139, no-tool 69/73). vs the requant control: +8 −8 cases (net 0), 12/212 identical replies; vs `gemma4:e4b` +10 −10. Failure mix moved: missing_call 15 → 6, but wrong_first_tool 10 → 13, param_mismatch 8 → 12, catalog 1 → 3. Replies are shorter (mean 912 vs 1239 chars; 8 vs 23 hit the length limit). No Octópus mentions. Not promoted |
| `ollama list` | `jellyfish-v10`, `gemma4:e4b`, `gemma4:12b`, `qwen3:8b`, `qwen3:4b`, plus the test import `jellyfish-g4-textonly-test` (shares the gemma4:e4b blob; safe to `ollama rm`), `jellyfish-g4-requant`, `jellyfish-g4-dpo` |

The `agent.cli --docker` row uses the scripted model with real nmap. The two `--gguf --docker` rows are the only live-model runs in these sessions. The only spending was the 2026-11-25 DPO pod ($1.89).

## Resume commands

From the repository root:

```powershell
git status --short
uv run python -m agent.smoke
uv run pytest -q
```

Fallback with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m agent.smoke
.\.venv\Scripts\python.exe -m pytest -q
```

For a live check follow `models/README.md` and `docs/DEMO_GUIDE.md`. Review actual calls and tool results, not just the CLI exit code.

## Next steps, in order

1. **Live demo follow-ups:** the real-model/real-tool path now completes. Two observations from that run: the model still starts with `-sC` and only drops it after the skill guide is injected (one wasted turn), and it repeated the identical nmap call twice. Consider a duplicate-call guard in `run_tool_loop` and measure on more than one prompt before changing defaults.
2. **Persona and evidence push (from the 2026-11-22 plan):** make gemma4 introduce itself only when asked; decide the default for the opt-in `evidence_push` (gemma4 RCA 0/5 base, 5/5 with push, but push costs 3 no-tool val cases).
3. **gemma4 DPO: no gain from the first run (2026-11-25).** 1 epoch on 305 native pairs left the 212-case score at 175/212, equal to the requant control and stock `gemma4:e4b`; the case flips (+8/−8) are within the measured noise. It did change behaviour: fewer missing tool calls (15 → 6) but more wrong first tools and wrong parameters, so the pairs teach "call a tool" more than "call the right tool with the right arguments". Before paying for another run: (a) inspect the 8 lost and 8 gained cases (`data/dpo/out/native_train/jellyfish-g4-{dpo,requant}_val.jsonl`) and the chosen answers behind the wrong_first_tool/param_mismatch pairs; (b) consider pairs that target tool choice and arguments specifically, or a lower beta / fewer missing_call pairs; (c) more epochs alone are unlikely to fix tool choice and cost ~$1.8 per epoch on an A40 (check a community RTX A6000 price first). RunPod balance $1.44. Free EVREN inference ends **2026-12-01**; back up `data/dpo/out/native_train/chosen_extra.jsonl` before then. v1.0 fallback `cloud/pod_dpo_v13.sh` (~$1.4, 758 legacy pairs) is still unrun.
4. **Real nmap integration:** scripted-model/real-tool and real-model/real-tool both pass in the local lab. Only nmap has a Docker image (`TOOL_IMAGES`); `whatweb`/`nuclei` suggested by the model would return "imaj yok". Scan only the local authorized lab.
5. **Per-tool CLI adapters and eval depth:** lab-test a few tools before claiming broad execution support; extend the held-out set to argument correctness, all emitted calls, and varied refusal probes.
6. **Artifact reproducibility:** any future training must keep the adapter, tokenizer, pinned environment, data manifest/split, and eval outputs before the pod is deleted.

## Known limitations to retain

- The three offline smoke checks use a scripted model and simulated scanning; only their temporary-workspace file operations are real.
- The normal CLI may exit zero when its transcript contains a backend error; the smoke runner has an explicit failure status.
- Catalog breadth is not integration coverage. Free-form `secenekler` and tool-specific argument/target behavior need validation; `shell=False` does not establish complete policy enforcement.
- The historical safety report includes a refusal followed by a tool call in one probe.
- `serve/jellyfish_chat.py` is a legacy Qwen/Unsloth entry point, not the current runtime.
- `CLAUDE.md` is Git-ignored and kept locally. `AGENTS.md` and this handoff are normal repository files.
- The old Claude memory folder `C:\Users\erkanrzgc\.claude\projects\C--Users-erkanrzgc-Desktop-Octopus` still exists next to the new one; it is harmless and can be removed once nothing references it.

## History pointers

- 2026-10-23 runtime/handoff work: `docs/superpowers/plans/2026-10-23-runtime-and-handoff.md`, `docs/superpowers/plans/2026-10-23-cybersecurity-reliability.md`.
- 2026-11-07 disk cleanup: only `jellyfish-v10` GGUF kept locally; v0.8.1/v0.9/v1.1 GGUFs on Hugging Face; v0.7/v0.8/v1.2 only as local adapters in `checkpoints_sft/`.
- 2026-11-21 EVREN preference data: `docs/superpowers/plans/2026-11-21-evren-preference-data.md`. The API key is read from `EVREN_API_KEY` / `HKCU\Environment` and must never be printed. EVREN training GPUs are image-only, so LLM training needs another provider.
- 2026-11-22 harness + base bake-off and live progress log: `docs/superpowers/plans/2026-11-22-harness-and-base-bakeoff.md`; report `eval/reports/gemma4_e4b_bakeoff_2026-11-22.md`.

## Agent-switch prompt

> `AGENTS.md` ve `PROJECT_HANDOFF.md` dosyalarını oku, mevcut git diff'ini kontrol et ve sıradaki adımdan devam et. Yapılan işleri tekrarlama; yeni doğrulama sonuçlarını ortak devir dosyasına yaz.
