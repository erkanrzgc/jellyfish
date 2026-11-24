# Jellyfish shared project handoff

> Read this first when switching between Codex and Claude. This file is the shared current state; it replaces reliance on chat history. Update it at the end of meaningful work.

**Last updated:** 2026-11-24 (Europe/Istanbul), Claude. The Octopus → Jellyfish rename is complete (including the local Docker lab), CI is green, 339 on-policy gemma4 DPO pairs are built locally ($0), and the gemma4 DPO trainer and pod recipe are prepared and checked locally ($0); paid training has not started.

**Workspace:** `C:\Users\erkanrzgc\Desktop\Jellyfish` (renamed from `Desktop\Octopus` on 2026-11-23; `.venv` rebuilt, Claude project memory moved).
**GitHub:** `erkanrzgc/jellyfish`, branch `main` at `8a35fcf` (PR #18 merged; before the `feat/gemma4-dpo-prep` PR).

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
| Docker lab renamed | Created network `jellyfish-lab` and container `jellyfish-target` (`bkimminich/juice-shop`, port 3000, no host port binding); removed old `octopus-target` and `octopus-lab`. `docs/DEMO_GUIDE.md` now shows the lab setup commands and `--port 3000` |

## Verification snapshot (2026-11-24)

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
| `ollama list` | `jellyfish-v10`, `gemma4:e4b`, `gemma4:12b`, `qwen3:8b`, `qwen3:4b`, plus the test import `jellyfish-g4-textonly-test` (shares the gemma4:e4b blob; safe to `ollama rm`) |

The `agent.cli --docker` row uses the scripted model with real nmap. The two `--gguf --docker` rows are the only live-model runs in these sessions. No training, pod, or spending happened.

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
3. **DPO stage 2 (paid, needs explicit user approval, cap $3):** prep (a) new env + native rendering and (b) template check are done (see Current state). (c) Ollama import: adapters are rejected by Ollama 0.40.16, a gemma4 GGUF with `RENDERER`/`PARSER` imports fine, so the pod merges and converts (see Current state). Unverified until the pod runs: TRL 1.18.2 training on `Gemma4ForConditionalGeneration` (the smoke step checks it before the full run) and llama.cpp's gemma4 conversion of the merged weights (the control arm converts the stock weights the same way). Then: create the pod (A40 SECURE, template `runpod-torch-v240`, volume ≥ 60 GB, container disk ≥ 40 GB), stage the files listed in `cloud/pod_dpo_gemma4.sh`, run it; stop after the smoke step if peak VRAM or s/step would break the $3 cap. Pull back the adapter + `run_manifest.json` (archive them on HF, see the adapter-archival rule) and both Q4_K_M GGUFs (~5.5 GB each). Locally: `ollama create` from `models/Modelfile.g4-dpo` and `models/Modelfile.g4-requant`, then the 212 val cases with `--split val --policy` for each. Success bar: DPO beats the requant control by clearly more than the measured noise (one run moved categories by 2–3 cases with identical weights; rerun both arms if the margin is small) without losing no-tool cases; Ollama `gemma4:e4b` 175/212 stays the reference. v1.0 fallback: `cloud/pod_dpo_v13.sh` (~$1.4) on the 758 legacy pairs. Free EVREN inference ends **2026-12-01**.
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
