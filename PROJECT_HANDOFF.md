# Jellyfish shared project handoff

> Read this first when switching between Codex and Claude. This file is the shared current state; it replaces reliance on chat history. Update it at the end of meaningful work.

**Last updated:** 2026-11-24 (Europe/Istanbul), Claude. The Octopus → Jellyfish rename is complete (including the local Docker lab), PR #11 and PR #12 are merged, CI is green again, and `main` is the only branch.

**Workspace:** `C:\Users\erkanrzgc\Desktop\Jellyfish` (renamed from `Desktop\Octopus` on 2026-11-23; `.venv` rebuilt, Claude project memory moved).
**GitHub:** `erkanrzgc/jellyfish`, branch `main` at `68c4a06`.

## Current state

- **Rename done everywhere:** local folder, `.venv`, Claude memory, GitHub repo, Hugging Face repos (`erkanrzgcc/jellyfish-v0.8.1`, `-v0.9`, `-v1.0`, `-v1.1`), Ollama tag (`jellyfish-v10`, same weights as the old `octopus-v10`), and code identifiers (`JELLYFISH_*` prompts, `jellyfish-lab`/`jellyfish-target` Docker defaults, `models/jellyfish-v8-tokenizer`).
- **Old names kept on purpose:** the legacy v1.0 training prompt text in `data/sft/persona.py` still says "Octópus" because v1.0 was trained on it. Dated eval reports under `eval/reports/` keep historical model names. Do not rewrite them.
- **Default runtime:** stock `gemma4:e4b` with native tool calling (`agent/config.py`, user decision 2026-11-22). v1.0 / `jellyfish-v10` remains the legacy fine-tuned path; v1.1 and v1.2 were not promoted.
- **Logo:** `assets/jellyfish.png` is the orange Flaticon jellyfish (sha256 prefix `981693fbaed8`), credited at the end of `README.md`. A pink icon was briefly merged in PR #11 by mistake and reverted in PR #12.
- **DPO data (EVREN teacher):** finished 2026-11-23 02:45: 758 pairs (train 678 / val 80) in gitignored `data/dpo/out/`. Pairs use the v1.0 ```` ```arac``` ```` format, not gemma4's native tool format.

## Changes in the 2026-11-23/24 sessions

| Item | Result |
|---|---|
| PR #11 `feat/evren-preference-data` → `main` | Merged as `92c496b`: gemma4:e4b default, native chat backend, constrained call repair, router fold, DPO data pipeline (`data/dpo/`), DPO trainer (`train/dpo_bf16.py`), pod recipe `cloud/pod_dpo_v13.sh`, base-model bake-off, rename commits |
| CI fix (`6a6f8b7`) | `.github/workflows/ci.yml` now installs `pyyaml` and `jinja2` besides pytest. `main` CI had been red since 2026-09 because `agent/skills.py` imports `yaml`. `tests/train/test_dpo_render.py` now skips the real-tokenizer test when `transformers` is absent |
| PR #12 `fix/logo-icon` | Merged as `68c4a06`; restores the orange logo |
| Branches | `feat/evren-preference-data` and `fix/logo-icon` deleted locally and on GitHub |
| Docker path without Kali (PR #15) | `DockerExecutor` and `_docker_container_ip` now call `docker` directly; optional `docker_prefix` (same pattern as `sandbox_executor`) can still route through a WSL distro. Kali WSL is no longer installed on this PC, so the old `wsl -d kali-linux` prefix failed with `WSL_E_DISTRO_NOT_FOUND`. `RealExecutor` (`--real`, tools run natively inside Kali) still requires Kali WSL |
| nmap skill aligned with policy (PR #17) | `agent/arguments.py` now also allows `-p-` (all ports). `agent/skills/tools/nmap.md` recommends only allowlisted flags (typical `-sV -Pn -p-`) and lists the rejected ones (`-sC`/`--script`, `-A`, `-O`, `-T<n>`, `-oN`/`-oX`). New `tests/agent/test_arguments.py` locks the skill examples to the allowlist |
| Docker lab renamed | Created network `jellyfish-lab` and container `jellyfish-target` (`bkimminich/juice-shop`, port 3000, no host port binding); removed old `octopus-target` and `octopus-lab`. `docs/DEMO_GUIDE.md` now shows the lab setup commands and `--port 3000` |

## Verification snapshot (2026-11-24)

| Command | Observed result |
|---|---|
| `uv run pytest -q` | 562 passed |
| Clean Python 3.12 venv with only `pytest`, `pyyaml`, `jinja2` (mirrors CI), `pytest -q` | 548 passed, 1 skipped |
| GitHub Actions `test` on `6a6f8b7` | pass |
| `uv run python -m agent.smoke` | 3/3 offline checks passed |
| `docker run --rm --network jellyfish-lab instrumentisto/nmap -Pn -p 3000 jellyfish-target` (executor argv without the `wsl` prefix) | 1 host up, `3000/tcp open` |
| `echo evet \| uv run python -m agent.cli --docker --target jellyfish-target --port 3000 --scope jellyfish-target` | approval prompt shown; real `docker run ... instrumentisto/nmap -Pn -sV -p3000 jellyfish-target`; `3000/tcp open`, host 172.25.0.2; `durum=completed` |
| `uv run python -m agent.cli --gguf --docker --target jellyfish-target` (gemma4:e4b native, Ollama 0.40.16), before PR #17 | IP resolved to 172.25.0.2; model called `nmap -sV -sC`, the nmap skill guide was injected, model re-called `-sV -sC -Pn -p-`; policy denied `-sC` (`agent/arguments.py`) before approval; `durum=denied`, 0 tools executed, 17 s. Persona line printed unprompted |
| same command, after PR #17 | model first called `nmap -sV -sC`, read the updated skill, re-called `-sV -Pn` (allowed); 2 approvals; real nmap ran twice with identical args; `3000/tcp open` on 172.25.0.2; model identified a Juice Shop-like web app and suggested `whatweb`/`nuclei`; `durum=completed`, 4 steps, 2 calls, 51 s |
| `ollama list` | `jellyfish-v10`, `gemma4:e4b`, `gemma4:12b`, `qwen3:8b`, `qwen3:4b` present |

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
3. **DPO training decision:** the 758 pairs target v1.0's ```` ```arac``` ```` format. Since stock gemma4 is now the default, training would need either a converter to gemma4's native tool format or a decision to train v1.0. Any run needs explicit user approval; cap $3 (plan `docs/superpowers/plans/2026-11-22-harness-and-base-bakeoff.md`, task 7). Free EVREN inference ends **2026-12-01**; any extra teacher data must be generated before then.
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
