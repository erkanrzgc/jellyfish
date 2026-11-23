# Repeatable runtime demos

Run every command from the repository root. The default model is `jellyfish-v10`; the evaluated tokenizer is `models/jellyfish-v8-tokenizer`. Runtime defaults are defined in `agent/config.py`.

## 1. Offline acceptance checks

```powershell
uv run python -m agent.smoke
```

| Check | What actually runs | Success condition |
|---|---|---|
| `mock-scan` | Scripted model -> parser/loop -> registry/policy -> mock nmap | Exactly one simulated execution and expected port output |
| `scope-denial` | Same loop, target outside the configured allowlist | Denial returned and security executor call count unchanged |
| `workspace-files` | Real file write/read in a temporary workspace | Stored/read content matches; path escape denied and no escaped file exists |

Expected summary: `3/3 offline checks passed`. Exit code is zero only when every check passes. The runner deletes its temporary workspace and audit logs. It does not use the GPU, model, network, WSL, or Docker.

For the longer demo transcripts:

```powershell
uv run python -m agent.cli
uv run python -m agent.cli --assistant
```

The normal CLI logs to ignored `lab/logs/audit.jsonl`. Its default scan uses simulated results. The assistant demo writes a note into its own temporary directory and demonstrates a rejected `../` path; that older demo leaves its directory available for inspection.

## 2. Real model, simulated scan

Prerequisites: local Ollama service, the v1.0 Q4 GGUF, and the evaluated tokenizer files. See [`models/README.md`](../models/README.md).

```powershell
ollama create jellyfish-v10 -f models/Modelfile.v10
uv run python -m agent.cli --gguf --target 10.10.10.5 --scope 10.10.10.0/24
```

Expected: the actual model generates a valid `nmap` call for the allowed target, receives **mock** port output, and interprets it. Skill injection may cause a regeneration before execution; a dropped call is cancelled. Record the actual call and final response. The CLI now reports task status and returns nonzero on failure or an unfinished task; an exit code alone does not establish model quality.

Explicit model/template selection remains available (the override model must already be imported into Ollama; see [models/README.md](../models/README.md)):

```powershell
uv run python -m agent.cli --gguf --model jellyfish-v9 --tokenizer-dir models/jellyfish-v8-tokenizer
```

An initial 2026-10-23 service check failed. The user subsequently started Ollama and live v1.0 inference was recorded. See `PROJECT_HANDOFF.md` for current report paths and results; service availability remains a per-run prerequisite.

## 3. Real nmap in an existing Docker lab

This is an optional live integration check, separate from the three offline checks. Prerequisites:

- A functioning Docker engine reachable from the `kali-linux` WSL distribution.
- A prepared `jellyfish-lab` Docker network and an authorized target container named `jellyfish-target`, exposing a known service. The reference lab uses `bkimminich/juice-shop`, which listens on port 3000; pass `--port` to match your target:

  ```powershell
  docker network create jellyfish-lab
  docker run -d --name jellyfish-target --network jellyfish-lab bkimminich/juice-shop
  ```
- The executor's `instrumentisto/nmap` image available to that engine.

Inspect your lab before running a scan:

```powershell
wsl -d kali-linux -- docker network inspect jellyfish-lab
wsl -d kali-linux -- docker inspect jellyfish-target
```

Then run the scripted-model integration check:

```powershell
uv run python -m agent.cli --docker --target jellyfish-target --port 3000 --scope jellyfish-target
```

Expected: a target/command preview and approval prompt. Enter `evet` or `yes` to approve that exact call, then inspect actual nmap output for the known target. Declining or an engine error leaves the task unfinished. Inspect the audit log for the target, decision, and actual execution. The Docker executor's built-in image map currently covers only nmap; the larger catalog does not imply Docker support for all tools.

After both the real-model/mock-tool check and the scripted-model/real-tool check pass, the combined path can be evaluated:

```powershell
uv run python -m agent.cli --gguf --docker --target jellyfish-target
```

The combined demo resolves the container IP and uses the existing lab scope defaults. Record its concrete IP and ensure it belongs to your intended lab before interpreting results. No Docker lab was created or scanned in the 2026-10-23 session.

## Record results when switching agents

Update `PROJECT_HANDOFF.md` with the date, exact command, observed result, model/template pair, and remaining work. Distinguish offline runtime checks, live model quality, and live tool execution. Preserve earlier evaluation reports; new experiments should get their own labels and reports.
