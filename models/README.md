# Local model artifacts

The evaluated runtime default is **v1.0 / `jellyfish-v10`**, with the v0.8 training chat template. Defaults are centralized in [`agent/config.py`](../agent/config.py).

## Required local files

```text
models/
  jellyfish-v10-Q4_K_M.gguf
  jellyfish-v8-tokenizer/
    tokenizer_config.json
    tokenizer.json
    tokenizer.model
    special_tokens_map.json
  Modelfile.v10
```

Weights and tokenizer directories are ignored by Git. A fresh clone contains import recipes, not the model artifacts. Restore the evaluated files from your trusted model archive before using `--gguf`; the offline smoke checks do not need them. Do not substitute a tokenizer solely because its version number is newer.

The existing v1.0 evaluation used `jellyfish-v8-tokenizer`; see the [quality/safety report](../eval/reports/v1.0_quality_safety_2026-08-26.md). `cloud/pod_gguf_v10.sh` describes the historical conversion, but running it requires the original adapter and base. The v1.0 bf16 adapter was not retained; the Q4 GGUF alone is not a replacement for it.

## Import and run

From the repository root, with Ollama installed and running:

```powershell
ollama create jellyfish-v10 -f models/Modelfile.v10
uv run python -m agent.cli --gguf
```

The real model drives the loop while security tools remain mocked. Look for a parsed `nmap` call, simulated port output, and a model response interpreting that output. An error or missing call is not a successful model check.

The backend applies the tokenizer's chat template and sends a raw prompt to Ollama. The Modelfile intentionally has no TEMPLATE/SYSTEM directive.

## Experiments and overrides

v1.1 and v1.2 were not promoted. Keep them separate from `jellyfish-v10`. The unversioned `Modelfile` and `Modelfile.v8`, `.v81`, `.v9`, `.v12` are historical/version-specific recipes, not the current default.

Only `jellyfish-v10` and its GGUF are kept locally. Older GGUFs (v0.8.1, v0.9, v1.1) live in the project's Hugging Face archive; download one into `models/` and run `ollama create` with its Modelfile before using it as an override. v0.7, v0.8 and v1.2 have no GGUF, only local bf16 adapters.

```powershell
uv run python -m agent.cli --gguf --model octopus-v9 --tokenizer-dir models/jellyfish-v8-tokenizer
```

When testing a different model family, supply its matching tokenizer/template explicitly. For demos and real-tool prerequisites, read the [demo guide](../docs/DEMO_GUIDE.md).
