#!/bin/bash
# Jellyfish gemma4 DPO: stock google/gemma-4-E4B-it (pinned revision) + a LoRA trained with TRL DPO on the
# on-policy native pairs. PAID: create the pod only after explicit user approval (cap $3).
# No HF token on the pod: the base model is public (Apache-2.0, not gated). The adapter (PEFT and GGUF), its run
# manifest and the logs are staged in $OUT and pulled with scp; locally: ollama create -f models/Modelfile.g4-dpo.
#
# Pod: cached template runpod-torch-v240 (torch 2.4 is upgraded below; TRL 1.18.2 needs torch>=2.6),
# SECURE cloud, A40 48GB. Step [2/5] runs two real steps and prints peak VRAM and s/step before the full run.
# Prerequisites staged under /workspace/jellyfish (scp, no git push):
#   train/__init__.py train/dpo_bf16.py train/dpo_gemma4.py train/templates/gemma4_e4b_chat_template.jinja
#   data/dpo/out/native_pairs/{pairs_train.jsonl,pairs_val.jsonl,tools.json,manifest.json}
# Self-stop watchdog: the pod stops itself after MAX_HOURS even if this script hangs.
set -eo pipefail
MAX_HOURS="${MAX_HOURS:-3}"
REPO=/workspace/jellyfish; OUT=/workspace/jellyfish-g4-dpo-out
export HF_HOME=/workspace/hf-cache
mkdir -p "$OUT"
nohup sh -c "sleep $((MAX_HOURS * 3600)); runpodctl stop pod \$RUNPOD_POD_ID" > /workspace/watchdog.log 2>&1 &
echo "watchdog: stop after ${MAX_HOURS}h"

echo "===== [0/5] pinned env (transformers = the version the rendering was tested with locally) ====="
pip install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip install -q "transformers==5.12.1" "trl==1.18.2" "peft==0.22.0" "accelerate==1.16.0" "datasets==5.0.0" \
  sentencepiece protobuf 2>&1 | tail -1
python -c "import torch, transformers, trl, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'| transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '|', \
torch.cuda.get_device_name(0))" | tee "$OUT/env.txt"
cd "$REPO"
sha256sum data/dpo/out/native_pairs/* train/templates/gemma4_e4b_chat_template.jinja | tee "$OUT/inputs.sha256"

echo "===== [1/5] pre-flight: render, tokenize, BOS/stop/length (no GPU) ====="
python -m train.dpo_gemma4 --check-only

echo "===== [2/5] smoke: two real optimizer steps, peak VRAM, s/step ====="
python -m train.dpo_gemma4 --smoke-steps 2 --out /workspace/g4-smoke 2>&1 | tee "$OUT/smoke.log"

echo "===== [3/5] DPO train ====="
python -m train.dpo_gemma4 --out "$OUT/g4-dpo-adapter" 2>&1 | tee "$OUT/train.log"

echo "===== [4/5] LoRA -> GGUF adapter (llama.cpp v1.0.0) for: FROM gemma4:e4b + ADAPTER ====="
# Ollama 0.40.16 on Windows cannot import safetensors (needs the MLX runtime). The stock gemma4:e4b blob is a
# llama.cpp GGUF (blk.N.attn_q..., imatrix Q4_K_M), so the adapter is converted with llama.cpp and applied on top
# of that exact base: the DPO arm differs from the 175/212 baseline only by the adapter.
cd /workspace; [ -d llama.cpp ] || git clone -q --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp
python -c "from huggingface_hub import snapshot_download as s; print(s('google/gemma-4-E4B-it', \
revision='4ecc2ba63bcac23ba4a767fc87025b1450f84c9c', allow_patterns=['*.json', '*.jinja'], local_dir='/workspace/g4-base-cfg'))"
PYTHONPATH=/workspace/llama.cpp/gguf-py python llama.cpp/convert_lora_to_gguf.py --base /workspace/g4-base-cfg \
  --outtype f16 --outfile "$OUT/jellyfish-g4-dpo-lora.gguf" "$OUT/g4-dpo-adapter" 2>&1 | tail -3
sha256sum "$OUT/jellyfish-g4-dpo-lora.gguf" | tee -a "$OUT/inputs.sha256"

echo "===== [5/5] staged for scp ====="
ls -la "$OUT" "$OUT/g4-dpo-adapter"
echo "G4_DPO_DONE"
