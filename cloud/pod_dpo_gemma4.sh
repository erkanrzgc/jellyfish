#!/bin/bash
# Jellyfish gemma4 DPO: stock google/gemma-4-E4B-it (pinned revision) + a LoRA trained with TRL DPO on the
# on-policy native pairs, merged and converted to a Q4_K_M GGUF. PAID: create the pod only after explicit user
# approval (cap $3). No HF token on the pod: the base model is public (Apache-2.0, not gated).
#
# Why a full GGUF and a control arm: Ollama 0.40.16 rejects new LoRA adapters ("LoRA adapters are no longer
# supported", server/create.go) and cannot import safetensors on Windows (needs the MLX runtime). It does import a
# gemma4 GGUF with RENDERER/PARSER (checked locally). The stock gemma4:e4b blob is an imatrix Q4_K_M; this pod has
# no imatrix, so the stock weights go through the same convert+quantize as the DPO weights (CONTROL=1) and the
# DPO effect is DPO-requant vs stock-requant on the 212 val cases. Locally: models/Modelfile.g4-{dpo,requant}.
#
# Pod: cached template runpod-torch-v240 (torch 2.4 is upgraded below; TRL 1.18.2 needs torch>=2.6),
# SECURE cloud, A40 48GB, volume >= 60 GB, container disk >= 40 GB (f16 GGUFs are written to /root).
# Step [2/6] runs two real steps and prints peak VRAM and s/step: stop there if the full run breaks the cap.
# Prerequisites staged under /workspace/jellyfish (scp, no git push):
#   train/__init__.py train/dpo_bf16.py train/dpo_gemma4.py train/templates/gemma4_e4b_chat_template.jinja
#   data/dpo/out/native_pairs/{pairs_train.jsonl,pairs_val.jsonl,tools.json,manifest.json}
# Pull back $OUT: the PEFT adapter + run manifest (archive them), logs, and the Q4_K_M GGUF(s).
# Self-stop watchdog: the pod stops itself after MAX_HOURS even if this script hangs.
set -eo pipefail
MAX_HOURS="${MAX_HOURS:-3}"; CONTROL="${CONTROL:-1}"
REPO=/workspace/jellyfish; OUT=/workspace/jellyfish-g4-dpo-out; MERGED=/workspace/g4-merged
BASE=google/gemma-4-E4B-it; REV=4ecc2ba63bcac23ba4a767fc87025b1450f84c9c
export HF_HOME=/workspace/hf-cache
mkdir -p "$OUT"
nohup sh -c "sleep $((MAX_HOURS * 3600)); runpodctl stop pod \$RUNPOD_POD_ID" > /workspace/watchdog.log 2>&1 &
echo "watchdog: stop after ${MAX_HOURS}h"

echo "===== [0/6] pinned env (transformers = the version the rendering was tested with locally) ====="
pip install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip install -q "transformers==5.12.1" "trl==1.18.2" "peft==0.22.0" "accelerate==1.16.0" "datasets==5.0.0" \
  sentencepiece protobuf cmake 2>&1 | tail -1
python -c "import torch, transformers, trl, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'| transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '|', \
torch.cuda.get_device_name(0))" | tee "$OUT/env.txt"
cd "$REPO"
sha256sum data/dpo/out/native_pairs/* train/templates/gemma4_e4b_chat_template.jinja | tee "$OUT/inputs.sha256"

echo "===== [1/6] pre-flight: render, tokenize, BOS/stop/length (no GPU) ====="
python -m train.dpo_gemma4 --check-only

echo "===== [2/6] smoke: two real optimizer steps, peak VRAM, s/step ====="
python -m train.dpo_gemma4 --smoke-steps 2 --out /workspace/g4-smoke 2>&1 | tee "$OUT/smoke.log"

echo "===== [3/6] DPO train ====="
python -m train.dpo_gemma4 --out "$OUT/g4-dpo-adapter" 2>&1 | tee "$OUT/train.log"

echo "===== [4/6] merge: full Gemma4ForConditionalGeneration (towers included, original tensor names) ====="
python - <<PY
import shutil, torch
from pathlib import Path
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText
base = snapshot_download("$BASE", revision="$REV")
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, "$OUT/g4-dpo-adapter").merge_and_unload()
model.save_pretrained("$MERGED", safe_serialization=True)
for f in Path(base).iterdir():  # tokenizer, processor, chat template, generation config
    if f.suffix in (".json", ".jinja") and not (Path("$MERGED") / f.name).exists():
        shutil.copy(f, "$MERGED")
print("MERGE_OK", base)
PY

echo "===== [5/6] llama.cpp v1.0.0: f16 GGUF (container disk) -> Q4_K_M ====="
cd /workspace; [ -d llama.cpp ] || git clone -q --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF 2>&1 | tail -1
cmake --build build --config Release -j --target llama-quantize 2>&1 | tail -1
to_q4() {  # $1 = HF model dir, $2 = output name
  PYTHONPATH=/workspace/llama.cpp/gguf-py python convert_hf_to_gguf.py "$1" --outfile "/root/$2-f16.gguf" --outtype f16 \
    2>&1 | tail -2
  ./build/bin/llama-quantize "/root/$2-f16.gguf" "$OUT/$2-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -2
  rm -f "/root/$2-f16.gguf"
  sha256sum "$OUT/$2-Q4_K_M.gguf" | tee -a "$OUT/inputs.sha256"
}
to_q4 "$MERGED" jellyfish-g4-dpo
rm -rf "$MERGED"
if [ "$CONTROL" = "1" ]; then
  to_q4 "$(python -c "from huggingface_hub import snapshot_download as s; print(s('$BASE', revision='$REV'))")" \
    gemma4-e4b-requant
fi

echo "===== [6/6] staged for scp ====="
ls -la "$OUT" "$OUT/g4-dpo-adapter"
echo "G4_DPO_DONE"
