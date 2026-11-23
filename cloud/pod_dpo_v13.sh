#!/bin/bash
# Octopus v1.3 = v1.0 SFT repro + DPO, one pod session, Q4_K_M GGUF out. No HF token on the pod:
# adapters + GGUF are staged in $OUT and pulled with scp; HF upload happens locally afterwards.
# Prerequisites staged under /workspace/octopus (scp, no git push):
#   train/sft_bf16.py train/dpo_bf16.py data/__init__.py data/sft/{__init__,normalize,persona}.py
#   data/sft/train.jsonl   (fb2e0c9 repro: build_sft --source distill seed_tr tools --seed-repeat 3,
#                           train=3231 sha e633b060...)
#   data/dpo/out/pairs/pairs_{train,val}.jsonl
# Self-stop watchdog: the pod stops itself after MAX_HOURS even if this script hangs.
set -eo pipefail
MAX_HOURS="${MAX_HOURS:-5}"
REPO=/workspace/octopus; BASE="ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
SFT=/workspace/v10-sft-adapter; DPO=/workspace/v13-dpo-adapter
MERGED=/workspace/v13-merged; OUT=/workspace/octopus-v13-out
mkdir -p "$OUT"
nohup sh -c "sleep $((MAX_HOURS * 3600)); runpodctl stop pod \$RUNPOD_POD_ID" > /workspace/watchdog.log 2>&1 &
echo "watchdog: stop after ${MAX_HOURS}h"

echo "===== [0/6] pinned env (v1.0-identical) ====="
pip install -q "transformers==4.49.0" "peft==0.14.0" "trl==0.15.2" "accelerate==1.4.0" "torch==2.4.1" \
  datasets sentencepiece protobuf 2>&1 | tail -1
cd "$REPO"
echo "$(sha256sum data/sft/train.jsonl)"; wc -l data/sft/train.jsonl data/dpo/out/pairs/*.jsonl

echo "===== [1/6] SFT repro (v1.0 recipe: full data, 1570 steps) ====="
python train/sft_bf16.py --base "$BASE" --max-train 0 --max-steps 1570 --out "$SFT" --no-gen
cp -r "$SFT" "$OUT/v10-sft-adapter"

echo "===== [2/6] DPO pre-flight + train ====="
python train/dpo_bf16.py --base "$BASE" --sft-adapter "$SFT" --out "$DPO" --check-only
python train/dpo_bf16.py --base "$BASE" --sft-adapter "$SFT" --out "$DPO"
cp -r "$DPO" "$OUT/v13-dpo-adapter"

echo "===== [3/6] merge base + SFT + DPO ====="
python - <<PY
import torch, shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download
m = AutoModelForCausalLM.from_pretrained("$BASE", torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, "$SFT").merge_and_unload()
m = PeftModel.from_pretrained(m, "$DPO").merge_and_unload()
try: m.generation_config.do_sample = True
except Exception: pass
m.save_pretrained("$MERGED", safe_serialization=True)
AutoTokenizer.from_pretrained("$SFT").save_pretrained("$MERGED")
shutil.copy(hf_hub_download("$BASE", "tokenizer.model"), "$MERGED/tokenizer.model")
print("MERGE_OK")
PY

echo "===== [4/6] llama.cpp ====="
pip install -q cmake 2>&1 | tail -1
cd /workspace; [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp; pip install -q -r requirements.txt 2>&1 | tail -1
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF 2>&1 | tail -1
cmake --build build --config Release -j --target llama-quantize 2>&1 | tail -2

echo "===== [5/6] GGUF f16 (container disk) -> Q4_K_M ====="
python convert_hf_to_gguf.py "$MERGED" --outfile /root/v13-f16.gguf --outtype f16
rm -rf "$MERGED"
./build/bin/llama-quantize /root/v13-f16.gguf "$OUT/jellyfish-v13-Q4_K_M.gguf" Q4_K_M
rm -f /root/v13-f16.gguf

echo "===== [6/6] staged for scp ====="
ls -la "$OUT"; sha256sum "$OUT/jellyfish-v13-Q4_K_M.gguf"
echo "V13_DONE"
