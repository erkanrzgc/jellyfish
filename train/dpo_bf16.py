"""Jellyfish DPO on top of an SFT LoRA: bf16 base + merged SFT adapter + new LoRA trained with TRL DPO.

The SFT adapter is merged first, so with ref_model=None TRL's reference is the SFT model (adapter off).
Pairs come from `data/dpo/out/pairs/` (TRL conversational records). They are rendered to text here with the
same chat template and BOS handling as `train/sft_bf16.py`, and a pre-flight check aborts before any GPU
step if the tokenized prompt does not start with exactly one BOS (the ADR 0003 double-BOS trap).

Pod (same env as v1.0: transformers 4.49 / peft 0.14 / trl 0.15.2 / accelerate 1.4 / torch 2.4.1):
    python train/dpo_bf16.py --sft-adapter /workspace/v10-sft-adapter --out /workspace/v10-dpo-adapter
Only data helpers are imported at module level, so tests run without torch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS_DIR = ROOT / "data" / "dpo" / "out" / "pairs"
LORA_R = 32
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BETA = 0.1
LEARNING_RATE = 2e-5
MAX_LENGTH = 1536
MAX_PROMPT_LENGTH = 1024
SEED = 3407


def render_pair(record: dict, tokenizer) -> dict:
    """Conversational record -> text prompt/chosen/rejected, exactly as SFT rendered the same turns."""
    prompt = tokenizer.apply_chat_template(record["prompt"], tokenize=False, add_generation_prompt=True)
    row = {"prompt": prompt}
    for side in ("chosen", "rejected"):
        full = tokenizer.apply_chat_template(record["prompt"] + record[side], tokenize=False)
        if not full.startswith(prompt):
            raise ValueError(f"chat template does not extend the prompt for {side} ({record.get('meta')})")
        row[side] = full[len(prompt):]
    return row


def bos_problem(input_ids: list[int], *, bos_id: int) -> str | None:
    """None when the sequence starts with exactly one BOS, else a description of the problem."""
    if not input_ids or input_ids[0] != bos_id:
        return "prompt does not start with BOS"
    if input_ids.count(bos_id) != 1:
        return f"prompt has {input_ids.count(bos_id)} BOS tokens"
    return None


def _load_rows(path: Path, tokenizer) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [render_pair(json.loads(line), tokenizer) for line in lines if line.strip()]


def main() -> None:  # pragma: no cover -- runs on the pod only
    import inspect

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="ytu-ce-cosmos/Turkish-Gemma-9b-v0.1")
    ap.add_argument("--sft-adapter", required=True, help="SFT LoRA merged into the base before DPO")
    ap.add_argument("--pairs", type=Path, default=PAIRS_DIR / "pairs_train.jsonl")
    ap.add_argument("--val-pairs", type=Path, default=PAIRS_DIR / "pairs_val.jsonl")
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true", help="render + BOS pre-flight, then exit")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter)
    train_rows = _load_rows(args.pairs, tokenizer)
    val_rows = _load_rows(args.val_pairs, tokenizer) if args.val_pairs.exists() else []
    problem = bos_problem(tokenizer(train_rows[0]["prompt"], add_special_tokens=False)["input_ids"],
                          bos_id=tokenizer.bos_token_id)
    print(f"[*] pairs train={len(train_rows)} val={len(val_rows)}; first prompt tail: "
          f"{train_rows[0]['prompt'][-80:]!r}")
    if problem:
        sys.exit(f"[X] BOS pre-flight failed: {problem}")
    if args.check_only:
        print("[OK] pre-flight passed")
        return

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager")
    model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
                                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    cfg = DPOConfig(output_dir=str(Path(args.out) / "checkpoints"), beta=args.beta, learning_rate=args.lr,
                    num_train_epochs=args.epochs, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                    max_length=MAX_LENGTH, max_prompt_length=MAX_PROMPT_LENGTH, warmup_ratio=0.1,
                    lr_scheduler_type="linear", logging_steps=5, bf16=True, seed=SEED, report_to="none",
                    save_strategy="no", eval_strategy="epoch" if val_rows else "no",
                    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    params = inspect.signature(DPOTrainer.__init__).parameters
    tok_kwarg = "processing_class" if "processing_class" in params else "tokenizer"
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=Dataset.from_list(train_rows),
                         eval_dataset=Dataset.from_list(val_rows) if val_rows else None, **{tok_kwarg: tokenizer})
    tokenized = trainer.train_dataset[0]
    problem = bos_problem(list(tokenized["prompt_input_ids"]), bos_id=tokenizer.bos_token_id)
    if problem:
        sys.exit(f"[X] BOS check after TRL tokenization failed: {problem}")
    result = trainer.train()
    print(f"[*] final loss {result.training_loss:.4f}; peak VRAM "
          f"{torch.cuda.max_memory_reserved() / 1024**3:.1f} GB")
    if val_rows:
        print(json.dumps(trainer.evaluate(), indent=2))
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[OK] DPO adapter -> {args.out}")


if __name__ == "__main__":
    main()
