"""Jellyfish DPO for stock gemma4 (google/gemma-4-E4B-it) in its native tool-call format.

Pairs come from `data/dpo/out/native_pairs/` (`data.dpo.native_pairs`). The prompt is the runtime system prompt,
every catalog tool and the user turn; the vendored template renders it exactly as Ollama 0.40.16's gemma4
renderer does (identical prompt token counts on all 2193 DPO cases, 2026-11-24).

Completions need one change. The stock template writes an assistant turn as tool calls first and prose second,
but gemma4 generates prose, a blank line, then the call, and stops on <|tool_response> (raw Ollama output,
2026-11-24). `patch_template` renders completions in generation order and ends a final model turn on its stop
token without the trailing newline. Rows are tokenized here, not by TRL: TRL appends <eos> to text completions
(gemma4 stops on <turn|> or <|tool_response>), and Arrow would merge the per-tool argument dicts into one
struct, filling missing keys with null.

Pod (needs transformers 5.x for Gemma 4; run from the repo root):
    python -m train.dpo_gemma4 --check-only                      # render, tokenize, BOS/stop/length pre-flight
    python -m train.dpo_gemma4 --smoke-steps 2 --out /workspace/g4-smoke   # real forward/backward, peak VRAM
    python -m train.dpo_gemma4 --out /workspace/g4-dpo-adapter
Only data helpers are imported at module level, so tests run without torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from train.dpo_bf16 import bos_problem, render_pair

ROOT = Path(__file__).resolve().parent.parent
BASE = "google/gemma-4-E4B-it"
REVISION = "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"  # pinned 2026-11-24; its chat template is the vendored one
PAIRS_DIR = ROOT / "data" / "dpo" / "out" / "native_pairs"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "gemma4_e4b_chat_template.jinja"
STOP_TOKENS = ("<turn|>", "<|tool_response>")
CALL_SEPARATOR = "\n\n"  # 586 of 784 gemma4 replies with prose and a call put a blank line between them
# Language-model projections only: the vision and audio towers also have q/k/v/o and MLP *_proj modules.
TARGET_MODULES = r"(model\.language_model|model)\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
LORA_R = 32
LORA_ALPHA = 32
BETA = 0.1
LEARNING_RATE = 2e-5
EPOCHS = 2.0
MAX_LENGTH = 8704  # prompts are 6943-7016 tokens (130 tool declarations); longest completion about 1.5k
SEED = 3407

_CONTENT_START = "            {%- set captured_content -%}"
_CONTENT_END = "{%- endset -%}\n\n            {{- captured_content -}}"
_CALLS = ("            {%- if message.get('tool_calls') -%}\n"
          "                {%- for tool_call in message.get('tool_calls') -%}")
_SEPARATOR = ("\n            {%- if message.get('tool_calls') and captured_content | trim | length > 0 -%}\n"
              "                {{- '" + CALL_SEPARATOR.replace("\n", "\\n") + "' -}}\n"
              "            {%- endif -%}\n")
_TURN_CLOSE = ("        {%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}\n"
               "            {{- '<turn|>\\n' -}}")
_TURN_CLOSE_AT_STOP = ("        {%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}\n"
                       "            {{- '<turn|>' -}}\n"
                       "            {%- if not (loop.last and role == 'model') -%}{{- '\\n' -}}{%- endif -%}")


def patch_template(template: str) -> str:
    """The stock template with assistant prose moved before its tool calls (blank line between) and the last
    model turn ending on <turn|>; prompts render unchanged. Raises when the template is not the vendored one."""
    for anchor in (_CONTENT_START, _CONTENT_END, _CALLS, _TURN_CLOSE):
        if template.count(anchor) != 1:
            raise ValueError(f"gemma4 template anchor not found exactly once: {anchor.strip()[:60]!r}")
    start = template.index(_CONTENT_START)
    end = template.index(_CONTENT_END) + len(_CONTENT_END)
    content = template[start:end]
    template = template[:start] + template[end:]
    template = template.replace(_CALLS, content + _SEPARATOR + _CALLS)
    return template.replace(_TURN_CLOSE, _TURN_CLOSE_AT_STOP)


def tokenize_row(row: dict, tokenizer) -> dict:
    """Prompt and completion ids for TRL's preference collator. The prompt+completion text is tokenized whole and
    must start with the prompt's own tokens, so the model is trained on the context it sees at inference."""
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    out = {"prompt_ids": prompt_ids}
    for side in ("chosen", "rejected"):
        ids = tokenizer(row["prompt"] + row[side], add_special_tokens=False)["input_ids"]
        if ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"the prompt tokenizes differently once the {side} completion is appended")
        out[f"{side}_ids"] = ids[len(prompt_ids):]
    return out


def preflight(rows: list[dict], *, bos_id: int, stop_ids: set[int], max_length: int) -> list[str]:
    """Every problem that would make training differ from inference; empty when the rows are usable."""
    problems = []
    for index, row in enumerate(rows):
        problem = bos_problem(row["prompt_ids"], bos_id=bos_id)
        if problem:
            problems.append(f"row {index}: {problem} (BOS)")
        for side in ("chosen", "rejected"):
            ids = row[f"{side}_ids"]
            if not ids or ids[-1] not in stop_ids:
                problems.append(f"row {index} {side}: does not end on a stop token")
            if len(row["prompt_ids"]) + len(ids) > max_length:
                problems.append(f"row {index} {side}: longer than max_length {max_length}")
    return problems


def load_rows(path: Path, tokenizer, *, tools: list[dict], chat_template: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [tokenize_row(render_pair(json.loads(line), tokenizer, tools=tools, chat_template=chat_template),
                         tokenizer) for line in lines if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--pairs", type=Path, default=PAIRS_DIR / "pairs_train.jsonl")
    ap.add_argument("--val-pairs", type=Path, default=PAIRS_DIR / "pairs_val.jsonl")
    ap.add_argument("--tools", type=Path, default=PAIRS_DIR / "tools.json")
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--epochs", type=float, default=EPOCHS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check-only", action="store_true", help="render + tokenize + pre-flight, no model")
    ap.add_argument("--smoke-steps", type=int, default=0, help="train this many steps, report peak VRAM, save nothing")
    args = ap.parse_args(argv)
    if not args.check_only and args.out is None:
        ap.error("--out is required unless --check-only")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    stock = TEMPLATE.read_text(encoding="utf-8")
    if (tokenizer.chat_template or "").strip() != stock.strip():
        sys.exit("[X] the base tokenizer's chat template differs from the vendored one that matched Ollama")
    template = patch_template(stock)
    tools = json.loads(args.tools.read_text(encoding="utf-8"))
    train_rows = load_rows(args.pairs, tokenizer, tools=tools, chat_template=template)
    val_rows = load_rows(args.val_pairs, tokenizer, tools=tools, chat_template=template) if args.val_pairs.exists() else []
    stop_ids = {tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS}
    problems = preflight(train_rows + val_rows, bos_id=tokenizer.bos_token_id, stop_ids=stop_ids,
                         max_length=MAX_LENGTH)
    lengths = [len(r["prompt_ids"]) + max(len(r["chosen_ids"]), len(r["rejected_ids"])) for r in train_rows + val_rows]
    print(f"[*] pairs train={len(train_rows)} val={len(val_rows)} tools={len(tools)}; longest row {max(lengths)} "
          f"tokens (max_length {MAX_LENGTH})")
    if problems:
        sys.exit("[X] pre-flight failed:\n  " + "\n  ".join(problems[:10]))
    print("[OK] pre-flight passed")
    if args.check_only:
        return
    _train(args, tokenizer, train_rows, val_rows)


def _train(args, tokenizer, train_rows: list[dict], val_rows: list[dict]) -> None:  # pragma: no cover -- pod only
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    from trl import DPOConfig, DPOTrainer

    class PreTokenizedDPOTrainer(DPOTrainer):
        """Rows already hold prompt/chosen/rejected ids (see the module docstring)."""
        def _prepare_dataset(self, dataset, processing_class, dpo_args, dataset_name):
            return dataset

    model = AutoModelForImageTextToText.from_pretrained(args.base, revision=args.revision, dtype=torch.bfloat16,
                                                        device_map="cuda", attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()  # gradient checkpointing under a frozen base
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
                                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    adapted = sorted({name.split(".lora_A")[0] for name, _ in model.named_parameters() if ".lora_A" in name})
    if not adapted or any("tower" in name for name in adapted):
        sys.exit(f"[X] unexpected LoRA targets: {len(adapted)} modules, e.g. {adapted[:3]}")
    print(f"[*] {type(model.base_model.model).__name__}: LoRA on {len(adapted)} language-model modules")
    model.print_trainable_parameters()

    smoke = args.smoke_steps > 0
    cfg = DPOConfig(output_dir=str(args.out / "checkpoints"), beta=args.beta, learning_rate=args.lr,
                    num_train_epochs=args.epochs, max_steps=args.smoke_steps if smoke else -1,
                    per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                    max_length=MAX_LENGTH, truncation_mode="keep_start", warmup_steps=0.1,  # float = ratio (tf 5)
                    lr_scheduler_type="linear", logging_steps=1 if smoke else 5, bf16=True, seed=SEED,
                    report_to="none", save_strategy="no", eval_strategy="no" if smoke or not val_rows else "epoch",
                    gradient_checkpointing=True)
    trainer = PreTokenizedDPOTrainer(model=model, ref_model=None, args=cfg, processing_class=tokenizer,
                                     train_dataset=Dataset.from_list(train_rows),
                                     eval_dataset=Dataset.from_list(val_rows) if val_rows else None)
    result = trainer.train()
    peak = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[*] final loss {result.training_loss:.4f}; peak VRAM {peak:.1f} GB")
    if smoke:
        seconds = result.metrics.get("train_runtime", 0.0) / args.smoke_steps
        full_steps = -(-len(train_rows) // cfg.gradient_accumulation_steps) * args.epochs
        print(f"[OK] smoke run finished; nothing saved. {seconds:.0f} s/step -> full run about "
              f"{seconds * full_steps / 60:.0f} min for {full_steps:.0f} steps")
        return
    metrics = trainer.evaluate() if val_rows else {}
    print(json.dumps(metrics, indent=2))
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    _write_manifest(args, result.training_loss, peak, metrics, len(train_rows), len(val_rows))
    print(f"[OK] DPO adapter -> {args.out}")


def _write_manifest(args, loss: float, peak: float, metrics: dict, n_train: int, n_val: int) -> None:  # pragma: no cover
    import peft
    import torch
    import transformers
    import trl
    manifest = {
        "base": args.base, "revision": args.revision, "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "max_length": MAX_LENGTH, "seed": SEED,
        "train_pairs": n_train, "val_pairs": n_val, "pairs_sha256": _sha256(args.pairs),
        "val_pairs_sha256": _sha256(args.val_pairs) if n_val else None, "tools_sha256": _sha256(args.tools),
        "template_sha256": _sha256(TEMPLATE), "final_loss": loss, "peak_vram_gb": round(peak, 1), "eval": metrics,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
                     "peft": peft.__version__},
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
