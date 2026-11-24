"""gemma4 DPO rendering: prompts as Ollama renders them, completions in the order gemma4 generates them."""
import re
from pathlib import Path

import pytest
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from train.dpo_gemma4 import (CALL_SEPARATOR, TARGET_MODULES, TEMPLATE, patch_template, preflight, render_pair,
                              tokenize_row)

STOCK = TEMPLATE.read_text(encoding="utf-8")
PATCHED = patch_template(STOCK)
TOOLS = [{"type": "function", "function": {
    "name": "dig", "description": "DNS sorgusu",
    "parameters": {"type": "object", "required": ["domain"],
                   "properties": {"domain": {"type": "string", "description": "alan adı"},
                                  "tip": {"type": "string", "description": "kayıt tipi"}}}}}]
PROMPT = [{"role": "system", "content": "Sen Jellyfish'sin."}, {"role": "user", "content": "app.local A kaydı?"}]
DIG = {"type": "function", "function": {"name": "dig", "arguments": {"tip": "A", "domain": "app.local"}}}
CALL_TEXT = '<|tool_call>call:dig{domain:<|"|>app.local<|"|>,tip:<|"|>A<|"|>}<tool_call|>'


class _JinjaTokenizer:
    """transformers' chat-template environment without a tokenizer file, so rendering is tested in CI."""
    bos_token = "<bos>"

    def apply_chat_template(self, messages, tools=None, chat_template=None, tokenize=False,
                            add_generation_prompt=False, **_):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])

        def raise_exception(message):
            raise ValueError(message)
        env.globals["raise_exception"] = raise_exception
        return env.from_string(chat_template).render(messages=messages, tools=tools, bos_token=self.bos_token,
                                                     add_generation_prompt=add_generation_prompt)


def _record(chosen: dict, rejected: dict) -> dict:
    return {"prompt": PROMPT, "chosen": [chosen], "rejected": [rejected], "meta": {"id": "x"}}


def _render(record: dict, template: str = PATCHED) -> dict:
    return render_pair(record, _JinjaTokenizer(), tools=TOOLS, chat_template=template)


def test_patched_template_renders_the_prompt_unchanged():
    tok = _JinjaTokenizer()
    stock = tok.apply_chat_template(PROMPT, tools=TOOLS, chat_template=STOCK, add_generation_prompt=True)
    patched = tok.apply_chat_template(PROMPT, tools=TOOLS, chat_template=PATCHED, add_generation_prompt=True)
    assert patched == stock
    assert stock.startswith("<bos><|turn>system\nSen Jellyfish'sin.<|tool>declaration:dig{")
    assert stock.endswith("<turn|>\n<|turn>user\napp.local A kaydı?<turn|>\n<|turn>model\n")


def test_stock_template_writes_calls_before_prose_which_gemma4_does_not_generate():
    message = {"role": "assistant", "content": "Sorguluyorum.", "tool_calls": [DIG]}
    row = _render(_record(message, message), template=STOCK)
    assert row["chosen"] == CALL_TEXT + "Sorguluyorum.<|tool_response>"


def test_completion_is_prose_blank_line_call_then_the_tool_response_stop():
    chosen = {"role": "assistant", "content": "app.local A kaydını sorguluyorum.", "tool_calls": [DIG]}
    rejected = {"role": "assistant", "content": "dig app.local komutunu çalıştır."}
    row = _render(_record(chosen, rejected))
    assert CALL_SEPARATOR == "\n\n"
    assert row["chosen"] == "app.local A kaydını sorguluyorum.\n\n" + CALL_TEXT + "<|tool_response>"
    assert row["rejected"] == "dig app.local komutunu çalıştır.<turn|>"


def test_call_without_prose_has_no_separator():
    row = _render(_record({"role": "assistant", "content": "", "tool_calls": [DIG]},
                          {"role": "assistant", "content": "Bilmiyorum."}))
    assert row["chosen"] == CALL_TEXT + "<|tool_response>"


def test_patch_fails_loudly_when_the_template_changes():
    with pytest.raises(ValueError, match="anchor"):
        patch_template("{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}")


class _CharTokenizer:
    """One id per character, so prefix behavior is easy to reason about."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_tokenize_row_splits_completion_ids_after_the_prompt():
    ids = tokenize_row({"prompt": "ab", "chosen": "cd", "rejected": "e"}, _CharTokenizer())
    assert ids == {"prompt_ids": [97, 98], "chosen_ids": [99, 100], "rejected_ids": [101]}


def test_tokenize_row_refuses_when_appending_changes_the_prompt_tokens():
    class _Merging(_CharTokenizer):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text.replace("bc", "X")]}
    with pytest.raises(ValueError, match="prompt"):
        tokenize_row({"prompt": "ab", "chosen": "cd", "rejected": "e"}, _Merging())


def test_preflight_names_bos_stop_and_length_problems():
    good = {"prompt_ids": [2, 5, 6], "chosen_ids": [7, 106], "rejected_ids": [8, 50]}
    assert preflight([good], bos_id=2, stop_ids={106, 50}, max_length=10) == []
    bad = {"prompt_ids": [2, 2, 5], "chosen_ids": [7, 1], "rejected_ids": [8] * 9 + [50]}
    problems = preflight([bad], bos_id=2, stop_ids={106, 50}, max_length=10)
    assert len(problems) == 3
    assert any("BOS" in p for p in problems)
    assert any("chosen" in p and "stop" in p for p in problems)
    assert any("rejected" in p and "10" in p for p in problems)


@pytest.mark.parametrize("name, hit", [
    ("model.language_model.layers.0.self_attn.q_proj", True),
    ("model.language_model.layers.41.mlp.down_proj", True),
    ("model.layers.3.self_attn.o_proj", True),
    ("model.vision_tower.encoder.layers.0.self_attn.q_proj", False),
    ("model.vision_tower.encoder.layers.0.mlp.up_proj.linear", False),
    ("model.audio_tower.layers.0.self_attn.k_proj", False),
    ("model.language_model.layers.0.per_layer_projection", False),
])
def test_lora_targets_only_the_language_model(name, hit):
    assert bool(re.fullmatch(TARGET_MODULES, name)) is hit


def _real_tokenizer():
    directory = Path("models/gemma-4-E4B-it-tokenizer")
    if not directory.is_dir():
        pytest.skip("local gemma4 tokenizer not present")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(directory))


def test_ci_renderer_matches_transformers_and_completions_end_on_a_stop_token():
    tokenizer = _real_tokenizer()
    chosen = {"role": "assistant", "content": "Sorguluyorum.", "tool_calls": [DIG]}
    rejected = {"role": "assistant", "content": "Bilmiyorum."}
    record = _record(chosen, rejected)
    real = render_pair(record, tokenizer, tools=TOOLS, chat_template=PATCHED)
    assert real == _render(record)
    ids = tokenize_row(real, tokenizer)
    stop = {tokenizer.convert_tokens_to_ids("<turn|>"), tokenizer.convert_tokens_to_ids("<|tool_response>")}
    assert preflight([ids], bos_id=tokenizer.bos_token_id, stop_ids=stop, max_length=8704) == []
    assert ids["chosen_ids"][-1] == tokenizer.convert_tokens_to_ids("<|tool_response>")
    assert ids["rejected_ids"][-1] == tokenizer.convert_tokens_to_ids("<turn|>")
