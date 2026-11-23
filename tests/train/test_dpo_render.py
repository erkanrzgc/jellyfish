import pytest

from train.dpo_bf16 import bos_problem, render_pair

RECORD = {"prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
          "chosen": [{"role": "assistant", "content": "iyi"}],
          "rejected": [{"role": "assistant", "content": "kotu"}], "meta": {"id": "x"}}


class _Gemma:
    """Gemma-2 chat template stand-in: literal <bos>, turns end with <end_of_turn>\\n."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        text = "<bos>" + "".join(f"<start_of_turn>{m['role'].replace('assistant', 'model')}\n{m['content']}"
                                 "<end_of_turn>\n" for m in messages)
        return text + ("<start_of_turn>model\n" if add_generation_prompt else "")


def test_pair_text_matches_the_sft_rendering():
    row = render_pair(RECORD, _Gemma())
    assert row["prompt"].startswith("<bos><start_of_turn>system\nS")
    assert row["prompt"].endswith("<start_of_turn>model\n")
    assert row["chosen"] == "iyi<end_of_turn>\n"
    assert row["rejected"] == "kotu<end_of_turn>\n"


def test_pair_rejects_a_template_that_does_not_extend_the_prompt():
    class _Broken(_Gemma):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
            return "X" if not add_generation_prompt else "Y"
    with pytest.raises(ValueError):
        render_pair(RECORD, _Broken())


def test_bos_check_requires_exactly_one_leading_bos():
    assert bos_problem([2, 106, 5], bos_id=2) is None
    assert bos_problem([2, 2, 106], bos_id=2)
    assert bos_problem([106, 5], bos_id=2)


def test_real_gemma_tokenizer_gives_one_bos_and_sft_turn_ending():
    from pathlib import Path
    directory = Path("models/jellyfish-v8-tokenizer")
    if not directory.is_dir():
        pytest.skip("local tokenizer not present")
    from agent.backends.gguf_model import load_tokenizer
    tokenizer = load_tokenizer(str(directory))
    row = render_pair(RECORD, tokenizer)
    ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    assert bos_problem(ids, bos_id=tokenizer.bos_token_id) is None
    assert row["chosen"] == "iyi<end_of_turn>\n"
