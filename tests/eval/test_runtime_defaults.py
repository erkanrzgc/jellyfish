"""Active evaluation entry points must use the same runtime as the demos."""
import sys

import pytest

from eval import run_subagent_eval, run_toolcall_eval


@pytest.mark.parametrize("runner", [run_toolcall_eval, run_subagent_eval])
def test_eval_defaults_match_promoted_runtime(monkeypatch, tmp_path, runner):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["eval"])
    monkeypatch.setattr(runner, "REPORTS", tmp_path)
    monkeypatch.setattr(runner, "load_cases", lambda path: [])
    if runner is run_toolcall_eval:
        def make_generate(model, tokenizer_dir):
            captured.update(model=model, tokenizer_dir=tokenizer_dir)
            return lambda prompt: ""
        monkeypatch.setattr(runner, "_make_generate", make_generate)
    else:
        def model(**kwargs):
            captured.update(kwargs)
            return lambda messages: ""
        monkeypatch.setattr(runner, "GgufModel", model)
    runner.main()
    assert captured == {
        "model": "jellyfish-v10", "tokenizer_dir": "models/jellyfish-v8-tokenizer",
    }
