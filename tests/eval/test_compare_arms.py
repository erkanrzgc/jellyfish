"""Paired arm comparison: shared cases only, per-arm scores and pairwise flips."""
from __future__ import annotations

import json

from eval.compare_arms import compare, load_arm

DIG = {"expected_tools": ["dig"], "expected_params": {"domain": "app.local"}, "allow_tools": True}
TALK = {"expected_tools": [], "expected_params": {}, "allow_tools": False}


def _call(name: str, params: dict) -> str:
    return "Bakıyorum.\n```arac\n" + json.dumps({"arac": name, "parametreler": params}) + "\n```"


GOOD = _call("dig", {"domain": "app.local"})
WRONG = _call("sqlmap", {"url": "http://app.local"})


def test_compare_counts_scores_and_flips_on_shared_cases():
    cases = {"a": DIG, "b": DIG, "c": TALK, "d": DIG}
    arms = {"base": {"a": GOOD, "b": WRONG, "c": "Bir açıklama.", "d": GOOD},
            "new": {"a": WRONG, "b": GOOD, "c": "Bir açıklama."}}  # "d" unanswered: compared on a, b, c only
    report = compare(cases, arms)
    assert report["cases"] == 3
    assert report["arms"]["base"]["passed"] == 2 and report["arms"]["new"]["passed"] == 2
    assert report["arms"]["new"]["tool_cases"] == 2 and report["arms"]["new"]["no_tool_passed"] == 1
    pair = report["pairs"][0]
    assert (pair["from"], pair["to"], pair["gained"], pair["lost"], pair["net"]) == ("base", "new", 1, 1, 0)
    assert pair["gained_ids"] == ["b"] and pair["lost_ids"] == ["a"] and pair["identical_replies"] == 1


def test_load_arm_reads_tagged_files_and_native_messages(tmp_path):
    message = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "dig", "arguments": {"domain": "app.local"}}}]}
    (tmp_path / "gemma4_e4b_tool_val.jsonl").write_text(json.dumps({"id": "a", "message": message}) + "\n",
                                                        encoding="utf-8")
    replies = load_arm("gemma4_e4b", tag="tool", root=tmp_path)
    assert list(replies) == ["a"] and '"arac": "dig"' in replies["a"].replace("'", '"')
    assert compare({"a": DIG}, {"x": replies})["arms"]["x"]["passed"] == 1
