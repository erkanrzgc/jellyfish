"""On-policy gemma4 preference pairs: gemma4's own failures vs teacher chosen, in native tool-call form."""
from __future__ import annotations

import json

from agent.backends.native_chat import NATIVE_SYSTEM_PROMPT, native_tools, to_arac_reply
from data.dpo.build import split_of
from data.dpo.common import classify_failures
from data.dpo.native_pairs import (build, failed_rows, gap_rows, is_command_quibble, normalize_message,
                                   to_native_message, usable_chosen)

DIG_CHOSEN = ('```dusunce\nA kaydı lazım.\n```\napp.local alan adının A kaydını sorguluyorum.\n\n'
              '```arac\n{"arac": "dig", "parametreler": {"domain": "app.local", "tip": "A"}}\n```')


def _case(cid: str, family: str, *, tools: bool = True) -> dict:
    return {"id": cid, "family_id": family, "category": "tooling", "prompt": f"istem {cid}",
            "expected_tools": ["dig"] if tools else [], "expected_params": {"domain": "app.local"} if tools else {},
            "allow_tools": tools}


def _gemma_call(name: str, arguments) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def test_chosen_text_becomes_a_native_message_without_reasoning_fences():
    message = to_native_message(DIG_CHOSEN)
    assert message["role"] == "assistant"
    assert message["content"] == "app.local alan adının A kaydını sorguluyorum."
    assert message["tool_calls"] == [{"type": "function", "function": {
        "name": "dig", "arguments": {"domain": "app.local", "tip": "A"}}}]


def test_converted_chosen_still_passes_the_structural_gate():
    case = _case("a", "fam-1")
    assert classify_failures(case, to_arac_reply(to_native_message(DIG_CHOSEN))) == []


def test_plain_answer_has_no_tool_calls_key():
    assert to_native_message("Bu bir kavram sorusu.") == {"role": "assistant", "content": "Bu bir kavram sorusu."}


def test_gemma_reply_keeps_invalid_arguments_visible():
    raw = {"role": "assistant", "content": "x", "thinking": "drop me",
           "tool_calls": [{"id": "1", "function": {"name": "nmap", "arguments": "{bad", "index": 0}}]}
    message = normalize_message(raw)
    assert message == {"role": "assistant", "content": "x",
                       "tool_calls": [{"type": "function", "function": {"name": "nmap", "arguments": "{bad"}}]}


def test_failed_and_gap_rows():
    cases = {"a": _case("a", "fam-1"), "b": _case("b", "fam-1"), "c": _case("c", "fam-1")}
    replies = {"a": _gemma_call("dig", {"domain": "app.local"}),  # passes
               "b": _gemma_call("dig", {"alan": "app.local"}),  # wrong key
               "c": {"role": "assistant", "content": "dig app.local komutunu çalıştır."}}  # no call
    failed = failed_rows(cases, replies)
    assert [row["id"] for row in failed] == ["b", "c"]
    assert failed[0]["raw_reply"] == to_arac_reply(replies["b"])
    chosen = {"b": {"id": "b", "chosen": DIG_CHOSEN}, "c": {"id": "c", "chosen": None}}
    assert [row["id"] for row in gap_rows(cases, failed, chosen)] == ["c"]


def test_old_brand_or_broken_chosen_goes_back_to_the_teacher():
    case = _case("a", "fam-1")
    assert usable_chosen(case, DIG_CHOSEN) == to_native_message(DIG_CHOSEN)
    assert usable_chosen(case, None) is None
    assert usable_chosen(case, "Merhaba, ben Octópus. " + DIG_CHOSEN) is None
    assert usable_chosen(case, "ben octopus'um. " + DIG_CHOSEN) is None
    # Seen in a real teacher answer (DPO-TOOL-0337): a one-line reasoning fence that is never closed.
    broken = DIG_CHOSEN.replace("```dusunce\nA kaydı lazım.\n```", "```dusunce```\nA kaydı lazım.")
    assert usable_chosen(case, broken) is None
    failed = [{"id": "a", "raw_reply": "", "failures": ["missing_call"]}]
    assert gap_rows({"a": case}, failed, {"a": {"id": "a", "chosen": broken}}) == failed


def test_build_writes_native_pairs_tools_and_manifest(tmp_path):
    train_family = next(f"fam-{i}" for i in range(100) if split_of(f"fam-{i}") == "train")
    val_family = next(f"fam-{i}" for i in range(100) if split_of(f"fam-{i}") == "val")
    cases = {"t1": _case("t1", train_family), "t2": _case("t2", train_family), "t3": _case("t3", train_family),
             "v1": _case("v1", val_family)}
    replies = {"t1": _gemma_call("dig", {"domain": "app.local"}),  # passes, no pair
               "t2": _gemma_call("dig", {"alan": "app.local"}),
               "t3": _gemma_call("dig", {"alan": "app.local"}),  # no chosen
               "v1": _gemma_call("whois_lookup", {"domain": "app.local"})}
    cases["t4"] = _case("t4", train_family)
    replies["t4"] = _gemma_call("dig", {"alan": "app.local"})
    chosen = {"t2": {"id": "t2", "chosen": DIG_CHOSEN, "teacher_model": "glm"},
              "t4": {"id": "t4", "chosen": "Ben Octópus. " + DIG_CHOSEN, "teacher_model": "glm"},
              "v1": {"id": "v1", "chosen": DIG_CHOSEN, "teacher_model": "glm"}}
    manifest = build(cases, replies, chosen, tmp_path / "out", model="gemma4:e4b")
    out = tmp_path / "out"
    train = [json.loads(l) for l in (out / "pairs_train.jsonl").read_text(encoding="utf-8").splitlines()]
    val = [json.loads(l) for l in (out / "pairs_val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["meta"]["id"] for r in train] == ["t2"] and [r["meta"]["id"] for r in val] == ["v1"]
    record = train[0]
    assert record["prompt"] == [{"role": "system", "content": NATIVE_SYSTEM_PROMPT},
                                {"role": "user", "content": "istem t2"}]
    assert record["chosen"] == [to_native_message(DIG_CHOSEN)]
    assert record["rejected"][0]["tool_calls"][0]["function"]["arguments"] == {"alan": "app.local"}
    assert "param_mismatch" in record["meta"]["failures"]
    tools = json.loads((out / "tools.json").read_text(encoding="utf-8"))
    assert tools == native_tools()
    assert manifest == json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "gemma4:e4b" and manifest["scored"] == 5 and manifest["model_passed"] == 1
    assert manifest["pairs"] == 2 and manifest["train"] == 1 and manifest["val"] == 1
    assert manifest["missing_chosen"] == 1 and manifest["unusable_chosen"] == 1
    assert manifest["tools"] == len(tools)


def _cmd(command: str) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "run_cmd", "arguments": {"komut": command}}}]}


def test_shell_command_string_differences_are_not_pairs():
    assert is_command_quibble(["param_mismatch"], _cmd("free"), _cmd("free -h"))
    assert not is_command_quibble(["param_mismatch", "catalog"], _cmd("free"), _cmd("free -h"))
    assert not is_command_quibble(["wrong_first_tool"], _gemma_call("dig", {}), _cmd("free -h"))
    dig = to_native_message(DIG_CHOSEN)
    assert not is_command_quibble(["param_mismatch"], normalize_message(_gemma_call("dig", {"alan": "x"})), dig)


def test_build_drops_command_quibbles(tmp_path):
    family = next(f"fam-{i}" for i in range(100) if split_of(f"fam-{i}") == "train")
    case = {"id": "c1", "family_id": family, "category": "incident", "prompt": "bellek",
            "expected_tools": ["run_cmd"], "expected_params": {"komut": "free -h"}, "allow_tools": True}
    chosen_text = 'Belleğe bakıyorum.\n\n```arac\n{"arac": "run_cmd", "parametreler": {"komut": "free -h"}}\n```'
    manifest = build({"c1": case}, {"c1": _cmd("free")}, {"c1": {"id": "c1", "chosen": chosen_text}},
                     tmp_path / "out")
    assert manifest["pairs"] == 0 and manifest["dropped_command_quibbles"] == 1


def test_build_caps_missing_call_pairs_deterministically(tmp_path):
    """Most v1 pairs were missing_call and the model mostly learned "call a tool"; the cap rebalances."""
    family = next(f"fam-{i}" for i in range(100) if split_of(f"fam-{i}") == "train")
    cases = {f"c{i}": _case(f"c{i}", family) for i in range(8)}
    replies = {f"c{i}": _gemma_call("dig", {"alan": "app.local"}) for i in range(3)}  # param_mismatch
    replies.update({f"c{i}": {"role": "assistant", "content": "A kaydı şudur."} for i in range(3, 8)})  # missing
    chosen = {cid: {"id": cid, "chosen": DIG_CHOSEN, "teacher_model": "glm"} for cid in cases}
    first = build(cases, replies, chosen, tmp_path / "a", max_missing_share=0.3)
    again = build(cases, replies, chosen, tmp_path / "b", max_missing_share=0.3)
    rows = [json.loads(l) for l in (tmp_path / "a" / "pairs_train.jsonl").read_text(encoding="utf-8").splitlines()]
    missing = [r for r in rows if "missing_call" in r["meta"]["failures"]]
    assert len(rows) == 4 and len(missing) == 1 and first["dropped_missing_call_cap"] == 4
    assert first["by_failure"] == {"catalog": 3, "missing_call": 1, "param_mismatch": 3}
    assert (tmp_path / "a" / "pairs_train.jsonl").read_text() == (tmp_path / "b" / "pairs_train.jsonl").read_text()
    assert again["pairs"] == first["pairs"]
    uncapped = build(cases, replies, chosen, tmp_path / "c")
    assert uncapped["pairs"] == 8 and uncapped["dropped_missing_call_cap"] == 0


def test_tagged_reply_and_chosen_files():
    from data.dpo.native_pairs import EXTRA_CHOSEN, NATIVE_TRAIN_DIR, OUT_DIR, chosen_paths, reply_paths
    assert reply_paths("gemma4:e4b", "") == [NATIVE_TRAIN_DIR / "gemma4_e4b_val.jsonl",
                                             NATIVE_TRAIN_DIR / "gemma4_e4b_train.jsonl"]
    assert reply_paths("gemma4:e4b", "tool")[1] == NATIVE_TRAIN_DIR / "gemma4_e4b_tool_train.jsonl"
    assert chosen_paths(["", "tool"]) == [OUT_DIR / "chosen.jsonl", EXTRA_CHOSEN,
                                          NATIVE_TRAIN_DIR / "chosen_tool.jsonl"]
