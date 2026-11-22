from agent.toolcall import parse_tool_reply
from data.dpo.build import split_of
from eval.native_tool_bakeoff import bakeoff_ids, native_tools, to_arac_reply


def test_native_tools_cover_the_catalog_with_own_keys():
    tools = {t["function"]["name"]: t["function"] for t in native_tools()}
    assert len(tools) == 130
    dig = tools["dig"]["parameters"]
    assert set(dig["properties"]) == {"domain", "tip"} and dig["required"] == ["domain"]
    assert tools["read_file"]["parameters"]["required"] == ["yol"]


def test_tool_calls_become_one_arac_block_each_and_scalars_become_text():
    message = {"content": "Bakiyorum.", "tool_calls": [
        {"function": {"name": "nmap", "arguments": {"hedef": "10.0.0.1", "secenekler": 22}}},
        {"function": {"name": "dig", "arguments": '{"domain": "a.local"}'}}]}
    reply = to_arac_reply(message)
    parsed = parse_tool_reply(reply, validate_catalog=True)
    assert parsed.errors == []
    assert [c.name for c in parsed.calls] == ["nmap", "dig"]
    assert parsed.calls[0].params["secenekler"] == "22"
    assert reply.startswith("Bakiyorum.")


def test_plain_answer_has_no_block():
    assert to_arac_reply({"content": "Kavramsal cevap."}) == "Kavramsal cevap."


def test_unparseable_arguments_stay_visible_as_bad_json():
    reply = to_arac_reply({"content": "", "tool_calls": [{"function": {"name": "nmap", "arguments": "{bad"}}]})
    assert parse_tool_reply(reply).errors


def test_bakeoff_uses_exactly_the_val_families():
    cases = [{"id": f"c{i}", "family_id": f"fam-{i // 3}"} for i in range(300)]
    ids = bakeoff_ids(cases)
    assert ids and set(ids) == {c["id"] for c in cases if split_of(c["family_id"]) == "val"}


def test_v10_arms_apply_catalog_repair_only_to_catalog_named_calls():
    from eval.native_tool_bakeoff import _v10_arms
    bad_key = '```arac\n{"arac":"dig","parametreler":{"alan":"a"}}\n```'
    invented = '```arac\n{"arac":"dns-dig","parametreler":{}}\n```'
    fixed = '```arac\n{"arac":"dig","parametreler":{"domain":"a"}}\n```'
    rows = [{"id": "a", "reply": bad_key, "reply_repair_any": fixed},
            {"id": "b", "reply": invented, "reply_repair_any": fixed}]
    arms = _v10_arms(rows)
    assert arms["octopus-v10"] == {"a": bad_key, "b": invented}
    assert arms["octopus-v10+repair"] == {"a": fixed, "b": invented}
    assert arms["octopus-v10+repair-any-name"] == {"a": fixed, "b": fixed}
