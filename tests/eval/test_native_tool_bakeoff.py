from agent.toolcall import parse_tool_reply
from eval.native_tool_bakeoff import native_tools, select_sample, to_arac_reply


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


def test_sample_is_stratified_deterministic_and_bounded():
    cases = [{"id": f"{cat}-{tools}-{i}", "category": cat, "allow_tools": tools}
             for cat in ("tooling", "authorization") for tools in (True, False) for i in range(50)]
    first = select_sample(cases, 40, seed=7)
    assert first == select_sample(cases, 40, seed=7)
    assert len(first) == 40
    strata = {(c["category"], c["allow_tools"]) for c in cases if c["id"] in set(first)}
    assert len(strata) == 4
