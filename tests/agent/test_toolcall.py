from agent.toolcall import ToolCall, parse_arac_calls, render_for_model
from agent.messages import Message
import pytest


def test_parse_single_call():
    txt = 'Tararim.\n```arac\n{"arac":"nmap","parametreler":{"hedef":"1.2.3.4","secenekler":"-sV"}}\n```'
    calls = parse_arac_calls(txt)
    assert len(calls) == 1
    assert calls[0].name == "nmap"
    assert calls[0].params == {"hedef": "1.2.3.4", "secenekler": "-sV"}


def test_malformed_skipped():
    txt = "```arac\n{bozuk json}\n```\n```arac\n{\"arac\":\"whois\",\"parametreler\":{}}\n```"
    calls = parse_arac_calls(txt)
    assert [c.name for c in calls] == ["whois"]


def test_no_calls_returns_empty():
    assert parse_arac_calls("Sadece Turkce cevap, arac yok.") == []


@pytest.mark.parametrize("value", [5, True, "bad", ["bad"], [["hedef", "10.0.0.1"]]])
def test_invalid_parameter_types_never_crash_or_coerce(value):
    import json
    text = '```arac\n' + json.dumps({"arac": "nmap", "parametreler": value}) + '\n```'
    assert parse_arac_calls(text) == []


@pytest.mark.parametrize("name", [None, 7, [], {}, "", "   "])
def test_invalid_tool_name_is_not_coerced_to_text(name):
    import json
    text = '```arac\n' + json.dumps({"arac": name, "parametreler": {}}) + '\n```'
    assert parse_arac_calls(text) == []


def test_render_flattens_tool_role():
    msgs = [Message("user", "tara"), Message("tool", "22/ssh 80/http")]
    out = render_for_model(msgs)
    # tool rolu -> user, "ARAC CIKTISI" oneki (Gemma-2 tool desteklemez)
    assert out[-1]["role"] == "user"
    assert "ARAÇ ÇIKTISI" in out[-1]["content"]
    assert "22/ssh 80/http" in out[-1]["content"]
