import json
import urllib.request

from agent.backends.gguf_model import GgufModel
from agent.catalog import CATALOG
from agent.constrained import call_schema, repair_prefix, splice_repaired_call
from agent.messages import Message
from agent.toolcall import parse_tool_reply


def test_schema_lists_every_catalog_tool_with_only_its_own_keys():
    schema = call_schema()
    branches = {b["properties"]["arac"]["const"]: b for b in schema["anyOf"]}
    assert set(branches) == set(CATALOG)
    dig = branches["dig"]["properties"]["parametreler"]
    assert set(dig["properties"]) == {"domain", "tip"}
    assert dig["additionalProperties"] is False


def test_schema_requires_the_single_target_key_and_assistant_required_keys():
    branches = {b["properties"]["arac"]["const"]: b for b in call_schema()["anyOf"]}
    assert "domain" in branches["dig"]["properties"]["parametreler"]["required"]
    assert branches["read_file"]["properties"]["parametreler"]["required"] == ["yol"]
    assert "required" not in branches["netdiscover"]["properties"]["parametreler"]


def test_schema_can_be_limited_to_a_tool_subset():
    names = {b["properties"]["arac"]["const"] for b in call_schema(["nmap", "dig", "uydurma"])["anyOf"]}
    assert names == {"nmap", "dig"}


def test_valid_reply_needs_no_repair():
    text = '```arac\n{"arac":"dig","parametreler":{"domain":"a.local"}}\n```'
    assert repair_prefix(text) is None


def test_prose_reply_needs_no_repair():
    assert repair_prefix("Bu bir kavram sorusu, araca gerek yok.") is None


def test_bad_param_key_is_cut_at_the_block_opener():
    text = 'Sorguluyorum.\n```arac\n{"arac":"dig","parametreler":{"alan":"a.local"}}\n```\nSonra bakarim.'
    assert repair_prefix(text) == "Sorguluyorum.\n```arac\n"


def test_unclosed_block_and_bad_json_with_a_catalog_name_are_repaired():
    assert repair_prefix('Bak:\n```arac\n{"arac":"nmap"') == "Bak:\n```arac\n"
    bad_json = '```arac\n{"arac": "grep", "parametreler": {"desen": "a" "yol": "b"}}\n```'
    assert repair_prefix(bad_json) == "```arac\n"


def test_invented_tool_is_not_repaired_by_default():
    # A refusal prompt answered with an invented tool must not become a runnable catalog call.
    text = 'Yardimci olamam.\n```arac\n{"arac":"social_engineering","parametreler":{}}\n```'
    assert repair_prefix(text) is None
    assert repair_prefix("```arac\n{oops") is None


def test_invented_tool_is_repaired_only_when_explicitly_allowed():
    text = '```arac\n{"arac":"dns-dig","parametreler":{}}\n```'
    assert repair_prefix(text, allow_unknown_names=True) == "```arac\n"


def test_valid_first_block_is_kept_and_only_the_bad_one_is_cut():
    good = '```arac\n{"arac":"nmap","parametreler":{"hedef":"10.0.0.1"}}\n```\n'
    text = good + '```arac\n{"arac":"dig","parametreler":{"alan":"a"}}\n```'
    assert repair_prefix(text) == good + "```arac\n"


def test_splice_closes_the_block():
    out = splice_repaired_call("X\n```arac\n", '{"arac":"dig","parametreler":{"domain":"a"}}\n')
    assert out == 'X\n```arac\n{"arac":"dig","parametreler":{"domain":"a"}}\n```'
    assert parse_tool_reply(out, validate_catalog=True).errors == []


def _scripted_urlopen(replies, bodies):
    def _fake(req, timeout=None):
        bodies.append(json.loads(req.data.decode("utf-8")))
        payload = json.dumps({"response": replies[len(bodies) - 1], "done_reason": "stop"}).encode()

        class _R:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return _R()
    return _fake


def _model(**kw):
    return GgufModel(renderer=lambda dicts: "PROMPT", **kw)


def test_constrained_model_repairs_invalid_call_with_format(monkeypatch):
    bodies = []
    monkeypatch.setattr(urllib.request, "urlopen", _scripted_urlopen([
        'Sorgu.\n```arac\n{"arac":"dig","parametreler":{"alan":"a.local"}}\n```',
        '{"arac":"dig","parametreler":{"domain":"a.local"}}',
    ], bodies))
    result = _model(constrain_calls=True).generate_result([Message(role="user", content="u")])
    assert len(bodies) == 2
    assert "format" not in bodies[0]
    assert bodies[1]["prompt"] == "PROMPT" + "Sorgu.\n```arac\n"
    assert bodies[1]["format"]["anyOf"]
    calls = parse_tool_reply(result.text, validate_catalog=True)
    assert calls.errors == [] and calls.calls[0].params == {"domain": "a.local"}
    assert result.metadata["call_repair"]["applied"] is True


def test_constrained_model_leaves_valid_replies_alone(monkeypatch):
    bodies = []
    monkeypatch.setattr(urllib.request, "urlopen", _scripted_urlopen(
        ['```arac\n{"arac":"dig","parametreler":{"domain":"a"}}\n```'], bodies))
    result = _model(constrain_calls=True).generate_result([Message(role="user", content="u")])
    assert len(bodies) == 1 and "call_repair" not in result.metadata


def test_flag_off_keeps_the_invalid_reply(monkeypatch):
    bodies = []
    bad = '```arac\n{"arac":"dig","parametreler":{"alan":"a"}}\n```'
    monkeypatch.setattr(urllib.request, "urlopen", _scripted_urlopen([bad], bodies))
    result = _model().generate_result([Message(role="user", content="u")])
    assert len(bodies) == 1 and result.text == bad


def test_failed_repair_keeps_the_original_reply(monkeypatch):
    bodies = []
    bad = '```arac\n{"arac":"dig","parametreler":{"alan":"a"}}\n```'
    monkeypatch.setattr(urllib.request, "urlopen", _scripted_urlopen([bad, "not json"], bodies))
    result = _model(constrain_calls=True).generate_result([Message(role="user", content="u")])
    assert result.text == bad
    assert result.metadata["call_repair"]["applied"] is False


def test_model_does_not_repair_invented_names_by_default(monkeypatch):
    bodies = []
    bad = '```arac\n{"arac":"social_engineering","parametreler":{}}\n```'
    monkeypatch.setattr(urllib.request, "urlopen", _scripted_urlopen([bad], bodies))
    result = _model(constrain_calls=True).generate_result([Message(role="user", content="u")])
    assert len(bodies) == 1 and result.text == bad
