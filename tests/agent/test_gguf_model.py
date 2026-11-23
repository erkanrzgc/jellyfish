import json
import urllib.error
import urllib.request

from agent.backends.gguf_model import GgufModel, render_prompt
from agent.messages import Message


class _FakeTok:
    """apply_chat_template'i taklit eder; gordugu argumanlari kaydeder."""
    def __init__(self) -> None:
        self.seen: dict | None = None

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw) -> str:
        self.seen = {"msgs": msgs, "tokenize": tokenize, "agp": add_generation_prompt, "kw": kw}
        return "<bos><start_of_turn>user\nU<end_of_turn>\n<start_of_turn>model\n"


def test_render_prompt_strips_bos_and_passes_flags():
    tok = _FakeTok()
    out = render_prompt(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], tok)
    assert not out.startswith("<bos>")                 # cift-BOS tuzagi: bastaki BOS siyrilir
    assert tok.seen["agp"] is True                     # add_generation_prompt=True
    assert tok.seen["tokenize"] is False
    assert tok.seen["msgs"][0]["role"] == "system"


def test_render_prompt_falls_back_when_enable_thinking_unsupported():
    class _StrictTok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "no-bos-here"                        # enable_thinking kwarg'i KABUL ETMEZ
    out = render_prompt([{"role": "user", "content": "U"}], _StrictTok())
    assert out == "no-bos-here"


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen_ok(capture: dict, payload: dict):
    def _fake(req, timeout=None):
        capture["url"] = req.full_url
        capture["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp(payload)
    return _fake


def _urlopen_raises(exc):
    def _fake(req, timeout=None):
        raise exc
    return _fake


def test_call_builds_raw_request_and_returns_response(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr(urllib.request, "urlopen",
                        _urlopen_ok(cap, {"response": "Tararim ```arac```"}))
    m = GgufModel(renderer=lambda d: "PROMPT")
    out = m([Message("user", "10.10.10.5 tara")])
    assert out == "Tararim ```arac```"
    assert cap["url"].endswith("/api/generate")
    assert cap["body"]["raw"] is True
    assert cap["body"]["prompt"] == "PROMPT"
    assert cap["body"]["options"]["stop"] == ["<end_of_turn>"]
    assert cap["body"]["options"]["temperature"] == 0.6


def test_call_injects_system_and_flattens_tool(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_ok({}, {"response": "ok"}))
    m = GgufModel(renderer=lambda d: seen.setdefault("dicts", d) or "P")
    m([Message("user", "U"), Message("tool", "nmap ciktisi")])
    roles = [d["role"] for d in seen["dicts"]]
    assert roles[0] == "system"                          # persona basta
    assert "ARAÇ ÇIKTISI:" in seen["dicts"][-1]["content"]  # tool -> user flatten


def test_default_system_prompt_describes_arac_format(monkeypatch):
    """Teşhis (2026-07-09): persona-only prompt -> model arac blogu basmaz (0/3).
    Varsayilan system prompt arac formatini TARIF ETMELI (araç-farkinda, 3/3)."""
    seen: dict = {}
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_ok({}, {"response": "ok"}))
    m = GgufModel(renderer=lambda d: seen.setdefault("dicts", d) or "P")
    m([Message("user", "10.10.10.5 tara")])
    system = seen["dicts"][0]["content"]
    assert "```arac" in system                           # arac format spec system'de
    assert "parametreler" in system
    assert "Octópus" in system                            # persona guardrail da KORUNUR


def test_connection_refused_returns_error_string(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        _urlopen_raises(urllib.error.URLError("refused")))
    out = GgufModel(renderer=lambda d: "P")([Message("user", "U")])
    assert out.startswith("HATA") and "Ollama" in out


def test_timeout_returns_error_string(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raises(TimeoutError()))
    out = GgufModel(renderer=lambda d: "P")([Message("user", "U")])
    assert out.startswith("HATA") and "zaman" in out.lower()


def test_model_not_found_returns_error_string(monkeypatch):
    err = urllib.error.HTTPError("u", 404, "not found", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raises(err))
    out = GgufModel(renderer=lambda d: "P")([Message("user", "U")])
    assert "ollama create" in out


def test_default_request_uses_promoted_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_ok(captured, {"response": "ok"}))
    GgufModel(renderer=lambda messages: "prompt")([Message("user", "U")])
    assert captured["body"]["model"] == "jellyfish-v10"


def test_default_renderer_loads_evaluated_tokenizer(monkeypatch):
    import agent.backends.gguf_model as gm
    seen = []
    monkeypatch.setattr(gm, "load_tokenizer", lambda path: seen.append(path) or _FakeTok())
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_ok({}, {"response": "ok"}))
    GgufModel()([Message("user", "U")])
    assert seen == ["models/jellyfish-v8-tokenizer"]


def test_missing_default_model_points_to_matching_modelfile(monkeypatch):
    err = urllib.error.HTTPError("u", 404, "not found", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raises(err))
    out = GgufModel(renderer=lambda messages: "P")([Message("user", "U")])
    assert "ollama create jellyfish-v10 -f models/Modelfile.v10" in out


def test_structured_result_preserves_raw_text_counters_and_truncation(monkeypatch):
    captured = {}
    payload = {"response": "  partial\n", "done": True, "done_reason": "length",
               "eval_count": 1024, "prompt_eval_count": 70, "eval_duration": 12345}
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_ok(captured, payload))
    model = GgufModel(renderer=lambda messages: "P", num_predict=1024, seed=41, num_ctx=4096)
    result = model.generate_result([Message("user", "U")])
    assert result.text == "  partial\n" and result.error is None
    assert result.truncated and result.done_reason == "length"
    assert result.metadata["response"] == {k: v for k, v in payload.items() if k != "response"}
    assert result.metadata["elapsed_seconds"] >= 0
    assert result.metadata["settings"]["seed"] == 41
    assert captured["body"]["options"]["num_ctx"] == 4096
    shown = model([Message("user", "U")])
    assert shown.startswith("HATA:") and shown.endswith("partial")


def test_qwen_native_roles_and_no_bos_stripping(monkeypatch):
    from agent.profiles import create_model
    import agent.backends.gguf_model as gm
    seen = {}
    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            return "<bos>QWEN"
    monkeypatch.setattr(gm, "load_tokenizer", lambda path: Tok())
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_ok(seen, {"response": "ok"}))
    create_model("qwen3-4b-instruct-2507")([Message("user", "U"), Message("tool", "output")])
    assert seen["messages"][-1] == {"role": "tool", "content": "output"}
    assert seen["body"]["prompt"] == "<bos>QWEN"
    assert seen["body"]["options"]["stop"] == ["<|im_end|>"]


def test_structured_errors_do_not_become_model_text(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raises(urllib.error.URLError("refused")))
    result = GgufModel(renderer=lambda messages: "P").generate_result([Message("user", "U")])
    assert result.error and result.text == "" and not result.truncated


def test_invalid_server_json_becomes_structured_error(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_ok({}, {"error": "model crashed"}))
    result = GgufModel(renderer=lambda messages: "P").generate_result([Message("user", "U")])
    assert result.error and "model crashed" in result.error


def test_tokenizer_load_is_local_only(monkeypatch):
    import sys
    from types import SimpleNamespace
    from agent.backends.gguf_model import load_tokenizer
    seen = {}
    def load(path, **kwargs):
        seen.update(kwargs)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    load_tokenizer("models/missing")
    assert seen["local_files_only"] is True


def test_error_payload_retains_partial_reply_and_request_fingerprint(monkeypatch):
    import hashlib
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_ok({}, {"response": "partial", "error": "failed"}))
    result = GgufModel(renderer=lambda messages: "PROMPT").generate_result([Message("user", "U")])
    assert result.text == "partial" and result.error
    assert result.metadata["prompt_sha256"] == hashlib.sha256(b"PROMPT").hexdigest()
    assert result.metadata["model"] == "jellyfish-v10"
