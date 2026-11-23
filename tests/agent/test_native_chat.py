import json
import urllib.request

from agent.backends.native_chat import NativeChatModel, to_chat_messages
from agent.loop import run_tool_loop
from agent.messages import Message
from agent.audit import AuditLog
from agent.executor import MockExecutor
from agent.policy import LabPolicy
from agent.registry import ToolRegistry
from agent.toolcall import parse_tool_reply

CALL = '```arac\n{"arac": "nmap", "parametreler": {"hedef": "10.10.10.5"}}\n```'


def test_history_turns_arac_text_back_into_native_tool_calls():
    chat = to_chat_messages([Message("user", "tara"), Message("assistant", "Tariyorum.\n" + CALL),
                             Message("tool", "22/tcp open ssh")], "SYS")
    assert chat[0] == {"role": "system", "content": "SYS"}
    assert chat[2]["content"] == "Tariyorum."
    assert chat[2]["tool_calls"] == [{"function": {"name": "nmap", "arguments": {"hedef": "10.10.10.5"}}}]
    assert chat[3] == {"role": "tool", "content": "22/tcp open ssh", "tool_name": "nmap"}


def test_plain_assistant_turn_has_no_tool_calls():
    chat = to_chat_messages([Message("user", "u"), Message("assistant", "Kavramsal cevap.")], "SYS")
    assert "tool_calls" not in chat[2]


def _urlopen(replies, bodies):
    def _fake(req, timeout=None):
        bodies.append(json.loads(req.data.decode("utf-8")))
        payload = json.dumps(replies[len(bodies) - 1]).encode()

        class _R:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return _R()
    return _fake


def test_request_uses_chat_api_with_catalog_tools_and_thinking_off(monkeypatch):
    bodies = []
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen(
        [{"message": {"content": "", "tool_calls": [
            {"function": {"name": "nmap", "arguments": {"hedef": "10.10.10.5"}}}]}, "done_reason": "stop"}], bodies))
    reply = NativeChatModel(model="gemma4:e4b")([Message("user", "tara")])
    body = bodies[0]
    assert body["model"] == "gemma4:e4b" and body["think"] is False and body["stream"] is False
    assert len(body["tools"]) == 130 and body["messages"][0]["role"] == "system"
    assert parse_tool_reply(reply, validate_catalog=True).calls[0].name == "nmap"


def test_connection_failure_becomes_an_error_string(monkeypatch):
    def _boom(req, timeout=None):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert NativeChatModel()([Message("user", "u")]).startswith("HATA")


def test_native_model_drives_the_unchanged_tool_loop(monkeypatch, tmp_path):
    bodies = []
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen([
        {"message": {"content": "", "tool_calls": [
            {"function": {"name": "nmap", "arguments": {"hedef": "10.10.10.5"}}}]}, "done_reason": "stop"},
        {"message": {"content": "22 numarali port acik."}, "done_reason": "stop"}], bodies))
    result = run_tool_loop([Message("user", "10.10.10.5 tara")], NativeChatModel(), ToolRegistry(
        LabPolicy(scope=["10.10.10.0/24"]), MockExecutor(), AuditLog(tmp_path / "a.jsonl")))
    assert [call.name for call in result.calls] == ["nmap"]
    assert bodies[1]["messages"][-1]["role"] == "tool"
    assert "22" in result.final


def test_native_persona_is_jellyfish_and_v10_training_prompt_is_untouched():
    from agent.backends.native_chat import NATIVE_SYSTEM_PROMPT
    from agent.persona import JELLYFISH_SYSTEM_PROMPT
    from data.sft.persona import OCTOPUS_SYSTEM_PROMPT
    assert NATIVE_SYSTEM_PROMPT.startswith(JELLYFISH_SYSTEM_PROMPT)
    assert "Jellyfish" in NATIVE_SYSTEM_PROMPT and "Octópus" not in NATIVE_SYSTEM_PROMPT
    assert "yalnızca" in JELLYFISH_SYSTEM_PROMPT          # introduce only when asked
    assert OCTOPUS_SYSTEM_PROMPT.startswith("Sen Octópus'sun")  # v10 was trained on this exact text


def test_advisory_mode_offers_no_tools(monkeypatch):
    bodies = []
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen(
        [{"message": {"content": "Aciklama."}, "done_reason": "stop"}], bodies))
    NativeChatModel(tools_enabled=False)([Message("user", "u")])
    assert "tools" not in bodies[0]
