from agent.audit import AuditLog
from agent.executor import MockExecutor
from agent.loop import run_tool_loop
from agent.messages import Message
from agent.policy import LabPolicy
from agent.registry import ToolRegistry


CALL = '```arac\n{"arac":"nmap","parametreler":{"hedef":"10.10.10.5"}}\n```'


def registry(tmp_path, executor=None):
    return ToolRegistry(LabPolicy(scope=["10.10.10.0/24"]),
                        executor or MockExecutor(), AuditLog(tmp_path / "audit.jsonl"))


def test_invalid_call_gets_one_repair_then_stops(tmp_path):
    attempts = []
    def generate(messages):
        attempts.append(1)
        return '```arac\n{"arac":"nmap","parametreler":5}\n```'
    result = run_tool_loop([Message("user", "tara")], generate, registry(tmp_path))
    assert len(attempts) == 2
    assert result.status == "failed" and result.calls == []
    assert "tamamlanmadi" in result.final.lower()


def test_unknown_tool_can_be_repaired_without_execution(tmp_path):
    replies = iter(['```arac\n{"arac":"whois_lookup","parametreler":{}}\n```', CALL, "Bulgular: mock çıktı."])
    result = run_tool_loop([Message("user", "tara")], lambda m: next(replies), registry(tmp_path))
    assert [c.name for c in result.calls] == ["nmap"]
    assert result.status == "completed"


def test_denial_does_not_allow_fabricated_completion(tmp_path):
    count = []
    def generate(messages):
        count.append(1)
        return CALL.replace("10.10.10.5", "192.0.2.1") if len(count) == 1 else "Tarama tamamlandı!"
    result = run_tool_loop([Message("user", "tara")], generate, registry(tmp_path))
    assert len(count) == 1 and result.calls == []
    assert result.status == "denied" and result.attempts[0].status == "denied"


def test_backend_failure_stops_without_completed_claim(tmp_path):
    class Failed(MockExecutor):
        def run(self, tool, params): return "HATA: unavailable"
    result = run_tool_loop([Message("user", "tara")], lambda m: CALL, registry(tmp_path, Failed()))
    assert result.status == "failed" and result.calls == []


def test_step_limit_is_incomplete_and_does_not_generate_extra_reply(tmp_path):
    seen = []
    def generate(messages):
        seen.append(1)
        return CALL
    result = run_tool_loop([Message("user", "tara")], generate, registry(tmp_path), max_steps=2)
    assert len(seen) == 2 and result.status == "incomplete"
    assert "tamamlanmadi" in result.final.lower()
