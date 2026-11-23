from agent.loop import run_tool_loop, ToolLoopResult
from agent.messages import Message
from agent.registry import ToolRegistry
from agent.policy import LabPolicy
from agent.executor import MockExecutor
from agent.audit import AuditLog


def _reg(tmp_path):
    return ToolRegistry(LabPolicy(scope=["10.10.10.0/24"]), MockExecutor(), AuditLog(tmp_path / "a.jsonl"))


def test_single_tool_then_final(tmp_path):
    # scripted model: 1. tur arac cagirir, 2. tur duz cevap
    scripted = iter([
        '```arac\n{"arac":"nmap","parametreler":{"hedef":"10.10.10.5","secenekler":"-sV"}}\n```',
        "Tarama bitti: 3 acik port var.",
    ])
    def generate(msgs): return next(scripted)
    res = run_tool_loop([Message("user", "tara")], generate, _reg(tmp_path))
    assert isinstance(res, ToolLoopResult)
    assert res.steps == 2 and "3 acik port" in res.final
    assert len(res.calls) == 1


def test_plain_answer_no_tools(tmp_path):
    res = run_tool_loop([Message("user", "selam")], lambda m: "Merhaba, ben Octópus.", _reg(tmp_path))
    assert res.steps == 1 and "Octópus" in res.final and res.calls == []


def test_max_steps_guard(tmp_path):
    # her turda arac cagiran sonsuz model -> max_steps'te durur
    call = '```arac\n{"arac":"nmap","parametreler":{"hedef":"10.10.10.5"}}\n```'
    res = run_tool_loop([Message("user", "x")], lambda m: call, _reg(tmp_path), max_steps=3)
    assert res.steps == 3


INCIDENT = "Sunucuda 'fork: retry: Resource temporarily unavailable' hatasi aliyorum, neden?"
PUSHED_CALL = '```arac\n{"arac":"run_cmd","parametreler":{"komut":"ulimit -u"}}\n```'


def test_evidence_push_is_off_by_default(tmp_path):
    seen = []
    res = run_tool_loop([Message("user", INCIDENT)], lambda m: seen.append(1) or "Muhtemelen limit.", _reg(tmp_path))
    assert len(seen) == 1 and res.final == "Muhtemelen limit."


def test_evidence_push_asks_once_for_a_tool_on_an_incident(tmp_path):
    replies = iter(["Muhtemelen limit.", "Bulgu yok, yine de bakalim.", "Bitti."])
    transcripts = []

    def generate(messages):
        transcripts.append(list(messages))
        return next(replies)
    res = run_tool_loop([Message("user", INCIDENT)], generate, _reg(tmp_path), evidence_push=True)
    assert len(transcripts) == 2                       # one push only, then the answer stands
    assert transcripts[1][-1].role == "user" and "KENDİN" in transcripts[1][-1].content
    assert res.final == "Bulgu yok, yine de bakalim."


def test_evidence_push_never_fires_outside_incident_language(tmp_path):
    seen = []
    run_tool_loop([Message("user", "SYN taramasi nedir?")], lambda m: seen.append(1) or "Aciklama.",
                  _reg(tmp_path), evidence_push=True)
    assert len(seen) == 1
