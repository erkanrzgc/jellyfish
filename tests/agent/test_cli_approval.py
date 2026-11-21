import sys

from agent import chat, cli


class RealLike:
    def __init__(self): self.calls = 0
    def preview(self, tool, params): return [tool, params["hedef"]]
    def run(self, tool, params):
        self.calls += 1
        return "22/tcp open ssh"


def test_legacy_real_demo_decline_is_visible_and_never_says_finished(monkeypatch):
    shown = []
    original = chat.approval_prompt
    monkeypatch.setattr(chat, "approval_prompt", lambda registry: original(
        registry, read=lambda prompt: "hayir", write=shown.append))
    executor = RealLike()
    out = cli._scan_demo(executor, "10.10.10.5", ["10.10.10.0/24"], "-sV", "gercek demo bitti")
    assert shown and executor.calls == 0
    assert out.status == "awaiting_approval"
    assert "Gorev tamamlanmadi" in out and "gercek demo bitti" not in out


def test_legacy_real_demo_approved_call_runs_once(monkeypatch):
    original = chat.approval_prompt
    monkeypatch.setattr(chat, "approval_prompt", lambda registry: original(
        registry, read=lambda prompt: "evet", write=lambda text: None))
    executor = RealLike()
    out = cli._scan_demo(executor, "10.10.10.5", ["10.10.10.0/24"], "-sV", "gercek demo bitti")
    assert executor.calls == 1 and out.status == "completed"
    assert "gercek demo bitti" in out


def test_demo_main_exits_nonzero_for_unfinished_run(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(cli, "run_demo", lambda scope: cli.DemoTranscript("failed fixture", "failed"))
    assert cli.main() == 2


def test_assistant_fixture_failure_is_not_reported_finished(monkeypatch, tmp_path):
    from agent.backends.assistant_executor import AssistantExecutor
    monkeypatch.setattr(AssistantExecutor, "run", lambda *args: "HATA: fixture unavailable")
    out = cli.run_assistant_demo(workspace=str(tmp_path))
    assert out.status == "failed" and "asistan demo bitti" not in out
