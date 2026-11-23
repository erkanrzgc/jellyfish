"""The offline demo checks behavior and must fail when a backend stops working."""
from agent import smoke
from agent.backends.assistant_executor import AssistantExecutor
from agent.executor import MockExecutor


def test_smoke_exercises_scan_scope_and_real_files(tmp_path):
    results = smoke.run_checks(tmp_path)
    assert [r.name for r in results] == ["mock-scan", "scope-denial", "workspace-files"]
    assert all(r.passed for r in results)
    assert (tmp_path / "workspace" / "note.txt").read_text(encoding="utf-8") == "Jellyfish lab note"
    assert not (tmp_path / "escaped.txt").exists()


def test_smoke_reports_failed_security_executor(monkeypatch, tmp_path):
    monkeypatch.setattr(MockExecutor, "run", lambda *args: "HATA: backend failed")
    results = {r.name: r for r in smoke.run_checks(tmp_path)}
    assert not results["mock-scan"].passed
    assert results["scope-denial"].passed


def test_smoke_reports_failed_file_executor(monkeypatch, tmp_path):
    monkeypatch.setattr(AssistantExecutor, "run", lambda *args: "HATA: filesystem failed")
    results = {r.name: r for r in smoke.run_checks(tmp_path)}
    assert not results["workspace-files"].passed


def test_smoke_command_exits_nonzero_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "run_checks", lambda root: [smoke.CheckResult("broken", False, "failure")])
    assert smoke.main() == 1
    assert "[FAIL] broken" in capsys.readouterr().out
