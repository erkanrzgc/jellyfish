import subprocess

from agent.audit import AuditLog
from agent.backends.assistant_executor import AssistantExecutor
from agent.backends.sandbox_executor import DockerSandboxExecutor, _wrap
from agent.catalog import get_spec
from agent.policy import LabPolicy
from agent.registry import ToolRegistry
from agent.toolcall import ToolCall, parse_tool_reply


def test_targetless_network_tool_cannot_hide_a_target_in_options():
    assert not LabPolicy(scope=["10.10.10.0/24"]).decide(
        get_spec("fping"), {"secenekler": "192.0.2.1"},
    ).allowed


def test_secondary_dns_resolver_is_not_a_record_type():
    assert not LabPolicy(scope=["lab.example"]).decide(
        get_spec("dig"), {"domain": "lab.example", "tip": "@192.0.2.1"},
    ).allowed


def test_wrong_parameter_value_type_is_a_parse_error():
    result = parse_tool_reply('```arac\n{"arac":"read_file","parametreler":{"yol":42}}\n```', validate_catalog=True)
    assert result.errors and not result.calls


def test_missing_required_file_parameter_is_a_parse_error():
    result = parse_tool_reply('```arac\n{"arac":"read_file","parametreler":{}}\n```', validate_catalog=True)
    assert result.errors and not result.calls


def test_real_registry_does_not_offer_unvalidated_tool_adapter(tmp_path):
    from agent.backends.real_executor import RealExecutor
    registry = ToolRegistry(LabPolicy(allow_high=True), RealExecutor(), AuditLog(tmp_path / "audit.jsonl"))
    result = registry.invoke_result(ToolCall("msfvenom", {}))
    assert result.status == "failed" and result.proposal_id is None


def test_direct_registry_wrong_path_type_is_denied_without_raising(tmp_path):
    registry = ToolRegistry(LabPolicy(workspace_root=str(tmp_path)), AssistantExecutor(str(tmp_path)),
                            AuditLog(tmp_path / "audit.jsonl"))
    assert registry.invoke_result(ToolCall("read_file", {"yol": 42})).status == "denied"


def test_nonzero_sandbox_exit_is_failed(tmp_path):
    executor = DockerSandboxExecutor(runner=lambda *args, **kwargs: subprocess.CompletedProcess(
        [], 125, "", "Docker daemon unavailable"))
    registry = ToolRegistry(LabPolicy(allow_high=True), executor, AuditLog(tmp_path / "audit.jsonl"))
    call = ToolCall("run_cmd", {"komut": "echo test"})
    proposal = registry.invoke_result(call)
    result = registry.invoke_result(call, approval_token=registry.approve(proposal.proposal_id))
    assert result.status == "failed" and "125" in result.output


def test_sandbox_wrapper_preserves_inner_command_status():
    wrapped = _wrap("exit 7")
    assert 'exit "$jellyfish_status"' in wrapped
    assert "jellyfish_status=$?" in wrapped


def test_sandbox_wrapper_exit_status_with_local_posix_shell(tmp_path):
    import os
    import shutil
    import pytest
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("POSIX shell unavailable on this Windows host; CI runs the command test")
    result = subprocess.run([shell, "-c", _wrap("echo fixture-output; exit 7")],
                            env={**os.environ, "TMPDIR": str(tmp_path)},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 7
    assert "fixture-output" in result.stdout
