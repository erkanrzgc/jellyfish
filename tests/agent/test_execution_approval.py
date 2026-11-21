from dataclasses import dataclass

from agent.audit import AuditLog
from agent.executor import MockExecutor
from agent.policy import LabPolicy
from agent.registry import ToolRegistry
from agent.toolcall import ToolCall


@dataclass
class RealLike:
    calls: int = 0

    def preview(self, tool, params):
        return [tool, params["hedef"]]

    def run(self, tool, params):
        self.calls += 1
        return "22/tcp open ssh"


def registry(tmp_path, executor=None):
    return ToolRegistry(LabPolicy(scope=["10.10.10.0/24"]),
                        executor or RealLike(), AuditLog(tmp_path / "audit.jsonl"))


def test_real_execution_waits_for_host_approval(tmp_path):
    reg = registry(tmp_path)
    result = reg.invoke_result(ToolCall("nmap", {"hedef": "10.10.10.5"}))
    assert result.status == "awaiting_approval"
    assert result.command == ["nmap", "10.10.10.5"]
    assert reg.executor.calls == 0


def test_approval_is_one_use_and_bound_to_parameters(tmp_path):
    reg = registry(tmp_path)
    call = ToolCall("nmap", {"hedef": "10.10.10.5"})
    proposal = reg.invoke_result(call)
    token = reg.approve(proposal.proposal_id)
    changed = reg.invoke_result(ToolCall("nmap", {"hedef": "10.10.10.6"}), approval_token=token)
    assert changed.status == "awaiting_approval" and reg.executor.calls == 0
    # A mismatching attempt invalidates the token instead of preserving stale permission.
    assert reg.invoke_result(call, approval_token=token).status == "awaiting_approval"
    fresh = reg.invoke_result(call)
    token = reg.approve(fresh.proposal_id)
    assert reg.invoke_result(call, approval_token=token).status == "executed"
    assert reg.executor.calls == 1
    assert reg.invoke_result(call, approval_token=token).status == "awaiting_approval"


def test_scope_change_invalidates_approval(tmp_path):
    reg = registry(tmp_path)
    call = ToolCall("nmap", {"hedef": "10.10.10.5"})
    token = reg.approve(reg.invoke_result(call).proposal_id)
    reg.policy.scope.append("192.0.2.0/24")
    assert reg.invoke_result(call, approval_token=token).status == "awaiting_approval"
    assert reg.executor.calls == 0


def test_hard_denial_cannot_be_approved(tmp_path):
    reg = registry(tmp_path)
    result = reg.invoke_result(ToolCall("nmap", {"hedef": "192.0.2.1"}))
    assert result.status == "denied" and result.proposal_id is None


def test_mock_is_not_reported_as_real_but_can_run_offline(tmp_path):
    result = registry(tmp_path, MockExecutor()).invoke_result(ToolCall("nmap", {"hedef": "10.10.10.5"}))
    assert result.status == "executed" and result.simulated


def test_failure_is_not_an_executed_success(tmp_path):
    class Failed(MockExecutor):
        def run(self, tool, params):
            return "HATA: tool failed"
    result = registry(tmp_path, Failed()).invoke_result(ToolCall("nmap", {"hedef": "10.10.10.5"}))
    assert result.status == "failed"


def test_approval_preview_does_not_share_mutable_call_params(tmp_path):
    reg = registry(tmp_path)
    call = ToolCall("nmap", {"hedef": "10.10.10.5"})
    result = reg.invoke_result(call)
    call.params["hedef"] = "10.10.10.6"
    assert result.call.params["hedef"] == "10.10.10.5"


def test_high_risk_approval_does_not_change_global_policy(tmp_path):
    class High(MockExecutor):
        is_mock = False
        def preview(self, tool, params): return [tool]
    reg = registry(tmp_path, High())
    call = ToolCall("msfvenom", {})
    proposal = reg.invoke_result(call)
    assert proposal.status == "awaiting_approval"
    token = reg.approve(proposal.proposal_id)
    assert reg.invoke_result(call, approval_token=token).status == "executed"
    assert reg.policy.allow_high is False


def test_expired_approval_never_executes(tmp_path):
    from agent.execution import ApprovalStore
    now = [10.0]
    reg = registry(tmp_path)
    reg.approvals = ApprovalStore(lifetime=5, clock=lambda: now[0])
    call = ToolCall("nmap", {"hedef": "10.10.10.5"})
    token = reg.approve(reg.invoke_result(call).proposal_id)
    now[0] = 16.0
    assert reg.invoke_result(call, approval_token=token).status == "awaiting_approval"
    assert reg.executor.calls == 0


def test_changed_preview_invalidates_approval(tmp_path):
    reg = registry(tmp_path)
    call = ToolCall("nmap", {"hedef": "10.10.10.5"})
    token = reg.approve(reg.invoke_result(call).proposal_id)
    reg.executor.preview = lambda tool, params: [tool, "--different-operation", params["hedef"]]
    assert reg.invoke_result(call, approval_token=token).status == "awaiting_approval"
    assert reg.executor.calls == 0
