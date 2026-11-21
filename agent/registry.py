"""Katalog + policy + executor + audit'i baglar. invoke(): yetki kontrol -> calistir -> logla.
Asla exception firlatmaz; hatayi metin olarak modele geri dondurur (dongu olmesin)."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from copy import deepcopy
import json
from agent.audit import AuditLog
from agent.catalog import CATALOG, get_spec, parameter_error
from agent.executor import Executor, MockExecutor
from agent.policy import LabPolicy
from agent.toolcall import ToolCall
from agent.execution import ApprovalStore, ExecutionResult


@dataclass
class ToolRegistry:
    policy: LabPolicy
    executor: Executor
    audit: AuditLog
    allowed: frozenset[str] | None = None   # None = scope YOK (tam katalog); aksi halde rol dilimi
    approvals: ApprovalStore = field(default_factory=ApprovalStore)

    @classmethod
    def default(cls) -> "ToolRegistry":
        return cls(LabPolicy.default(), MockExecutor(), AuditLog.default())

    @classmethod
    def scoped(cls, role: str, *, policy: "LabPolicy | None" = None,
               executor: "Executor | None" = None, audit: "AuditLog | None" = None) -> "ToolRegistry":
        """role dilimiyle (role_tools) kisitli registry. Diger bilesenler default'a duser."""
        from agent.roles import role_tools
        return cls(policy or LabPolicy.default(), executor or MockExecutor(),
                   audit or AuditLog.default(), allowed=role_tools(role))

    def tool_names(self) -> list[str]:
        return sorted(self.allowed) if self.allowed is not None else list(CATALOG)

    def invoke(self, call: ToolCall) -> str:
        """Compatibility API; approval-aware callers should use invoke_result."""
        return self.invoke_result(call).output

    def approve(self, proposal_id: str) -> str:
        """Host/UI API. Model-generated text is never passed to this method."""
        return self.approvals.approve(proposal_id)

    def _backend(self, call: ToolCall):
        backend = self.executor
        if hasattr(backend, "executor_for"):
            backend = backend.executor_for(call.name)
        if hasattr(backend, "execution_backend"):
            backend = backend.execution_backend(call.name)
        return backend

    def invoke_result(self, call: ToolCall, *, approval_token: str | None = None) -> ExecutionResult:
        call = ToolCall(call.name, deepcopy(call.params))
        self.audit.write("tool.proposed", call.name)
        def outcome(status: str, output: str, **kwargs):
            self.audit.write(f"tool.{status}", call.name)
            return ExecutionResult(call, status, output, **kwargs)
        if self.allowed is not None and call.name not in self.allowed:
            msg = (f"BU ROL İÇİN DEĞİL: '{call.name}'. Bu görevde şu araçlardan seç: "
                   f"{', '.join(sorted(self.allowed))}")
            return outcome("denied", msg)
        spec = get_spec(call.name)
        if spec is None:
            msg = f"HATA: bilinmeyen arac '{call.name}'"
            return outcome("denied", msg)
        if parameter_error(spec, call.params):
            return outcome("denied", "REDDEDILDI: bilinmeyen/gecersiz parametre")
        decision = self.policy.decide(spec, call.params)
        if decision.requires_approval:
            # Recheck every hard constraint without changing the shared policy.
            decision = replace(self.policy, allow_high=True).decide(spec, call.params)
        if not decision.allowed:
            return outcome("denied", f"REDDEDILDI: {decision.reason}")
        backend = self._backend(call)
        simulated = bool(getattr(backend, "is_mock", False))
        command = None
        try:
            if not simulated:
                preview = getattr(backend, "preview", None)
                if preview is None:
                    return outcome("failed", "HATA: executor icin dogrulanmis komut onizlemesi yok")
                command = list(preview(call.name, call.params))
                context = {
                    "scope": sorted(self.policy.scope), "workspace": self.policy.workspace_root,
                    "allow_high": self.policy.allow_high, "role_tools": sorted(self.allowed) if self.allowed is not None else None,
                    "executor": type(backend).__qualname__, "executor_id": id(backend),
                    "command": command, "tool": call.name, "params": call.params,
                }
                binding = json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False)
                if not self.approvals.consume(approval_token, binding):
                    proposal = self.approvals.propose(binding)
                    return outcome("awaiting_approval", f"ONAY GEREKLI: {call.name}",
                                   command=command, proposal_id=proposal)
            elif spec.risk == "high" and not self.policy.allow_high:
                return outcome("awaiting_approval", f"ONAY GEREKLI: '{call.name}' yuksek riskli", simulated=True)
        except (ValueError, TypeError, OSError) as error:
            return outcome("failed", f"HATA: komut hazirlanamadi ({error})")
        try:
            out = self.executor.run(call.name, call.params)
        except Exception as exc:  # noqa: BLE001 - modele bildir, cokme
            out = f"HATA: {type(exc).__name__}: {exc}"
        failed = not isinstance(out, str) or out.startswith(("HATA", "REDDEDILDI", "ONAY GEREKLI"))
        return outcome("failed" if failed else "executed", str(out), command=command, simulated=simulated)
