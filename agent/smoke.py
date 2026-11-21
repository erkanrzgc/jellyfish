"""Offline acceptance demos: scripted model, simulated scans, real temporary files.

Run from the repository root: python -m agent.smoke
No Ollama, Docker, network access, or security binaries are used.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent.audit import AuditLog
from agent.backends.assistant_executor import AssistantExecutor
from agent.backends.mock_model import ScriptedModel
from agent.composite_executor import CompositeExecutor
from agent.executor import MockExecutor
from agent.loop import run_tool_loop
from agent.messages import Message
from agent.policy import LabPolicy
from agent.registry import ToolRegistry
from agent.toolcall import ToolCall


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class _ObservedMock(MockExecutor):
    calls: int = 0

    def run(self, tool: str, params: dict) -> str:
        self.calls += 1
        return super().run(tool, params)


def run_checks(root: Path) -> list[CheckResult]:
    """Exercise the existing runtime; keep all files and audit logs under root."""
    root.mkdir(parents=True, exist_ok=True)
    security = _ObservedMock()
    registry = ToolRegistry(
        LabPolicy(scope=["10.10.10.0/24"]), security, AuditLog(root / "scan.jsonl"),
    )

    def scan(target: str) -> str:
        call = {"arac": "nmap", "parametreler": {"hedef": target, "secenekler": "-sV"}}
        model = ScriptedModel([f"```arac\n{json.dumps(call)}\n```", "Demo completed."])
        messages = [Message("user", f"Scan lab target {target}")]
        run_tool_loop(messages, model, registry)
        return "\n".join(m.content for m in messages if m.role == "tool")

    output = scan("10.10.10.5")
    scan_ok = security.calls == 1 and "22/ssh" in output and "HATA" not in output
    results = [CheckResult("mock-scan", scan_ok, "scripted call -> policy -> simulated port output")]

    before = security.calls
    denied = scan("192.0.2.1")
    results.append(CheckResult(
        "scope-denial", denied.startswith("REDDEDILDI") and security.calls == before,
        "out-of-scope target rejected; executor not called",
    ))

    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    files = ToolRegistry(
        LabPolicy(workspace_root=str(workspace)),
        CompositeExecutor(security, AssistantExecutor(str(workspace))),
        AuditLog(root / "files.jsonl"),
    )
    content = "Octopus lab note"
    def approved_file_call(call):
        result = files.invoke_result(call)
        if result.status == "awaiting_approval":
            # Trusted smoke fixture authorizes only these concrete temporary-file operations.
            return files.invoke_result(call, approval_token=files.approve(result.proposal_id)).output
        return result.output
    written = approved_file_call(ToolCall("write_file", {"yol": "note.txt", "icerik": content}))
    read = approved_file_call(ToolCall("read_file", {"yol": "note.txt"}))
    escape = files.invoke(ToolCall("write_file", {"yol": "../escaped.txt", "icerik": "blocked"}))
    note = workspace / "note.txt"
    files_ok = (
        written.startswith("yazildi") and read == content
        and note.exists() and note.read_text(encoding="utf-8") == content
        and escape.startswith("REDDEDILDI") and not (root / "escaped.txt").exists()
    )
    results.append(CheckResult(
        "workspace-files", files_ok, "real file write/read; path escape rejected",
    ))
    return results


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="octopus-smoke-") as directory:
        results = run_checks(Path(directory))
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    passed = sum(r.passed for r in results)
    print(f"{passed}/{len(results)} offline checks passed (no live model or security tools).")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
