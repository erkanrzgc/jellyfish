"""Execution outcomes and one-use, expiring approvals issued only by host code."""
from __future__ import annotations

from dataclasses import dataclass
import secrets
import time

from agent.toolcall import ToolCall


@dataclass(frozen=True)
class ExecutionResult:
    call: ToolCall
    status: str
    output: str
    command: list[str] | None = None
    proposal_id: str | None = None
    simulated: bool = False


class ApprovalStore:
    def __init__(self, lifetime: float = 300, clock=time.monotonic):
        self.lifetime, self.clock = lifetime, clock
        self._proposals: dict[str, tuple[str, float]] = {}
        self._tokens: dict[str, tuple[str, float]] = {}

    def propose(self, binding: str) -> str:
        now = self.clock()
        self._proposals = {k: v for k, v in self._proposals.items() if v[1] > now}
        proposal = secrets.token_urlsafe(24)
        self._proposals[proposal] = (binding, now + self.lifetime)
        return proposal

    def approve(self, proposal: str) -> str:
        record = self._proposals.pop(proposal, None)
        if record is None or record[1] <= self.clock():
            raise ValueError("onay onerisi bulunamadi veya suresi doldu")
        token = secrets.token_urlsafe(32)
        self._tokens[token] = record
        return token

    def consume(self, token: str | None, binding: str) -> bool:
        record = self._tokens.pop(token, None) if token else None
        return bool(record and record[0] == binding and record[1] > self.clock())
