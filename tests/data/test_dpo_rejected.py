"""Rejected stage: run the local model on-policy and label structural failures."""
from __future__ import annotations

import json
from types import SimpleNamespace

from data.dpo.rejected import relabel, run

CASES = [
    {"id": "DPO-TOOL-0001", "family_id": "f1", "category": "tooling", "prompt": "10.0.0.5 portlarını tara",
     "expected_tools": ["nmap"], "expected_params": {"hedef": "10.0.0.5"}, "allow_tools": True},
    {"id": "DPO-FUND-0001", "family_id": "f2", "category": "fundamentals", "prompt": "SYN taraması nedir?",
     "expected_tools": [], "expected_params": {}, "allow_tools": False},
]


class _Model:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate_result(self, messages):
        self.seen.append(messages[-1].content)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            return SimpleNamespace(text="", error=str(reply), done_reason=None, truncated=False, metadata={})
        return SimpleNamespace(text=reply, error=None, done_reason="stop", truncated=False, metadata={})


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_labels_failures_and_passes(tmp_path):
    cases, out = tmp_path / "c.jsonl", tmp_path / "r.jsonl"
    _write(cases, CASES)
    model = _Model(["Sadece anlatıyorum.", '```arac\n{"arac":"nmap","parametreler":{"hedef":"1.1.1.1"}}\n```'])
    stats = run(model, cases, out)
    rows = {r["id"]: r for r in _read(out)}
    assert rows["DPO-TOOL-0001"]["failures"] == ["missing_call"]
    assert rows["DPO-FUND-0001"]["failures"] == ["unnecessary_call"]
    assert stats == {"done": 2, "failed": 2, "infrastructure": 0, "skipped": 0}


def test_resume_skips_finished_cases(tmp_path):
    cases, out = tmp_path / "c.jsonl", tmp_path / "r.jsonl"
    _write(cases, CASES)
    run(_Model(["ok", "SYN yarı açıktır."]), cases, out)
    model = _Model([])
    stats = run(model, cases, out)
    assert model.seen == [] and stats["skipped"] == 2 and len(_read(out)) == 2


def test_relabel_recomputes_failures_from_stored_replies(tmp_path):
    cases, out = tmp_path / "c.jsonl", tmp_path / "r.jsonl"
    _write(cases, CASES)
    good = '```arac\n{"arac":"nmap","parametreler":{"hedef":"10.0.0.5"}}\n```'
    _write(out, [{"id": "DPO-TOOL-0001", "raw_reply": good, "failures": ["param_mismatch"], "done_reason": "stop"},
                 {"id": "DPO-FUND-0001", "raw_reply": "SYN yarı açıktır.", "failures": [], "done_reason": "stop"}])
    stats = relabel(cases, out)
    rows = _read(out)
    assert stats == {"rows": 2, "changed": 1, "failed": 0}
    assert rows[0]["failures"] == [] and rows[0]["done_reason"] == "stop"


def test_infrastructure_error_is_not_a_model_failure_and_is_retried(tmp_path):
    cases, out = tmp_path / "c.jsonl", tmp_path / "r.jsonl"
    _write(cases, CASES[:1])
    stats = run(_Model([RuntimeError("ollama down")]), cases, out)
    assert stats["infrastructure"] == 1 and not out.exists()
