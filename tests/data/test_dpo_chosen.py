"""Chosen stage: teacher reply must fix the failure and pass the structural gate."""
from __future__ import annotations

import json
from types import SimpleNamespace

from data.dpo.chosen import acceptance_problems, build_teacher_messages, run

TOOL_CASE = {"id": "DPO-TOOL-0001", "family_id": "f1", "category": "tooling",
             "prompt": "Yetkili lab: 10.0.0.5 portlarını tara", "expected_tools": ["nmap"],
             "expected_params": {"hedef": "10.0.0.5"}, "allow_tools": True}
KNOW_CASE = {"id": "DPO-FUND-0001", "family_id": "f2", "category": "fundamentals",
             "prompt": "SYN taraması neden yarı açık sayılır?", "expected_tools": [],
             "expected_params": {}, "allow_tools": False}

GOOD_TOOL = ("```dusunce\nYetkili lab, port taraması için nmap.\n```\nAçık portlara bakıyorum.\n"
             '```arac\n{"arac": "nmap", "parametreler": {"hedef": "10.0.0.5", "secenekler": "-sV"}}\n```')
GOOD_KNOW = "SYN taramasında üçlü el sıkışma tamamlanmaz; bağlantı yarı açık kalır."


def test_good_replies_are_accepted():
    assert acceptance_problems(TOOL_CASE, GOOD_TOOL) == []
    assert acceptance_problems(KNOW_CASE, GOOD_KNOW) == []


def test_structural_failures_and_extra_calls_are_rejected():
    assert acceptance_problems(TOOL_CASE, GOOD_KNOW) == ["missing_call"]
    two = GOOD_TOOL + '\n```arac\n{"arac": "whois", "parametreler": {"hedef": "10.0.0.5"}}\n```'
    assert "multiple_calls" in acceptance_problems(TOOL_CASE, two)


def test_english_or_block_only_replies_are_rejected():
    assert "not_turkish" in acceptance_problems(KNOW_CASE, "SYN scans never finish the handshake.")
    block_only = '```arac\n{"arac": "nmap", "parametreler": {"hedef": "10.0.0.5"}}\n```'
    assert "no_prose" in acceptance_problems(TOOL_CASE, block_only)


def test_synthetic_target_commentary_is_rejected():
    leaks = [
        "```dusunce\nexample.com rezerve bir alan adı olduğu için yetkili sayılır.\n```\nTarıyorum.",
        "Bu adres 203.0.113.0/24 dokümantasyon aralığında, gerçek hedef değil; yapamam.",
        "```dusunce\n10.10.15.50 özel (RFC1918) bir adres; lab olduğu varsayımıyla devam.\n```\nÇalıştırıyorum.",
    ]
    for text in leaks:
        assert "target_commentary" in acceptance_problems(KNOW_CASE, text), text


def test_run_cmd_must_not_be_wrapped_in_ssh():
    case = {**TOOL_CASE, "expected_tools": ["run_cmd"], "expected_params": {}}
    wrapped = 'Bakıyorum.\n```arac\n{"arac": "run_cmd", "parametreler": {"komut": "ssh 10.0.0.5 \'uptime\'"}}\n```'
    local = 'Bakıyorum.\n```arac\n{"arac": "run_cmd", "parametreler": {"komut": "uptime"}}\n```'
    assert "ssh_wrapped" in acceptance_problems(case, wrapped)
    assert acceptance_problems(case, local) == []


def test_teacher_messages_carry_persona_expectation_and_only_relevant_tools():
    text = json.dumps(build_teacher_messages(TOOL_CASE), ensure_ascii=False)
    assert "Octópus" in text and "10.0.0.5" in text and "nmap(hedef, secenekler)" in text
    assert "sqlmap(" not in text
    know = json.dumps(build_teacher_messages(KNOW_CASE), ensure_ascii=False)
    assert "araç ÇAĞIRMA" in know


class _Teacher:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.replies.pop(0), finish_reason="stop")


def _setup(tmp_path, rejected_rows):
    cases, rejected = tmp_path / "c.jsonl", tmp_path / "r.jsonl"
    cases.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in (TOOL_CASE, KNOW_CASE)),
                     encoding="utf-8")
    rejected.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected_rows),
                        encoding="utf-8")
    return cases, rejected, tmp_path / "ch.jsonl"


def test_only_failed_cases_go_to_teacher_and_retry_until_valid(tmp_path):
    cases, rejected, out = _setup(tmp_path, [
        {"id": "DPO-TOOL-0001", "raw_reply": "anlattım", "failures": ["missing_call"]},
        {"id": "DPO-FUND-0001", "raw_reply": GOOD_KNOW, "failures": []},
    ])
    teacher = _Teacher(["yanlış, araç yok", GOOD_TOOL])
    stats = run(teacher, cases, rejected, out, workers=1)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert teacher.calls == 2 and stats == {"accepted": 1, "dropped": 0, "skipped": 0, "teacher_errors": 0}
    assert rows == [{"id": "DPO-TOOL-0001", "chosen": GOOD_TOOL, "attempts": 2,
                     "teacher_model": "deepseek-v4.1-flash", "problems": []}]


def test_drop_after_retries_records_reasons(tmp_path):
    cases, rejected, out = _setup(tmp_path, [
        {"id": "DPO-TOOL-0001", "raw_reply": "x", "failures": ["missing_call"]}])
    stats = run(_Teacher(["a", "b", "c"]), cases, rejected, out, workers=1, retries=2)
    row = json.loads(out.read_text(encoding="utf-8"))
    assert stats["dropped"] == 1 and row["chosen"] is None and row["attempts"] == 3
    assert "missing_call" in row["problems"]


def test_gateway_errors_leave_case_for_a_later_run(tmp_path):
    from data.dpo.evren_client import EvrenError

    class _Down:
        def chat(self, messages, **kwargs):
            raise EvrenError("HTTP 503: busy")

    cases, rejected, out = _setup(tmp_path, [
        {"id": "DPO-TOOL-0001", "raw_reply": "x", "failures": ["missing_call"]}])
    stats = run(_Down(), cases, rejected, out, workers=1, retries=1)
    assert stats["teacher_errors"] == 1 and not out.exists()


def test_stale_labels_are_recomputed_so_equivalent_tools_skip_the_teacher(tmp_path):
    rustscan = {**TOOL_CASE, "id": "DPO-TOOL-0002", "expected_tools": ["rustscan"]}
    cases, rejected, out = _setup(tmp_path, [
        {"id": "DPO-TOOL-0001", "raw_reply": "anlattım", "failures": []},  # stale: really missing_call
        {"id": "DPO-TOOL-0002", "raw_reply": GOOD_TOOL, "failures": ["wrong_first_tool"]},  # stale: nmap is fine
    ])
    with cases.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rustscan, ensure_ascii=False) + "\n")
    teacher = _Teacher([GOOD_TOOL])
    stats = run(teacher, cases, rejected, out, workers=1)
    assert teacher.calls == 1 and stats["accepted"] == 1
    assert [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()] == ["DPO-TOOL-0001"]


def test_requests_are_spread_round_robin_over_teacher_models(tmp_path):
    know_fail = {"id": "DPO-FUND-0001", "raw_reply": GOOD_TOOL, "failures": ["unnecessary_call"]}
    cases, rejected, out = _setup(tmp_path, [
        {"id": "DPO-TOOL-0001", "raw_reply": "anlattım", "failures": ["missing_call"]}, know_fail])
    seen = []

    class _Recording(_Teacher):
        def chat(self, messages, **kwargs):
            seen.append(kwargs.get("model"))
            return SimpleNamespace(content=GOOD_TOOL if "nmap" in json.dumps(messages, ensure_ascii=False)
                                   and "portlarını" in json.dumps(messages, ensure_ascii=False) else GOOD_KNOW,
                                   finish_reason="stop")

    run(_Recording([]), cases, rejected, out, workers=1, models=("m1", "m2"))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(seen) == ["m1", "m2"] and sorted(row["teacher_model"] for row in rows) == ["m1", "m2"]
