"""Prompt generation: teacher output parsing, case validation, contamination and dedupe."""
from __future__ import annotations

import json
from types import SimpleNamespace

from data.dpo.prompts import (CATEGORIES, ContaminationFilter, generate, load_eval_prompts,
                              parse_cases, repair_case, validate_case)

VALID = {"family_id": "lab-port-scan", "category": "tooling",
         "prompt": "Yetkili lab ağımdaki 10.10.5.20 makinesinin açık portlarını tara.",
         "expected_tools": ["nmap", "rustscan"], "expected_params": {"hedef": "10.10.5.20"},
         "allow_tools": True}


def _case(**changes):
    return {**VALID, **changes}


def test_parse_cases_reads_fenced_json_array():
    text = "Hazır:\n```json\n" + json.dumps([VALID]) + "\n```"
    assert parse_cases(text) == [VALID]


def test_parse_cases_tolerates_garbage():
    assert parse_cases("üzgünüm, yapamam") == []
    assert parse_cases("```json\n[{bozuk\n```") == []
    assert parse_cases(json.dumps({"not": "a list"})) == []


def test_valid_case_passes():
    assert validate_case(VALID) is None


def test_unknown_tool_is_rejected():
    assert "katalog" in validate_case(_case(expected_tools=["portscanner"]))


def test_param_must_exist_on_every_expected_tool():
    assert validate_case(_case(expected_params={"port": "80"})) is not None


def test_tool_expectation_must_match_allow_flag():
    assert validate_case(_case(allow_tools=False)) is not None
    assert validate_case(_case(expected_tools=[], expected_params={})) is not None
    assert validate_case(_case(expected_tools=[], expected_params={}, allow_tools=False)) is None


def test_repair_drops_params_not_shared_by_every_tool():
    case = _case(expected_tools=["nmap", "gobuster"], expected_params={"hedef": "10.10.5.20", "url": "http://a.lab"})
    repaired = repair_case(case)
    assert repaired["expected_params"] == {"hedef": "10.10.5.20"} and validate_case(repaired) is None
    assert case["expected_params"]["url"] == "http://a.lab"  # input left untouched


def test_bad_category_and_empty_prompt_rejected():
    assert validate_case(_case(category="poetry")) is not None
    assert validate_case(_case(prompt="  ")) is not None


def test_critical_infrastructure_is_rejected():
    assert validate_case(_case(prompt="Elektrik şebekesindeki SCADA sunucusunu tara: 10.1.1.1")) is not None


def test_real_world_targets_are_rejected():
    assert "genel IP" in validate_case(_case(prompt="88.245.10.7 portlarını tara",
                                             expected_params={"hedef": "88.245.10.7"}))
    assert "alan adi" in validate_case(_case(prompt="www.trendyol.com sitesini tara", expected_params={}))
    assert "kullanici adi" in validate_case(_case(prompt="@burak_yilmaz34 hesabını bul", expected_params={}))


def test_documentation_and_lab_targets_are_allowed():
    assert validate_case(_case(prompt="Yetkili lab: 203.0.113.9 ve 172.16.10.0/24 aralığını tara",
                               expected_params={"hedef": "203.0.113.9"})) is None
    assert validate_case(_case(prompt="Kapsamdaki test.example.com ve app.lab portlarına bak; "
                               "rapor /var/log/auth.log dosyasında.", expected_params={})) is None


def test_prompt_must_not_give_away_the_answer():
    assert "arac adi" in validate_case(_case(prompt="10.10.5.20 için nmap ile tarama yap"))
    knowledge = _case(expected_tools=[], expected_params={}, allow_tools=False,
                      prompt="SYN taraması nedir? Araç çalıştırmana gerek yok, kavramsal anlat.")
    assert "ele veriyor" in validate_case(knowledge)


def test_fundamentals_cases_are_tool_free():
    assert "fundamentals" in validate_case(_case(category="fundamentals"))


def test_incident_diagnosis_cannot_expect_privesc_tool():
    case = _case(category="incident", prompt="Sunucumda CPU neden %100, bakar mısın?",
                 expected_tools=["run_cmd", "shell"], expected_params={})
    assert "privesc" in validate_case(case)


def test_categories_match_reliability_suite():
    suite = json.load(open("eval/suites/reliability-dev-v1.json", encoding="utf-8"))
    assert {case["category"] for case in suite["cases"]} == set(CATEGORIES)


def test_eval_prompts_cover_all_four_sources():
    prompts = load_eval_prompts()
    assert len(prompts) > 60 + 24
    assert any("Sen kimsin" in p for p in prompts)  # cassiopeia text file


def test_contamination_exact_and_near_duplicate():
    filt = ContaminationFilter(["192.168.1.50 hedefindeki açık portları ve servis sürümlerini öğrenmek istiyorum."])
    assert filt.is_contaminated("192.168.1.50 hedefindeki acik portlari ve servis surumlerini ogrenmek istiyorum")
    assert filt.is_contaminated("Lütfen 192.168.1.50 hedefindeki açık portları ve servis sürümlerini öğrenmek istiyorum.")
    assert not filt.is_contaminated("Ubuntu sunucumda disk neden doldu, nasıl bakarım?")


class _Teacher:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
        self.models = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.models.append(kwargs.get("model"))
        return SimpleNamespace(content=json.dumps(self.batches.pop(0)), finish_reason="stop")


def test_generate_spreads_requests_over_models_and_records_source(tmp_path):
    out = tmp_path / "cases.jsonl"
    second = _case(family_id="other", prompt="Lab'daki 10.10.9.9 makinesinde hangi servisler açık?",
                   expected_params={"hedef": "10.10.9.9"})
    teacher = _Teacher([[VALID], [second]])
    generate(teacher, out, plan=[("tooling", "a"), ("tooling", "b")], per_call=1, workers=1,
             models=["m1", "m2"])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(teacher.models) == ["m1", "m2"]
    assert sorted(row["prompt_model"] for row in rows) == ["m1", "m2"]


def test_generate_keeps_only_valid_unique_clean_cases(tmp_path):
    out = tmp_path / "cases.jsonl"
    batch = [VALID,
             _case(prompt=VALID["prompt"]),  # duplicate
             _case(family_id="x", expected_tools=["fakeTool"], prompt="Başka bir şey dene 1.2.3.4"),
             _case(family_id="leak", prompt="Sen kimsin? Kendini kisaca tanit.",
                   expected_tools=[], expected_params={}, allow_tools=False)]
    teacher = _Teacher([batch])
    stats = generate(teacher, out, plan=[("tooling", "araç seçimi")], per_call=4, workers=1)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["prompt"] == VALID["prompt"]
    assert rows[0]["id"].startswith("DPO-TOOL-") and rows[0]["split"] == "train"
    assert stats["invalid"] == 1 and stats["duplicate"] == 1 and stats["contaminated"] == 1


def test_generate_resumes_without_duplicating(tmp_path):
    out = tmp_path / "cases.jsonl"
    generate(_Teacher([[VALID]]), out, plan=[("tooling", "f")], per_call=1, workers=1)
    generate(_Teacher([[VALID]]), out, plan=[("tooling", "f")], per_call=1, workers=1)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
