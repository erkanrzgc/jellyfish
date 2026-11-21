"""Shared DPO helpers: Turkish-aware normalization, failure labels, catalog text."""
from __future__ import annotations

from data.dpo.common import catalog_excerpt, classify_failures, fold, jaccard, prose_outside_blocks

NMAP_CASE = {"expected_tools": ["nmap", "rustscan"], "expected_params": {"hedef": "10.0.0.5"},
             "allow_tools": True}
KNOWLEDGE_CASE = {"expected_tools": [], "expected_params": {}, "allow_tools": False}


def _call(name: str, params: dict) -> str:
    import json
    return "Tarıyorum.\n```arac\n" + json.dumps({"arac": name, "parametreler": params}) + "\n```"


def test_fold_handles_turkish_letters_and_case():
    assert fold("İSTANBUL'da Şifre Çözüm ığüö") == "istanbul'da sifre cozum iguo"


def test_jaccard_on_folded_tokens():
    assert jaccard("Açık portları tara", "acik portlari tara") == 1.0
    assert jaccard("a b", "c d") == 0.0


def test_correct_call_has_no_failures():
    assert classify_failures(NMAP_CASE, _call("nmap", {"hedef": "10.0.0.5"})) == []


def test_invented_tool_is_catalog_and_wrong_tool():
    labels = classify_failures(NMAP_CASE, _call("portscan", {"hedef": "10.0.0.5"}))
    assert "catalog" in labels and "wrong_first_tool" in labels


def test_wrong_tool_and_missing_call_are_distinguished():
    assert classify_failures(NMAP_CASE, _call("masscan", {"hedef": "10.0.0.5"})) == ["wrong_first_tool"]
    assert classify_failures(NMAP_CASE, "Sadece açıklama yazdım.") == ["missing_call"]


def test_parameter_mismatch_only_after_right_tool():
    assert classify_failures(NMAP_CASE, _call("nmap", {"hedef": "10.0.0.9"})) == ["param_mismatch"]


def test_parameter_formatting_differences_are_not_failures():
    case = {"expected_tools": ["gobuster"], "expected_params": {"url": "http://app.lab"}, "allow_tools": True}
    assert classify_failures(case, _call("gobuster", {"url": "http://app.lab/"})) == []
    assert classify_failures(case, _call("gobuster", {"url": " HTTP://APP.LAB "})) == []
    assert classify_failures(case, _call("gobuster", {"url": "http://other.lab"})) == ["param_mismatch"]


def test_target_keys_are_interchangeable_like_the_runtime():
    case = {"expected_tools": ["gobuster"], "expected_params": {"url": "http://app.lab"}, "allow_tools": True}
    assert classify_failures(case, _call("gobuster", {"hedef": "http://app.lab"})) == []
    assert classify_failures(case, _call("gobuster", {"hedef": "http://x.lab"})) == ["param_mismatch"]
    path_case = {"expected_tools": ["volatility3"], "expected_params": {"dosya": "/tmp/a/ram.raw"},
                 "allow_tools": True}
    assert classify_failures(path_case, _call("volatility3", {"dosya": "ram.raw"})) == ["param_mismatch"]


def test_unnecessary_call_on_knowledge_question():
    assert classify_failures(KNOWLEDGE_CASE, _call("nmap", {"hedef": "10.0.0.5"})) == ["unnecessary_call"]
    assert classify_failures(KNOWLEDGE_CASE, "SYN taraması yarı açıktır.") == []


def test_malformed_block_counts_as_parse_failure():
    assert "parse" in classify_failures(NMAP_CASE, "```arac\n{bozuk\n```")


def test_prose_outside_blocks_ignores_dusunce_and_arac():
    text = "```dusunce\nplan\n```\nÖnce bakalım.\n" + _call("nmap", {"hedef": "x"})
    assert prose_outside_blocks(text) == "Önce bakalım.\nTarıyorum."


def test_catalog_excerpt_lists_names_with_params():
    text = catalog_excerpt(["nmap"])
    assert "nmap(hedef, secenekler)" in text and "sqlmap" not in text
    assert "nmap(" in catalog_excerpt()
