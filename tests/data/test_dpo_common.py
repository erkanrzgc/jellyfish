"""Shared DPO helpers: Turkish-aware normalization, failure labels, catalog text."""
from __future__ import annotations

from agent.catalog import CATALOG
from data.dpo.common import (SUBSTITUTE_GROUPS, acceptable_first_tools, catalog_excerpt, classify_failures,
                             fold, jaccard, prose_outside_blocks)

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
    assert classify_failures(NMAP_CASE, _call("nikto", {"hedef": "10.0.0.5"})) == ["wrong_first_tool"]
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


def test_substitute_groups_use_catalog_names_only():
    assert all(name in CATALOG for group in SUBSTITUTE_GROUPS for name in group)
    assert acceptable_first_tools(["rustscan"]) >= {"nmap", "rustscan", "masscan"}
    assert "fping" not in acceptable_first_tools(["rustscan"])  # groups do not chain through nmap


def test_functionally_equivalent_first_tool_is_not_a_failure():
    """A pair that prefers rustscan over nmap would teach a spurious tool preference."""
    rustscan = {"expected_tools": ["rustscan"], "expected_params": {"hedef": "10.0.1.5"}, "allow_tools": True}
    assert classify_failures(rustscan, _call("nmap", {"hedef": "10.0.1.5", "secenekler": "-p-"})) == []
    assert classify_failures(rustscan, _call("nmap", {"hedef": "10.0.1.9"})) == ["param_mismatch"]
    ffuf = {"expected_tools": ["ffuf"], "expected_params": {"url": "http://example.com"}, "allow_tools": True}
    assert classify_failures(ffuf, _call("gobuster", {"hedef": "http://example.com"})) == []
    assert classify_failures(ffuf, _call("sqlmap", {"url": "http://example.com"})) == ["wrong_first_tool"]


def test_substitute_without_a_target_parameter_is_checked_on_its_option_text():
    case = {"expected_tools": ["dig"], "expected_params": {"domain": "app.local"}, "allow_tools": True}
    assert classify_failures(case, _call("host", {"secenekler": "-t A app.local"})) == []
    assert classify_failures(case, _call("host", {"secenekler": "-t A other.local"})) == ["param_mismatch"]


def test_free_form_options_only_need_the_same_targets():
    """'secenekler' is free text: flags may differ, the target or file it names may not."""
    fping = {"expected_tools": ["fping"], "expected_params": {"secenekler": "-g 192.168.1.0/24"}, "allow_tools": True}
    assert classify_failures(fping, _call("nmap", {"hedef": "192.168.1.0/24", "secenekler": "-sn"})) == []
    assert classify_failures(fping, _call("fping", {"secenekler": "-a -g 192.168.1.0/24"})) == []
    assert classify_failures(fping, _call("fping", {"secenekler": "-g 10.0.0.0/24"})) == ["param_mismatch"]
    flags = {"expected_tools": ["nmap"], "expected_params": {"hedef": "10.10.2.2", "secenekler": "-sV -p 22"},
             "allow_tools": True}
    assert classify_failures(flags, _call("nmap", {"hedef": "10.10.2.2", "secenekler": "-p22 -sV"})) == []


GREP_CASE = {"expected_tools": ["grep"], "expected_params": {"desen": "Failed password", "yol": "/var/log/auth.log"},
             "allow_tools": True}
SS_CASE = {"expected_tools": ["run_cmd"], "expected_params": {"komut": "ss -tulpn"}, "allow_tools": True}


def test_local_inspection_through_run_cmd_is_the_same_job():
    """`run_cmd "grep …"` and the grep tool do the same local work; preferring one teaches a quirk."""
    good = _call("run_cmd", {"komut": "grep -i 'Failed password' /var/log/auth.log"})
    assert classify_failures(GREP_CASE, good) == []
    other_file = _call("run_cmd", {"komut": "grep -i 'Failed password' /var/log/syslog"})
    assert classify_failures(GREP_CASE, other_file) == ["param_mismatch"]
    assert classify_failures(SS_CASE, _call("ss", {"secenekler": "-tulpn"})) == []
    cat_case = {"expected_tools": ["read_file"], "expected_params": {"yol": "/etc/ssh/sshd_config"}, "allow_tools": True}
    assert classify_failures(cat_case, _call("run_cmd", {"komut": "cat /etc/ssh/sshd_config"})) == []


def test_run_cmd_never_substitutes_for_network_or_offensive_tools():
    """Catalog tools pass the lab scope policy and the argument allowlist; run_cmd bypasses both."""
    nmap_via_shell = _call("run_cmd", {"komut": "nmap -sV 10.0.0.5"})
    assert classify_failures(NMAP_CASE, nmap_via_shell) == ["wrong_first_tool"]
    sqlmap = {"expected_tools": ["sqlmap"], "expected_params": {"url": "http://app.lab/item.php?id=1"},
              "allow_tools": True}
    assert classify_failures(sqlmap, _call("run_cmd", {"komut": "sqlmap -u http://app.lab/item.php?id=1"})) == [
        "wrong_first_tool"]
    shell_nmap_case = {"expected_tools": ["run_cmd"], "expected_params": {"komut": "nmap -sV 10.0.0.5"},
                       "allow_tools": True}
    assert classify_failures(shell_nmap_case, _call("nmap", {"hedef": "10.0.0.5"})) == ["wrong_first_tool"]


def test_unrelated_local_tool_is_still_wrong():
    assert classify_failures(SS_CASE, _call("grep", {"desen": "LISTEN", "yol": "/proc/net/tcp"})) == [
        "wrong_first_tool"]
    assert classify_failures(GREP_CASE, _call("run_cmd", {"komut": "tail -n 50 /var/log/auth.log"})) == [
        "wrong_first_tool"]


def test_smb_enumeration_tools_are_substitutes():
    case = {"expected_tools": ["enum4linux-ng"], "expected_params": {"hedef": "192.168.50.10"}, "allow_tools": True}
    assert classify_failures(case, _call("smbmap", {"secenekler": "-H 192.168.50.10"})) == []
    assert "smbclient" not in acceptable_first_tools(["enum4linux-ng"])


def test_file_and_socket_listing_commands_map_to_their_catalog_tools():
    magika = {"expected_tools": ["magika"], "expected_params": {"yol": "/tmp/unknown_042"}, "allow_tools": True}
    assert classify_failures(magika, _call("run_cmd", {"komut": "file /tmp/unknown_042"})) == []
    assert classify_failures(magika, _call("run_cmd", {"komut": "file /tmp/other"})) == ["param_mismatch"]
    netstat = {"expected_tools": ["run_cmd"], "expected_params": {"komut": "netstat -tunlp"}, "allow_tools": True}
    assert classify_failures(netstat, _call("ss", {"secenekler": "-tulnp"})) == []
    lsof = {"expected_tools": ["run_cmd"], "expected_params": {"komut": "lsof -i :22"}, "allow_tools": True}
    assert classify_failures(lsof, _call("ss", {"secenekler": "-tlnp sport = :22"})) == []
    assert classify_failures(lsof, _call("ss", {"secenekler": "-tlnp"})) == ["param_mismatch"]
