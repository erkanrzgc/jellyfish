"""Shared helpers: Turkish-aware text matching, structural failure labels, JSONL I/O."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from agent.catalog import CATALOG, TARGET_KEYS
from agent.toolcall import parse_tool_reply, strip_dusunce
from eval.reliability_eval import structural_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "dpo" / "out"
TURKISH_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")
_FOLD = str.maketrans("çğıöşüâîûÇĞÖŞÜÂÎÛ", "cgiosuaiuCGOSUAIU")
_WORD = re.compile(r"\w+")
_ARAC_BLOCK = re.compile(r"```arac\b.*?```", re.S | re.I)


def fold(text: str) -> str:
    """Lowercase and strip Turkish diacritics so 'Açık' and 'acik' compare equal."""
    return text.replace("İ", "i").lower().translate(_FOLD)


def tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(fold(text)))


def token_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard(a: str, b: str) -> float:
    return token_jaccard(tokens(a), tokens(b))


def normalized_key(text: str) -> str:
    return " ".join(_WORD.findall(fold(text)))


def prose_outside_blocks(text: str) -> str:
    return _ARAC_BLOCK.sub("", strip_dusunce(text)).strip()


def _loose(value: object) -> str:
    return str(value).strip().rstrip("/").lower()


def _param_matches(params: dict, key: str, value: object) -> bool:
    # The runtime reads hedef/url/hedef_url/domain as one network target (agent.catalog.target_value).
    keys = TARGET_KEYS if key in TARGET_KEYS else (key,)
    return any(_loose(params[k]) == _loose(value) for k in keys if k in params)


# Tools that do the same job: picking one over another is a preference, not an error. Groups are tight
# (one job each) and never chain, so a pair can only teach "right job", not "nmap is wrong".
SUBSTITUTE_GROUPS: tuple[frozenset[str], ...] = tuple(frozenset(group) for group in (
    ("nmap", "rustscan", "masscan"),                                   # port scan
    ("nmap", "netdiscover", "arp-scan", "fping"),                      # live-host discovery
    ("gobuster", "ffuf", "feroxbuster", "dirsearch", "dirb", "wfuzz"),  # content discovery
    ("tcpdump", "tshark"),                                             # packet capture
    ("amass", "subfinder"),                                            # subdomain enumeration
    ("dig", "host"),                                                   # DNS lookup
    ("hydra", "medusa"),                                               # online brute force
    ("john", "hashcat"),                                               # offline cracking
    ("iptables", "nftables", "ufw"),                                   # host firewall
    ("suricata", "snort"),                                             # network IDS
    ("rkhunter", "chkrootkit"),                                        # rootkit scan
    ("ghidra", "radare2"),                                             # reverse engineering
    ("dalfox", "xsstrike"),                                            # XSS scanning
    ("bettercap", "ettercap"),                                         # MITM
    ("apktool", "jadx"),                                               # APK decompiling
))


def acceptable_first_tools(expected: Iterable[str]) -> frozenset[str]:
    """Expected tools plus their direct substitutes (one hop, no chaining)."""
    expected = frozenset(expected)
    return expected.union(*(group for group in SUBSTITUTE_GROUPS if group & expected))


def _option_targets(value: object) -> list[str]:
    """Targets, paths and URLs inside free-form options; flags are left to the model's choice."""
    return [_loose(token) for token in str(value).split() if any(mark in token for mark in "./:")]


def _params_match_loosely(case: dict, text: str) -> bool:
    """Ignore case, spaces, a trailing slash and target-key choice, so formatting is not a failure.
    A substitute tool without the expected key must carry the value in its other (option) text."""
    calls = parse_tool_reply(text).calls
    if not calls:
        return False
    params = calls[0].params
    spec = CATALOG.get(calls[0].name)
    spec_keys = set(spec.params) if spec else set(params)
    option_text = " ".join(str(value).lower() for value in params.values())
    for key, value in case["expected_params"].items():
        keys = TARGET_KEYS if key in TARGET_KEYS else (key,)
        if key == "secenekler":
            if not all(token in option_text for token in _option_targets(value)):
                return False
        elif spec_keys.intersection(keys):
            if not _param_matches(params, key, value):
                return False
        elif _loose(value) not in option_text:
            return False
    return True


def classify_failures(case: dict, text: str) -> list[str]:
    """Structural failure labels for one reply; an empty list means every check passed."""
    _, parse_errors, checks = structural_metrics(case, text)
    labels = []
    if parse_errors:
        labels.append("parse")
    if not checks["catalog_valid"]:
        labels.append("catalog")
    if checks["no_unnecessary_tools"] is False:
        labels.append("unnecessary_call")
    if checks["expected_first_tool_match"] is False:
        calls = parse_tool_reply(text).calls
        if checks["call_count"] == 0:
            labels.append("missing_call")
        elif not calls or calls[0].name not in acceptable_first_tools(case["expected_tools"]):
            labels.append("wrong_first_tool")
        elif not _params_match_loosely(case, text):
            labels.append("param_mismatch")
    elif checks["expected_parameter_subset_match"] is False and not _params_match_loosely(case, text):
        labels.append("param_mismatch")
    return labels


def catalog_excerpt(names: Iterable[str] | None = None) -> str:
    specs = list(CATALOG.values()) if names is None else [CATALOG[n] for n in names if n in CATALOG]
    return "\n".join(f"- {spec.name}({', '.join(spec.params)}) [{spec.domain}]" for spec in specs)


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
