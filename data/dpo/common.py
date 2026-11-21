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


def _params_match_loosely(case: dict, text: str) -> bool:
    """Ignore case, spaces, a trailing slash and target-key choice, so formatting is not a failure."""
    calls = parse_tool_reply(text).calls
    if not calls:
        return False
    params = calls[0].params
    return all(_param_matches(params, key, value) for key, value in case["expected_params"].items())


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
        labels.append("missing_call" if checks["call_count"] == 0 else "wrong_first_tool")
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
