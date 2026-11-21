"""Offline integrity checks for the open development corpus (no live tools)."""
import json
from collections import Counter
from pathlib import Path
import unicodedata
from urllib.parse import urlsplit

import pytest

from agent.catalog import CATALOG


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "eval/suites/reliability-dev-v1.json"
CATEGORIES = {"fundamentals", "tooling", "pentest", "forensics", "incident", "authorization"}
FIELDS = {"id", "family_id", "category", "split", "prompt", "expected_tools",
          "expected_params", "allow_tools", "reference_answer", "rubric", "sources", "critical"}


@pytest.fixture(scope="module")
def suite():
    assert SUITE.is_file(), "The first 60 development cases have not been authored"
    return json.loads(SUITE.read_text(encoding="utf-8"))


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompts(value):
    """Recognize legacy prompt/soru records and SFT user messages recursively."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"prompt", "soru", "question"} and isinstance(item, str):
                yield item
            elif key == "content" and value.get("role") == "user" and isinstance(item, str):
                yield item
            else:
                yield from prompts(item)
    elif isinstance(value, list):
        for item in value:
            yield from prompts(item)


def test_dev_suite_identity_and_balance(suite):
    assert suite["schema_version"] == 1
    assert suite["suite_id"] == "reliability-dev-v1"
    assert suite["review_status"] == "source-backed-draft"
    assert len(suite["cases"]) == 60
    assert Counter(c["category"] for c in suite["cases"]) == Counter({k: 10 for k in CATEGORIES})
    assert Counter(c["category"] for c in suite["cases"] if c["critical"]) == Counter({k: 10 for k in CATEGORIES})


def test_cases_are_unique_complete_and_dev_only(suite):
    cases = suite["cases"]
    for field in ("id", "family_id", "prompt"):
        assert len({normalize(c[field]) for c in cases}) == len(cases), field
    for case in cases:
        assert set(case) == FIELDS, case["id"]
        assert case["split"] == "dev"
        assert type(case["allow_tools"]) is bool
        assert type(case["critical"]) is bool
        assert len(case["prompt"].strip()) >= 80
        assert len(case["reference_answer"].strip()) >= 40
        assert isinstance(case["rubric"], list) and len(case["rubric"]) >= 3
        assert all(isinstance(r, str) and len(r.strip()) >= 15 for r in case["rubric"])
        assert isinstance(case["sources"], list) and case["sources"]
        for source in case["sources"]:
            url = urlsplit(source)
            assert url.scheme == "https" and url.hostname and not url.username
            assert url.hostname in {"nmap.org", "www.rfc-editor.org", "man.openbsd.org",
                "www.wireshark.org", "docs.kernel.org", "www.freedesktop.org",
                "wstg.owasp.org", "cheatsheetseries.owasp.org", "yara.readthedocs.io",
                "www.gnu.org", "csrc.nist.gov", "nvlpubs.nist.gov", "bind9.readthedocs.io"}


def test_first_tool_and_parameter_contracts_match_catalog(suite):
    for case in suite["cases"]:
        assert isinstance(case["expected_tools"], list)
        assert isinstance(case["expected_params"], dict)
        assert len(case["expected_tools"]) == len(set(case["expected_tools"]))
        if not case["allow_tools"]:
            assert case["expected_tools"] == []
            assert case["expected_params"] == {}
        if case["expected_params"]:
            assert case["expected_tools"], case["id"]
        for tool in case["expected_tools"]:
            assert tool in CATALOG, (case["id"], tool)
            assert set(case["expected_params"]) <= set(CATALOG[tool].params), case["id"]


def test_no_exact_prompt_overlap_with_legacy_eval_or_sft(suite):
    existing = set()
    for directory in (ROOT / "eval/data", ROOT / "data/sft"):
        for path in directory.rglob("*"):
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    if line.strip():
                        existing.update(normalize(p) for p in prompts(json.loads(line)))
            elif path.suffix == ".json":
                existing.update(normalize(p) for p in prompts(json.loads(path.read_text(encoding="utf-8-sig"))))
            elif path.suffix == ".txt":
                existing.update(normalize(p) for p in path.read_text(encoding="utf-8-sig").splitlines()
                                if p.strip() and not p.lstrip().startswith("#"))
    assert existing, "Legacy contamination check must inspect existing prompts"
    assert not {normalize(c["prompt"]) for c in suite["cases"]} & existing
