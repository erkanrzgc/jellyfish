"""nmap secenek dogrulayicisi (agent/arguments.py) ve nmap skill'i ile tutarliligi."""
import re
from pathlib import Path

import pytest

from agent.arguments import nmap_flags

_NMAP_SKILL = Path("agent/skills/tools/nmap.md")


def test_all_ports_shorthand_allowed():
    # -p- = 1-65535; zararsiz, yalniz port araligi.
    assert nmap_flags("-sV -Pn -p-") == ["-sV", "-Pn", "-p-"]


@pytest.mark.parametrize("options", ["-sC", "--script vuln", "-A", "-O", "-oN out.txt", "-p-1"])
def test_unverified_options_still_rejected(options):
    with pytest.raises(ValueError):
        nmap_flags(options)


def test_skill_typical_example_passes_policy():
    """Skill'in onerdigi tipik komut policy'den gecmeli; yoksa model skill'e uyup reddedilir."""
    text = _NMAP_SKILL.read_text(encoding="utf-8")
    line = next(row for row in text.splitlines() if row.startswith("Tipik:"))
    example = re.search(r"`([^`]+)`", line).group(1)
    nmap_flags(example)


def test_skill_does_not_recommend_rejected_flags():
    text = _NMAP_SKILL.read_text(encoding="utf-8")
    flags_section = text.split("## Ana flag'ler", 1)[1].split("##", 1)[0]
    for token in re.findall(r"`([^`]+)`", flags_section):
        nmap_flags(token)
