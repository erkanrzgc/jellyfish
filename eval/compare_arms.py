"""Paired comparison of policy reply files on the same cases: score per arm and case flips between arms.

    uv run python -m eval.compare_arms gemma4_e4b jellyfish-g4-requant jellyfish-g4-dpo
    uv run python -m eval.compare_arms gemma4_e4b gemma4_12b --cases data/dpo/out/cases_tool.jsonl

An arm is a reply-file stem under data/dpo/out/native_train/ (`<stem>[_<tag>]_<split>.jsonl`). Only cases every
arm has answered are compared, so a partial run is compared on its finished cases. One single run of the same
weights moved ±9 cases (2026-11-25), so treat smaller net differences as noise.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from agent.backends.native_chat import to_arac_reply
from data.dpo.common import OUT_DIR, case_tag, classify_failures, read_jsonl
from eval.native_tool_bakeoff import NATIVE_TRAIN_DIR

NOISE_CASES = 9


def load_arm(arm: str, *, tag: str = "", split: str = "val", root: Path = NATIVE_TRAIN_DIR) -> dict[str, str]:
    """case id -> reply as ```arac``` text (the form `classify_failures` scores)."""
    stem = f"{arm}_{tag}" if tag else arm
    rows = read_jsonl(root / f"{stem}_{split}.jsonl")
    return {row["id"]: to_arac_reply(row["message"]) if "message" in row else row["reply"] for row in rows}


def compare(cases: dict[str, dict], arms: dict[str, dict[str, str]]) -> dict:
    """Score every arm on the cases all arms share, then count pairwise gains and losses."""
    answered = [set(replies) for replies in arms.values() if replies]
    ids = sorted(set(cases).intersection(*answered)) if answered else []
    labels = {arm: {i: classify_failures(cases[i], replies[i]) for i in ids} for arm, replies in arms.items() if replies}
    tool_ids = [i for i in ids if cases[i].get("allow_tools")]
    report = {"cases": len(ids), "arms": {}, "pairs": []}
    for arm, found in labels.items():
        report["arms"][arm] = {
            "passed": sum(not found[i] for i in ids),
            "tool_passed": sum(not found[i] for i in tool_ids), "tool_cases": len(tool_ids),
            "no_tool_passed": sum(not found[i] for i in ids if i not in tool_ids), "no_tool_cases": len(ids) - len(tool_ids),
            "labels": dict(collections.Counter(label for i in ids for label in found[i]).most_common()),
        }
    names = list(labels)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            gained = [i for i in ids if not labels[second][i] and labels[first][i]]
            lost = [i for i in ids if labels[second][i] and not labels[first][i]]
            report["pairs"].append({"from": first, "to": second, "gained": len(gained), "lost": len(lost),
                                    "net": len(gained) - len(lost),
                                    "identical_replies": sum(arms[first][i] == arms[second][i] for i in ids),
                                    "gained_ids": gained, "lost_ids": lost})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="reply-file stems, e.g. gemma4_e4b jellyfish-g4-dpo")
    parser.add_argument("--cases", type=Path, default=OUT_DIR / "cases.jsonl")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    args = parser.parse_args(argv)
    cases = {case["id"]: case for case in read_jsonl(args.cases)}
    tag = case_tag(args.cases)
    report = compare(cases, {arm: load_arm(arm, tag=tag, split=args.split) for arm in args.arms})
    print(json.dumps({**report, "noise_cases": NOISE_CASES}, ensure_ascii=False, indent=1))
    return 0 if report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
