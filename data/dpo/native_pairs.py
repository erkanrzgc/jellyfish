"""On-policy preference pairs for stock gemma4 in its native tool-call format.

The v1.0 pairs (`data.dpo.build`) pair teacher answers with v1.0's failures; gemma4 does not make most of
those mistakes. Here `rejected` is gemma4's own failing reply and `chosen` is a teacher answer converted from
```arac``` text to native `tool_calls`.

    uv run python -m eval.native_tool_bakeoff --model gemma4:e4b --split train   # gemma4 replies (local)
    uv run python -m data.dpo.native_pairs gaps                                     # failures without chosen
    uv run python -m data.dpo.chosen --rejected data/dpo/out/native_train/gemma4_e4b_gaps.jsonl \\
        --out data/dpo/out/native_train/chosen_extra.jsonl                          # EVREN teacher
    uv run python -m data.dpo.native_pairs build
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from agent.backends.native_chat import NATIVE_SYSTEM_PROMPT, native_tools, to_arac_reply
from agent.toolcall import parse_tool_reply
from data.dpo.build import repair_dusunce, split_of
from data.dpo.common import OUT_DIR, classify_failures, prose_outside_blocks, read_jsonl

MODEL = "gemma4:e4b"
NATIVE_TRAIN_DIR = OUT_DIR / "native_train"
EXTRA_CHOSEN = NATIVE_TRAIN_DIR / "chosen_extra.jsonl"
PAIRS_DIR = OUT_DIR / "native_pairs"


def _stem(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def to_native_message(text: str) -> dict:
    """Teacher ```arac``` text -> assistant message with native tool_calls; reasoning fences are dropped."""
    text = repair_dusunce(text)
    message = {"role": "assistant", "content": prose_outside_blocks(text)}
    calls = parse_tool_reply(text).calls
    if calls:
        message["tool_calls"] = [{"type": "function", "function": {"name": call.name, "arguments": dict(call.params)}}
                                 for call in calls]
    return message


def normalize_message(message: dict) -> dict:
    """Ollama reply -> the same message shape as `to_native_message`; invalid arguments stay as returned."""
    normalized = {"role": "assistant", "content": (message.get("content") or "").strip()}
    calls = [{"type": "function", "function": {"name": call.get("function", {}).get("name"),
                                               "arguments": call.get("function", {}).get("arguments", {})}}
             for call in message.get("tool_calls") or []]
    if calls:
        normalized["tool_calls"] = calls
    return normalized


def failed_rows(cases: dict, replies: dict[str, dict]) -> list[dict]:
    """The model's failing replies in `data.dpo.chosen`'s rejected-file shape (id, raw_reply)."""
    rows = []
    for case_id in sorted(replies):
        if case_id not in cases:
            continue
        text = to_arac_reply(replies[case_id])
        failures = classify_failures(cases[case_id], text)
        if failures:
            rows.append({"id": case_id, "raw_reply": text, "failures": failures})
    return rows


def gap_rows(failed: list[dict], chosen: dict[str, dict]) -> list[dict]:
    return [row for row in failed if not (chosen.get(row["id"]) or {}).get("chosen")]


def load_chosen(*paths: Path) -> dict[str, dict]:
    """Later files win, but only with an accepted answer: a dropped retry must not hide an earlier answer."""
    merged: dict[str, dict] = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("chosen") or row["id"] not in merged:
                merged[row["id"]] = row
    return merged


def load_replies(model: str = MODEL) -> dict[str, dict]:
    """Val replies come from the bake-off run, train replies from the `--split train` run."""
    replies = {}
    for path in (OUT_DIR / "bakeoff" / f"{_stem(model)}.jsonl", NATIVE_TRAIN_DIR / f"{_stem(model)}.jsonl"):
        replies.update({row["id"]: row["message"] for row in read_jsonl(path)})
    return replies


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build(cases: dict, replies: dict[str, dict], chosen: dict[str, dict], out_dir: Path, *,
          model: str = MODEL) -> dict:
    pairs: dict[str, list[dict]] = {"train": [], "val": []}
    by_failure, by_category = Counter(), Counter()
    missing = unconvertible = 0
    failed = failed_rows(cases, replies)
    for row in failed:
        case = cases[row["id"]]
        answer = (chosen.get(row["id"]) or {}).get("chosen")
        if not answer:
            missing += 1
            continue
        good = to_native_message(answer)
        if classify_failures(case, to_arac_reply(good)):
            unconvertible += 1
            continue
        meta = {"id": case["id"], "family_id": case["family_id"], "category": case["category"],
                "failures": row["failures"], "teacher_model": chosen[row["id"]].get("teacher_model"),
                "policy_model": model}
        pairs[split_of(case["family_id"])].append({
            "prompt": [{"role": "system", "content": NATIVE_SYSTEM_PROMPT}, {"role": "user", "content": case["prompt"]}],
            "chosen": [good], "rejected": [normalize_message(replies[row["id"]])], "meta": meta})
        by_failure.update(row["failures"])
        by_category[case["category"]] += 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "pairs_train.jsonl", pairs["train"])
    _write_jsonl(out_dir / "pairs_val.jsonl", pairs["val"])
    tools = native_tools()
    tools_text = json.dumps(tools, ensure_ascii=False, indent=1)
    (out_dir / "tools.json").write_text(tools_text, encoding="utf-8")
    scored = sum(case_id in cases for case_id in replies)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": model,
        "scored": scored, "model_passed": scored - len(failed),
        "pairs": len(pairs["train"]) + len(pairs["val"]), "train": len(pairs["train"]), "val": len(pairs["val"]),
        "missing_chosen": missing, "unconvertible_chosen": unconvertible,
        "tools": len(tools), "tools_sha256": hashlib.sha256(tools_text.encode("utf-8")).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(NATIVE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "by_failure": dict(sorted(by_failure.items())), "by_category": dict(sorted(by_category.items())),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["gaps", "build"])
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)
    cases = {case["id"]: case for case in read_jsonl(OUT_DIR / "cases.jsonl")}
    replies = load_replies(args.model)
    chosen = load_chosen(OUT_DIR / "chosen.jsonl", EXTRA_CHOSEN)
    if args.stage == "gaps":
        failed = failed_rows(cases, replies)
        gaps = gap_rows(failed, chosen)
        out = NATIVE_TRAIN_DIR / f"{_stem(args.model)}_gaps.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out, gaps)
        print(json.dumps({"replies": len(replies), "failed": len(failed), "have_chosen": len(failed) - len(gaps),
                          "gaps": len(gaps), "gaps_file": str(out)}, ensure_ascii=False))
        return 0
    manifest = build(cases, replies, chosen, PAIRS_DIR, model=args.model)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
