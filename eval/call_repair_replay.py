"""Measure constrained call repair (agent.constrained) on stored v1.0 replies, with no free regeneration.

At temperature 0 with a fixed seed, `data/dpo/out/rejected.jsonl` raw replies ARE the free first pass, so
only invalid replies need one constrained call each. Each repair is stored once (unknown names allowed,
the superset); the default arm (catalog names only) is the subset whose prefix it would also repair.
Run it on the same Ollama build that produced the replies.

    uv run python -m eval.call_repair_replay            # resumable
    uv run python -m eval.call_repair_replay --summary  # scores only
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

from agent.backends.gguf_model import GenerationResult
from agent.constrained import repair_prefix
from agent.messages import Message
from data.dpo.common import OUT_DIR, append_jsonl, classify_failures, read_jsonl

log = logging.getLogger(__name__)
DEFAULT_OUT = OUT_DIR / "repair_replay.jsonl"


def replay(model, cases: dict, rows: list[dict], out_path: Path) -> dict:
    done = {row["id"] for row in read_jsonl(out_path)}
    model.repair_unknown_names = True
    stats = collections.Counter()
    for row in rows:
        if row["id"] in done or repair_prefix(row["raw_reply"], allow_unknown_names=True) is None:
            continue
        prompt = model._render(model_messages(model, cases[row["id"]]))
        result = model._repair_calls(prompt, GenerationResult(text=row["raw_reply"], done_reason="stop"))
        info = result.metadata.get("call_repair", {})
        append_jsonl(out_path, {"id": row["id"], "repaired_reply": result.text,
                                "applied": bool(info.get("applied")), "repair_error": info.get("repair_error"),
                                "elapsed_seconds": info.get("elapsed_seconds")})
        stats["repaired" if info.get("applied") else "kept"] += 1
        if sum(stats.values()) % 25 == 0:
            log.info("%s", dict(stats))
    return dict(stats)


def model_messages(model, case: dict) -> list[dict]:
    from agent.toolcall import render_for_model
    from data.sft.normalize import ensure_system
    return ensure_system(render_for_model([Message("user", case["prompt"])]), model.system_prompt)


def _arm_reply(row: dict, repaired: dict, *, allow_unknown: bool) -> str:
    fix = repaired.get(row["id"])
    if fix and fix["applied"] and repair_prefix(row["raw_reply"], allow_unknown_names=allow_unknown) is not None:
        return fix["repaired_reply"]
    return row["raw_reply"]


def summarize(cases: dict, rows: list[dict], repaired: dict) -> dict:
    arms = {"baseline": None, "repair_catalog_names": False, "repair_any_name": True}
    report = {}
    for arm, allow in arms.items():
        labels, failed, no_tool_calls = collections.Counter(), 0, 0
        for row in rows:
            case = cases[row["id"]]
            reply = row["raw_reply"] if allow is None else _arm_reply(row, repaired, allow_unknown=allow)
            found = classify_failures(case, reply)
            labels.update(found)
            failed += bool(found)
            if not case.get("allow_tools") and "```arac" in reply and reply != row["raw_reply"]:
                no_tool_calls += 1
        report[arm] = {"rows": len(rows), "failed": failed, "labels": dict(labels.most_common()),
                       "changed_no_tool_replies": no_tool_calls}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=OUT_DIR / "cases.jsonl")
    parser.add_argument("--rejected", type=Path, default=OUT_DIR / "rejected.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cases = {case["id"]: case for case in read_jsonl(args.cases)}
    rows = [row for row in read_jsonl(args.rejected) if row["id"] in cases]
    if not args.summary:
        from agent.profiles import create_model, get_profile
        profile = get_profile("octopus-v10")
        model = create_model(profile.name, tokenizer_dir=profile.tokenizer_dir, modelfile=profile.modelfile,
                             temperature=0, num_ctx=4096, num_predict=1024, seed=3407, host=args.host,
                             constrain_calls=True)
        print(json.dumps(replay(model, cases, rows, args.out), ensure_ascii=False))
    repaired = {row["id"]: row for row in read_jsonl(args.out)}
    print(json.dumps(summarize(cases, rows, repaired), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
