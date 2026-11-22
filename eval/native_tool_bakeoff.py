"""Base-model bake-off: stock models with native tool calling vs v1.0, scored with the SAME labels.

Stock models get every catalog tool as an Ollama native tool (/api/chat `tools`). Their tool_calls are
converted to ```arac``` text, so `data.dpo.common.classify_failures` scores every arm identically.
v1.0 arms are read from stored replies (rejected.jsonl, repair_replay.jsonl) on the same val-family ids.

    uv run python -m eval.native_tool_bakeoff --model qwen3:8b       # resumable
    uv run python -m eval.native_tool_bakeoff --summary
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import time
import urllib.request
from pathlib import Path

from agent.backends.native_chat import NATIVE_SYSTEM_PROMPT, native_tools, to_arac_reply
from agent.constrained import repair_prefix
from data.dpo.build import split_of
from data.dpo.common import OUT_DIR, append_jsonl, classify_failures, read_jsonl

log = logging.getLogger(__name__)
BAKEOFF_DIR = OUT_DIR / "bakeoff"
def bakeoff_ids(cases: list[dict]) -> list[str]:
    """The DPO val families (held out from preference training), so this table is also the pre-DPO baseline."""
    return sorted(case["id"] for case in cases if split_of(case["family_id"]) == "val")


def chat(host: str, model: str, prompt: str, *, num_ctx: int, timeout: int = 600) -> dict:
    body = {"model": model, "stream": False, "think": False, "tools": native_tools(),
            "messages": [{"role": "system", "content": NATIVE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": 3407, "num_ctx": num_ctx, "num_predict": 1024}}
    request = urllib.request.Request(host.rstrip("/") + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_model(model: str, cases: dict, ids: list[str], host: str, num_ctx: int) -> dict:
    out = BAKEOFF_DIR / f"{model.replace(':', '_').replace('/', '_')}.jsonl"
    done = {row["id"] for row in read_jsonl(out)}
    stats = collections.Counter()
    for case_id in ids:
        if case_id in done:
            continue
        started = time.perf_counter()
        try:
            data = chat(host, model, cases[case_id]["prompt"], num_ctx=num_ctx)
        except Exception as error:  # noqa: BLE001 -- one bad case must not stop the run
            log.warning("%s: %s", case_id, error)
            stats["error"] += 1
            continue
        message = data.get("message", {})
        append_jsonl(out, {"id": case_id, "model": model, "reply": to_arac_reply(message), "message": message,
                           "prompt_eval_count": data.get("prompt_eval_count"), "done_reason": data.get("done_reason"),
                           "elapsed_seconds": time.perf_counter() - started})
        stats["done"] += 1
        if stats["done"] % 25 == 0:
            log.info("%s %s", model, dict(stats))
    return dict(stats)


def run_v10(cases: dict, ids: list[str], host: str) -> dict:
    """v1.0 on the SAME Ollama build as the stock arms: free reply plus the repair from the same call."""
    from agent.messages import Message
    from agent.profiles import create_model, get_profile
    profile = get_profile("octopus-v10")
    model = create_model(profile.name, tokenizer_dir=profile.tokenizer_dir, modelfile=profile.modelfile,
                         temperature=0, num_ctx=4096, num_predict=1024, seed=3407, host=host,
                         constrain_calls=True, repair_unknown_names=True)
    out = BAKEOFF_DIR / "octopus-v10.jsonl"
    done = {row["id"] for row in read_jsonl(out)}
    stats = collections.Counter()
    for case_id in ids:
        if case_id in done:
            continue
        result = model.generate_result([Message("user", cases[case_id]["prompt"])])
        if result.error:
            log.warning("%s: %s", case_id, result.error)
            stats["error"] += 1
            continue
        repair = result.metadata.get("call_repair") or {}
        append_jsonl(out, {"id": case_id, "model": "octopus-v10", "reply": repair.get("original", result.text),
                           "reply_repair_any": result.text, "repair_applied": bool(repair.get("applied")),
                           "truncated": result.truncated})
        stats["done"] += 1
    return dict(stats)


def _v10_arms(rows: list[dict]) -> dict[str, dict[str, str]]:
    free = {r["id"]: r["reply"] for r in rows}
    catalog = {r["id"]: r["reply_repair_any"] if repair_prefix(r["reply"]) is not None else r["reply"] for r in rows}
    return {"octopus-v10": free, "octopus-v10+repair": catalog,
            "octopus-v10+repair-any-name": {r["id"]: r["reply_repair_any"] for r in rows}}


def _score(cases: dict, replies: dict[str, str], ids: list[str]) -> dict:
    labels, failed = collections.Counter(), 0
    scored = [case_id for case_id in ids if case_id in replies]
    for case_id in scored:
        found = classify_failures(cases[case_id], replies[case_id])
        labels.update(found)
        failed += bool(found)
    tool_ids = [i for i in scored if cases[i].get("allow_tools")]
    no_tool_ids = [i for i in scored if not cases[i].get("allow_tools")]
    def passed(group):
        return sum(not classify_failures(cases[i], replies[i]) for i in group)
    return {"scored": len(scored), "passed": len(scored) - failed, "tool_cases_passed": f"{passed(tool_ids)}/{len(tool_ids)}",
            "no_tool_cases_passed": f"{passed(no_tool_ids)}/{len(no_tool_ids)}", "labels": dict(labels.most_common())}


def summary(cases: dict, ids: list[str]) -> dict:
    baseline = {row["id"]: row["raw_reply"] for row in read_jsonl(OUT_DIR / "rejected.jsonl")}
    fixes = {row["id"]: row for row in read_jsonl(OUT_DIR / "repair_replay.jsonl")}
    repaired = {}
    for case_id, reply in baseline.items():
        fix = fixes.get(case_id)
        use = fix and fix["applied"] and repair_prefix(reply) is not None
        repaired[case_id] = fix["repaired_reply"] if use else reply
    report = {"sample_ids": len(ids), "v10@ollama-0.30 (stored)": _score(cases, baseline, ids),
              "v10+repair@ollama-0.30 (stored, partial)": _score(cases, repaired, ids)}
    for path in sorted(BAKEOFF_DIR.glob("*.jsonl")):
        rows = read_jsonl(path)
        if rows and "reply_repair_any" in rows[0]:
            report.update({arm: _score(cases, replies, ids) for arm, replies in _v10_arms(rows).items()})
        else:
            report[rows[0]["model"] if rows else path.stem] = _score(cases, {r["id"]: r["reply"] for r in rows}, ids)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--v10", action="store_true", help="run octopus-v10 (+repair) on this Ollama build")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cases = {case["id"]: case for case in read_jsonl(OUT_DIR / "cases.jsonl")}
    ids = bakeoff_ids(list(cases.values()))
    if not args.summary:
        if args.v10:
            print(json.dumps(run_v10(cases, ids, args.host), ensure_ascii=False))
        for model in args.model:
            print(json.dumps(run_model(model, cases, ids, args.host, args.num_ctx), ensure_ascii=False))
    print(json.dumps(summary(cases, ids), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
