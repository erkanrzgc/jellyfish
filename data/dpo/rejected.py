"""Stage 2: local v1.0 replies (on-policy) labelled with structural failures.

Needs a running Ollama with `octopus-v10`. Resumable: finished case ids are skipped.

    uv run python -m data.dpo.rejected
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agent.messages import Message
from data.dpo.common import OUT_DIR, append_jsonl, classify_failures, read_jsonl

log = logging.getLogger(__name__)
MAX_CONSECUTIVE_INFRA_ERRORS = 3


def run(model, cases_path: Path, out_path: Path) -> dict:
    done = {row["id"] for row in read_jsonl(out_path)}
    stats = {"done": 0, "failed": 0, "infrastructure": 0, "skipped": 0}
    consecutive_errors = 0
    for case in read_jsonl(cases_path):
        if case["id"] in done:
            stats["skipped"] += 1
            continue
        result = model.generate_result([Message("user", case["prompt"])])
        if result.error or result.truncated:  # unusable reply; left for a later run
            stats["infrastructure"] += 1
            consecutive_errors += 1
            log.warning("%s atlandi: %s", case["id"], result.error or "truncated")
            if consecutive_errors >= MAX_CONSECUTIVE_INFRA_ERRORS:
                log.error("art arda %d altyapi hatasi; durduruluyor (Ollama acik mi?)", consecutive_errors)
                break
            continue
        consecutive_errors = 0
        failures = classify_failures(case, result.text)
        append_jsonl(out_path, {"id": case["id"], "raw_reply": result.text, "failures": failures,
                                "done_reason": result.done_reason,
                                "elapsed_seconds": (result.metadata or {}).get("elapsed_seconds")})
        stats["done"] += 1
        stats["failed"] += bool(failures)
        if stats["done"] % 25 == 0:
            log.info("%d vaka bitti, %d hatali", stats["done"], stats["failed"])
    return stats


def relabel(cases_path: Path, rejected_path: Path) -> dict:
    """Recompute failure labels from stored replies, so a labelling fix needs no new model run."""
    cases = {case["id"]: case for case in read_jsonl(cases_path)}
    rows = read_jsonl(rejected_path)
    updated = [{**row, "failures": classify_failures(cases[row["id"]], row["raw_reply"])}
               if row["id"] in cases else row for row in rows]
    tmp = Path(rejected_path).with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated), encoding="utf-8")
    tmp.replace(rejected_path)
    return {"rows": len(rows),
            "changed": sum(old["failures"] != new["failures"] for old, new in zip(rows, updated)),
            "failed": sum(bool(row["failures"]) for row in updated)}


def main(argv: list[str] | None = None) -> int:
    from agent.profiles import create_model, get_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=OUT_DIR / "cases.jsonl")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "rejected.jsonl")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--relabel", action="store_true", help="recompute labels from stored replies only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.relabel:
        print(json.dumps(relabel(args.cases, args.out), ensure_ascii=False))
        return 0
    profile = get_profile("octopus-v10")
    # Same generation settings as eval/reliability_eval.py so rejected replies are on-policy.
    model = create_model(profile.name, tokenizer_dir=profile.tokenizer_dir, modelfile=profile.modelfile,
                         temperature=0, num_ctx=4096, num_predict=1024, seed=3407, host=args.host)
    stats = run(model, args.cases, args.out)
    print(json.dumps(stats, ensure_ascii=False))
    return 0 if stats["done"] or stats["skipped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
