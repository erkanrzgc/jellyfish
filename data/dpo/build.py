"""Stage 4: TRL conversational preference pairs, SFT correction view and manifest.

    uv run python -m data.dpo.build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from data.dpo.common import OUT_DIR, ROOT, classify_failures, read_jsonl
from data.sft.persona import LEGACY_TOOL_SYSTEM_PROMPT

VAL_PERCENT = 10


_EMPTY_DUSUNCE = re.compile(r"```dusunce```\s*(.*?)\s*```\s*", re.S)


def repair_dusunce(text: str) -> str:
    """Teachers sometimes emit ```dusunce``` <reasoning> ``` ; rebuild it as a proper fenced block."""
    return _EMPTY_DUSUNCE.sub(lambda m: f"```dusunce\n{m.group(1)}\n```\n", text, count=1)


def split_of(family_id: str) -> str:
    """Deterministic family-grouped split: every variant of a scenario lands on one side."""
    bucket = int.from_bytes(hashlib.sha256(family_id.encode()).digest()[:4], "big") % 100
    return "val" if bucket < VAL_PERCENT else "train"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build(cases_path: Path, rejected_path: Path, chosen_path: Path, out_dir: Path, *,
          gguf_sha256: str | None = None, system_prompt: str = LEGACY_TOOL_SYSTEM_PROMPT) -> dict:
    cases = {row["id"]: row for row in read_jsonl(cases_path)}
    chosen = {row["id"]: row for row in read_jsonl(chosen_path)}
    pairs: dict[str, list[dict]] = {"train": [], "val": []}
    sft, by_failure, by_category = [], Counter(), Counter()
    model_passed = dropped = relabelled = in_catalog_wrong_tool = 0
    for rejected in read_jsonl(rejected_path):
        case = cases.get(rejected["id"])
        if case is None:
            continue
        # Stored labels may predate a classifier fix; the raw reply is the source of truth.
        failures = classify_failures(case, rejected["raw_reply"])
        relabelled += failures != rejected["failures"]
        if not failures:
            model_passed += 1
            continue
        answer = repair_dusunce((chosen.get(rejected["id"]) or {}).get("chosen") or "")
        if not answer:
            dropped += 1
            continue
        prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": case["prompt"]}]
        meta = {"id": case["id"], "family_id": case["family_id"], "category": case["category"],
                "failures": failures, "teacher_model": chosen[rejected["id"]].get("teacher_model")}
        record = {"prompt": prompt, "chosen": [{"role": "assistant", "content": answer}],
                  "rejected": [{"role": "assistant", "content": rejected["raw_reply"]}], "meta": meta}
        pairs[split_of(case["family_id"])].append(record)
        sft.append({"messages": prompt + record["chosen"], "meta": meta})
        by_failure.update(failures)
        in_catalog_wrong_tool += "wrong_first_tool" in failures and "catalog" not in failures
        by_category[case["category"]] += 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "pairs_train.jsonl", pairs["train"])
    _write_jsonl(out_dir / "pairs_val.jsonl", pairs["val"])
    _write_jsonl(out_dir / "sft_corrections.jsonl", sft)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pairs": len(sft), "train": len(pairs["train"]), "val": len(pairs["val"]),
        "model_passed": model_passed, "dropped_chosen": dropped, "relabelled": relabelled,
        "in_catalog_wrong_tool": in_catalog_wrong_tool,
        "by_failure": dict(sorted(by_failure.items())), "by_category": dict(sorted(by_category.items())),
        "gguf_sha256": gguf_sha256,
        "inputs": {Path(p).name: _sha256(p) for p in (cases_path, rejected_path, chosen_path)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    from agent.profiles import get_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    weights = ROOT / get_profile("jellyfish-v10").weights_path
    manifest = build(args.dir / "cases.jsonl", args.dir / "rejected.jsonl", args.dir / "chosen.jsonl",
                     args.dir / "pairs", gguf_sha256=_sha256(weights) if weights.exists() else None)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
