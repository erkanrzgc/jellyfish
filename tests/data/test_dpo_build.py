"""Build stage: TRL preference records, family-grouped split, SFT view and manifest."""
from __future__ import annotations

import json

from data.dpo.build import build, split_of
from data.sft.persona import OCTOPUS_TOOL_SYSTEM_PROMPT


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _fixture(tmp_path, n=30):
    cases, rejected, chosen = [], [], []
    for i in range(n):
        cid = f"DPO-TOOL-{i:04d}"
        cases.append({"id": cid, "family_id": f"fam-{i % 10}", "category": "tooling",
                      "prompt": f"istem {i}", "expected_tools": ["nmap"],
                      "expected_params": {}, "allow_tools": True})
        rejected.append({"id": cid, "raw_reply": f"kötü {i}", "failures": ["missing_call"]})
        chosen.append({"id": cid, "chosen": f"iyi {i}" if i != 0 else None, "attempts": 1,
                       "teacher_model": "deepseek-v4.1-flash", "problems": [] if i else ["missing_call"]})
    rejected[1]["raw_reply"] = 'Tarıyorum.\n```arac\n{"arac": "nmap", "parametreler": {"hedef": "10.0.0.1"}}\n```'
    rejected[1]["failures"] = ["missing_call"]  # stale label: build recomputes, the model was right
    paths = tmp_path / "c.jsonl", tmp_path / "r.jsonl", tmp_path / "ch.jsonl"
    for path, rows in zip(paths, (cases, rejected, chosen)):
        _jsonl(path, rows)
    return paths


def test_split_is_deterministic_and_grouped_by_family():
    assert split_of("fam-1") == split_of("fam-1")
    assert {split_of(f"fam-{i}") for i in range(200)} == {"train", "val"}


def test_build_writes_preference_records_sft_view_and_manifest(tmp_path):
    cases, rejected, chosen = _fixture(tmp_path)
    manifest = build(cases, rejected, chosen, tmp_path / "out", gguf_sha256="abc")
    out = tmp_path / "out"
    pairs = [json.loads(l) for f in ("pairs_train.jsonl", "pairs_val.jsonl")
             for l in (out / f).read_text(encoding="utf-8").splitlines()]
    assert len(pairs) == 28  # 30 minus one dropped chosen, minus one passing rejected
    record = pairs[0]
    assert record["prompt"][0] == {"role": "system", "content": OCTOPUS_TOOL_SYSTEM_PROMPT}
    assert record["prompt"][1]["role"] == "user"
    assert record["chosen"][0]["role"] == "assistant" and record["rejected"][0]["role"] == "assistant"
    assert record["chosen"][0]["content"] != record["rejected"][0]["content"]
    families = {}
    for name in ("pairs_train.jsonl", "pairs_val.jsonl"):
        for line in (out / name).read_text(encoding="utf-8").splitlines():
            families.setdefault(json.loads(line)["meta"]["family_id"], set()).add(name)
    assert all(len(v) == 1 for v in families.values())
    sft = [json.loads(l) for l in (out / "sft_corrections.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(sft) == 28 and [m["role"] for m in sft[0]["messages"]] == ["system", "user", "assistant"]
    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    assert manifest["pairs"] == 28 and manifest["dropped_chosen"] == 1 and manifest["model_passed"] == 1
    assert manifest["by_failure"]["missing_call"] == 28 and manifest["gguf_sha256"] == "abc"
    assert manifest["relabelled"] == 1 and manifest["in_catalog_wrong_tool"] == 0


def test_empty_dusunce_fence_is_repaired_into_a_real_block():
    from data.dpo.build import repair_dusunce
    broken = "```dusunce``` Kullanici kendi lab'inda. ```  Once yedek aliyorum.\n```arac\n{}\n```"
    assert repair_dusunce(broken) == "```dusunce\nKullanici kendi lab'inda.\n```\nOnce yedek aliyorum.\n```arac\n{}\n```"
    good = "```dusunce\nTamam.\n```\nCevap."
    assert repair_dusunce(good) == good
