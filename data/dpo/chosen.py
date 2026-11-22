"""Stage 3: teacher-written replies that fix v1.0's failure and pass the structural gate.

    uv run python -m data.dpo.chosen
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.toolcall import parse_tool_reply
from data.dpo.common import (OUT_DIR, TURKISH_LETTERS, append_jsonl, catalog_excerpt, classify_failures,
                             fold, prose_outside_blocks, read_jsonl)
from data.dpo.evren_client import DEFAULT_MODEL, EvrenClient, EvrenError, load_key
from data.sft.persona import OCTOPUS_TOOL_SYSTEM_PROMPT

log = logging.getLogger(__name__)

INSTRUCTIONS = """Sen, Türkçe siber güvenlik modeli Octópus için örnek (altın) cevaplar yazan kıdemli bir güvenlik uzmanısın.
Kurallar:
1. Yalnızca Türkçe yaz; Octópus kendi ağzından konuşur.
2. İstersen en başa 1-3 cümlelik bir ```dusunce``` bloğu koy.
3. Araç gerekiyorsa kısa bir Türkçe açıklama cümlesi yaz, ardından TAM OLARAK BİR blok:
```arac
{"arac": "<katalog adı>", "parametreler": {"<parametre>": "<metin değer>"}}
```
Yalnızca verilen parametre adlarını kullan, değerler metin olsun. Araç çıktısını UYDURMA; çağrıdan sonra dur.
4. Araç gerekmiyorsa ```arac``` bloğu YAZMA; doğru, öz ve kanıt sınırına sadık bir cevap ver.
5. Yetkisiz veya gerçek üçüncü taraf hedeflere yönelik isteklerde nazikçe reddet ve araç çağırma.
6. Bilmediğin sürüm, CVE veya kaynak uydurma.
7. Hedefleri gerçekmiş gibi ele al. IP aralığı veya alan adı hakkında yorum yapma ("RFC1918", "özel adres",
   "dokümantasyon aralığı", "rezerve alan adı" deme). Yetki kararını yalnızca kullanıcının anlattığı bağlama göre
   ver: açıkça yetkisiz bir üçüncü taraf ise reddet; aksi halde iyi niyet varsay, yetki sorgulamadan yardım et.
8. run_cmd komutu kullanıcının kendi makinesinde çalışır; komutu ssh ile sarmalama."""

# Synthetic-target reasoning that must not be learned: authorization judged from address ranges.
_TARGET_COMMENTARY = ("rfc1918", "rfc 1918", "ozel adres", "ozel ip", "ozel bir adres", "dokumantasyon",
                      "rezerve", "test amacli ayrilmis", "ornek alan adi", "203.0.113.0/24", "198.51.100.0/24")


def build_teacher_messages(case: dict) -> list[dict]:
    tools = case["expected_tools"]
    if tools:
        expectation = (f"Bu istekte TEK bir araç çağrısı yap; araç şunlardan biri olmalı: {', '.join(tools)}.\n"
                       f"Kullanabileceğin araçlar ve parametreleri:\n{catalog_excerpt(tools)}")
        if case["expected_params"]:
            expectation += ("\nÇağrı en az şu parametreleri bu değerlerle içermeli: "
                            + json.dumps(case["expected_params"], ensure_ascii=False))
    else:
        expectation = "Bu istekte araç ÇAĞIRMA; ```arac``` bloğu yazma."
    user = (f"## Octópus sistem talimatı\n{OCTOPUS_TOOL_SYSTEM_PROMPT}\n\n## Kullanıcı mesajı\n{case['prompt']}\n\n"
            f"## Beklenen davranış\n{expectation}\n\nYalnızca Octópus'un vereceği asistan cevabını yaz.")
    return [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}]


def acceptance_problems(case: dict, text: str) -> list[str]:
    """Why a teacher reply cannot be a `chosen` answer; an empty list means accepted."""
    problems = classify_failures(case, text)
    calls = parse_tool_reply(text).calls
    if len(calls) > 1:
        problems.append("multiple_calls")
    if any(call.name == "run_cmd" and str(call.params.get("komut", "")).lstrip().startswith("ssh ")
           for call in calls):
        problems.append("ssh_wrapped")
    prose = prose_outside_blocks(text)
    if not prose:
        problems.append("no_prose")
    if not TURKISH_LETTERS & set(prose):
        problems.append("not_turkish")
    folded = fold(text)
    if any(phrase in folded for phrase in _TARGET_COMMENTARY):
        problems.append("target_commentary")
    return problems


def _solve(teacher, case: dict, retries: int, model: str) -> dict | None:
    """Return a result row, or None when the gateway failed so a later run can retry."""
    messages = build_teacher_messages(case)
    problems: list[str] = []
    answered = False
    for attempt in range(1, retries + 2):
        try:
            reply = teacher.chat(messages, model=model, max_tokens=4096, temperature=0.3)
        except EvrenError as error:
            log.warning("%s teacher hatasi: %s", case["id"], error)
            continue
        answered = True
        text = (reply.content or "").strip()
        problems = ["truncated"] if reply.finish_reason == "length" else acceptance_problems(case, text)
        if not problems:
            return {"id": case["id"], "chosen": text, "attempts": attempt, "teacher_model": model,
                    "problems": []}
        messages = [*messages, {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Bu cevap kabul edilmedi ({', '.join(problems)}). "
                                                "Kurallara uyan yeni bir cevap yaz."}]
    if not answered:
        return None
    return {"id": case["id"], "chosen": None, "attempts": retries + 1, "teacher_model": model,
            "problems": problems}


def run(teacher, cases_path: Path, rejected_path: Path, out_path: Path, *, workers: int = 8,
        retries: int = 2, model: str = DEFAULT_MODEL) -> dict:
    cases = {case["id"]: case for case in read_jsonl(cases_path)}
    done = {row["id"] for row in read_jsonl(out_path)}
    # Recompute instead of trusting stored labels, so a classifier fix reaches a rejected file written earlier.
    failed = [row["id"] for row in read_jsonl(rejected_path)
              if row["id"] in cases and classify_failures(cases[row["id"]], row["raw_reply"])]
    todo = [cases[cid] for cid in failed if cid not in done]
    stats = {"accepted": 0, "dropped": 0, "skipped": len(failed) - len(todo), "teacher_errors": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_solve, teacher, case, retries, model) for case in todo]
        for future in as_completed(futures):
            row = future.result()
            if row is None:
                stats["teacher_errors"] += 1
                continue
            append_jsonl(out_path, row)
            stats["accepted" if row["chosen"] else "dropped"] += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=OUT_DIR / "cases.jsonl")
    parser.add_argument("--rejected", type=Path, default=OUT_DIR / "rejected.jsonl")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "chosen.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run(EvrenClient(load_key()), args.cases, args.rejected, args.out, workers=args.workers)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
