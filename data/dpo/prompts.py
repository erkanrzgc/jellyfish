"""Stage 1: teacher-written training prompts in the reliability-suite case schema.

    uv run python -m data.dpo.prompts --total 1500
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.catalog import CATALOG
from data.dpo.common import (OUT_DIR, ROOT, append_jsonl, catalog_excerpt, fold, normalized_key,
                             read_jsonl, token_jaccard, tokens)
from data.dpo.evren_client import DEFAULT_MODEL, EvrenClient, load_key

log = logging.getLogger(__name__)

CATEGORIES = ("fundamentals", "tooling", "pentest", "forensics", "incident", "authorization")
ID_PREFIX = {"fundamentals": "FUND", "tooling": "TOOL", "pentest": "PENT", "forensics": "FORN",
             "incident": "INCD", "authorization": "AUTH"}
REQUIRED_FIELDS = ("family_id", "category", "prompt", "expected_tools", "expected_params", "allow_tools")
CONTAMINATION_JACCARD = 0.6
DUPLICATE_JACCARD = 0.8

# EVREN terms v1 section 2: no critical-infrastructure attack content. Matched on folded text.
_CRITICAL_STEMS = ("scada", "modbus", "dnp3", "sebeke", "santral", "nukleer", "dogalgaz",
                   "su aritma", "enerji dagitim", "hastane")
_CRITICAL_WORDS = frozenset({"ics", "plc", "hmi"})

# Real-world targets: only private/documentation IPs, example.* and lab-style names are allowed.
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PUBLIC_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|tr|gov|edu|co|info|biz|me|ai|app|dev|"
                            r"xyz|de|uk|ru|cn|us|eu)\b")
_ALLOWED_DOMAIN = re.compile(r"(?:^|\.)example\.(?:com|org|net)$")
_HANDLE = re.compile(r"(?<![\w.])@[A-Za-z0-9_]{3,}")
# Phrases that tell the model not to use tools make a no-tool case trivially easy.
_GIVEAWAY = ("gerek yok", "calistirmadan", "kullanmadan", "arac calistirma", "komut calistirma",
             "komut istemiyorum", "teorik olarak", "kavramsal olarak")

# Category weights and focus areas follow the dev60 weak spots (eval/reports/reliability-v10-*).
FOCUS: dict[str, tuple[float, tuple[str, ...]]] = {
    "tooling": (0.25, (
        "Birbirine benzeyen işler (port tarama, dizin keşfi, alt alan adı, DNS, TLS) arasında doğru aracı seçmek.",
        "Hedef IP, URL, alan adı veya dosya yolu istemde açıkça geçen ve ilk çağrıda aynen verilmesi gereken görevler.",
        "Kullanıcının katalogda OLMAYAN uydurma bir araç adı söylediği ama katalogda karşılığı olan görevler.",
        "Yerel dosya veya log içinde arama, okuma ve özetleme görevleri.",
    )),
    "pentest": (0.20, (
        "Kapsamı yazılı yetkili lab veya CTF'te keşif adımı.",
        "Lab web uygulaması testi: dizin, parametre veya zafiyet keşfi, URL ile.",
        "Lab ağında SMB, SSH, HTTP gibi servisleri numaralandırma, IP ile.",
    )),
    "fundamentals": (0.10, (
        "Bir güvenlik aracının veya tekniğin nasıl çalıştığını soran, cevabı bilgi olan sorular.",
        "Kullanıcının yapıştırdığı hazır tarama çıktısını veya raporu yorumlama; kanıt sınırını doğru çizmek gerekir.",
        "Protokol ve kavram karşılaştırmaları (TLS, Kerberos, DNS, TCP bayrakları, hash ve şifreleme).",
    )),
    "incident": (0.10, (
        "Kendi sunucusunda bir olay (yük, disk, bağlantı, giriş denemesi); doğru cevap sonuca atlamadan "
        "tek bir tanı komutuyla kanıt toplamaktır.",
        "Kullanıcının mesajına yapıştırdığı log veya metrik parçasından kök neden yorumu; ek araç gerekmez.",
    )),
    "authorization": (0.15, (
        "Kullanıcı yetkisi olmayan bir üçüncü tarafı hedefliyor (rakip firma, komşu, eski iş arkadaşı); "
        "doğru cevap nazik ret ve araç çağırmamaktır.",
        "Eğitim veya kurgu bahanesiyle belirli bir kişiyi ya da kurumu hedefleme; doğru cevap ret.",
        "Kendi sunucusu veya yetkili lab açıkça belirtilmiş; doğru cevap yardım ve doğru araç.",
    )),
    "forensics": (0.20, (
        "Yerel bellek, disk imajı veya pcap dosyası analizi, dosya yolu ile.",
        "Kullanıcının mesajına yapıştırdığı adli çıktı parçasını yorumlama; ek araç gerekmez.",
    )),
}

# Tool-choice wave (2026-11-25): the clusters where gemma4:e4b picked a wrong first tool or bent an argument
# (native_pairs v1: binwalk for memory/disk images, nuclei for XSS, sqlmap moving a GET parameter to --data,
# gobuster URL in the wrong key, smbclient share format, tcpdump without its filter or output file).
TOOL_CHOICE_FOCUS: dict[str, tuple[float, tuple[str, ...]]] = {
    "forensics": (0.35, (
        "Bellek dökümü (.raw, .mem, .vmem) analizi: süreçler, ağ bağlantıları veya enjekte kod; dosya yolu ile.",
        "Disk imajı (.dd, .E01, .img) üzerinde dosya sistemi, silinmiş dosya veya zaman çizelgesi incelemesi; "
        "dosya yolu ile.",
        "Firmware veya gömülü cihaz imajındaki dosya sistemlerini ve gömülü dosyaları çıkarma; dosya yolu ile.",
        "Şüpheli bir çalıştırılabilir dosyanın yeteneklerini (ağ, kalıcılık, şifreleme) belirleme ya da tersine "
        "mühendislik; dosya yolu ile.",
        "Uzantısı yanıltıcı veya bilinmeyen bir dosyanın gerçek türünü belirleme; dosya yolu ile.",
        "Kayıtlı bir pcap dosyasında belirli bir protokol veya host trafiğini filtreleme; dosya yolu ile.",
    )),
    "pentest": (0.40, (
        "Yetkili lab web uygulamasında yansıyan veya DOM tabanlı XSS testi; parametreli tam URL ile.",
        "Yetkili lab web uygulamasında SQL enjeksiyonu testi; GET parametresi URL'nin içinde ve URL aynen "
        "verilmeli.",
        "Yetkili lab web uygulamasında gizli HTTP parametrelerini keşfetme; URL ile.",
        "Yetkili lab web uygulamasında belirli uzantılarla dizin ve dosya keşfi; URL ile.",
        "Yetkili lab ağında SMB paylaşımlarını ve izinlerini numaralandırma; IP ile.",
        "Yetkili lab ağında bilinen bir SMB paylaşımına bağlanıp dosyaları listeleme; //sunucu/paylasim biçiminde.",
    )),
    "tooling": (0.25, (
        "Belirli bir arayüzde belirli host ve port trafiğini yakalayıp verilen dosya yoluna kaydetme.",
        "Yerel bir log veya yapılandırma dosyasında belirli bir deseni arama; dosya yolu ve desen istemde açık.",
        "Yerel sunucuda dinleyen portları ve bunlara bağlı süreçleri listeleme.",
        "Birbirine benzeyen web işleri (genel zafiyet taraması, XSS, SQL enjeksiyonu, dizin keşfi) arasında doğru "
        "aracı seçmek; URL ile.",
    )),
}
PLANS = {"default": FOCUS, "tool-choice": TOOL_CHOICE_FOCUS}

SYSTEM_PROMPT = """Sen, Türkçe siber güvenlik modeli Jellyfish'in eğitim verisi için kullanıcı istemleri yazan bir veri mühendisisin.
Çıktın YALNIZCA bir JSON dizisi olacak; dizinin her elemanı şu alanları taşır:
- "family_id": kısa, benzersiz kebab-case senaryo kimliği.
- "prompt": Türkçe, doğal kullanıcı dili, 1-3 cümle, somut. Araç gerekiyorsa hedef somut olsun:
  RFC1918 IP (10.x, 192.168.x, 172.16-31.x), *.lab veya *.local alan adları, example.com/example.org, 127.0.0.1,
  ya da /tmp, /var/log, /home altında dosya yolları.
- "expected_tools": isteği çözmek için İLK adımda kabul edilebilir katalog araç adları (alternatifler); araç gerekmiyorsa [].
- "expected_params": ilk çağrıda olması gereken parametre alt kümesi. İstemde IP, alan adı, URL veya dosya yolu
  geçiyorsa onu kabul eden parametreyi aynı metinle yaz (ör. {"hedef": "10.10.5.20"}). Anahtar, listedeki HER
  araçta bulunan bir parametre adı olmalı; böyle bir ortak parametre yoksa {}.
- "allow_tools": expected_tools boş değilse true, boşsa false.
Kurallar:
1. İstemde beklenen aracın adını YAZMA. Kullanıcı görevi doğal dille anlatır, aracı model seçer.
2. Araç gerektirmeyen vakalarda "araç kullanma", "komut çalıştırma", "kavramsal anlat" gibi ipuçları VERME.
3. Ofansif işler yalnızca yetkili lab, CTF veya kapsamı yazılı pentest çerçevesinde olur. Kritik altyapı
   (enerji, su, SCADA/ICS, sağlık) KULLANMA.
4. Gerçek marka, gerçek kişi adı, sosyal medya kullanıcı adı veya gerçek genel IP KULLANMA. Yetkisiz senaryolarda
   "rakip firma", "komşum", "eski iş arkadaşım" gibi roller, 203.0.113.x / 198.51.100.x IP'leri ve
   *.example.com alan adları kullan. Lab hedefleri için *.lab, *.local, example.com ve RFC1918 IP'leri kullan.
5. Sunucu tanısı için run_cmd, read_file veya grep uygundur; "shell" bir privesc aracıdır, tanı alternatifi değildir.
6. Yalnızca aşağıdaki katalogdaki araç adlarını ve parametre adlarını kullan.

Araç kataloğu (ad(parametreler) [alan]):
{catalog}"""


def is_critical_infrastructure(prompt: str) -> bool:
    folded = fold(prompt)
    return any(stem in folded for stem in _CRITICAL_STEMS) or bool(_CRITICAL_WORDS & tokens(prompt))


def real_world_target(prompt: str) -> str | None:
    for raw in _IPV4.findall(prompt):
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:  # e.g. a four-part version number
            continue
        if not (address.is_private or address.is_loopback):
            return "gercek genel IP"
    for match in _PUBLIC_DOMAIN.finditer(prompt.lower()):
        if not _ALLOWED_DOMAIN.search(match.group(0)):
            return "gercek alan adi"
    if _HANDLE.search(prompt):
        return "gercek kullanici adi"
    return None


def names_expected_tool(prompt: str, tools: list[str]) -> bool:
    folded = fold(prompt)
    return any(re.search(rf"(?<![\w-]){re.escape(fold(tool))}(?![\w-])", folded) for tool in tools)


def repair_case(case: object) -> object:
    """Drop expected parameters that not every alternative tool accepts; the case itself is kept."""
    if not isinstance(case, dict):
        return case
    tools, params = case.get("expected_tools"), case.get("expected_params")
    if (not isinstance(tools, list) or not tools or not isinstance(params, dict)
            or any(tool not in CATALOG for tool in tools)):
        return case
    shared = {key: value for key, value in params.items()
              if all(key in CATALOG[tool].params for tool in tools)}
    return {**case, "expected_params": shared}


def validate_case(case: object) -> str | None:
    """Return a reason the teacher case is unusable, or None."""
    if not isinstance(case, dict) or any(key not in case for key in REQUIRED_FIELDS):
        return "eksik alan"
    if case["category"] not in CATEGORIES:
        return "bilinmeyen kategori"
    prompt, family = case["prompt"], case["family_id"]
    if not isinstance(prompt, str) or not prompt.strip():
        return "bos istem"
    if not isinstance(family, str) or not family.strip():
        return "bos family_id"
    tools, params, allow = case["expected_tools"], case["expected_params"], case["allow_tools"]
    if not isinstance(tools, list) or not isinstance(params, dict) or not isinstance(allow, bool):
        return "alan tipi hatali"
    unknown = [tool for tool in tools if tool not in CATALOG]
    if unknown:
        return f"katalog disi arac: {unknown}"
    if allow != bool(tools):
        return "allow_tools ile expected_tools celisiyor"
    if params and not tools:
        return "aracsiz vakada parametre olamaz"
    for key, value in params.items():
        if not isinstance(value, str) or any(key not in CATALOG[tool].params for tool in tools):
            return f"parametre katalogla uyusmuyor: {key}"
    if case["category"] == "fundamentals" and tools:  # this category targets unnecessary calls
        return "fundamentals vakasi aracsiz olmali"
    if case["category"] == "incident" and any(CATALOG[tool].domain == "privesc" for tool in tools):
        return "olay tanisinda privesc araci beklenemez"
    if is_critical_infrastructure(prompt):
        return "kritik altyapi hedefi"
    real_target = real_world_target(prompt)
    if real_target:
        return real_target
    if names_expected_tool(prompt, tools):
        return "beklenen arac adi istemde geciyor"
    if not allow and any(phrase in fold(prompt) for phrase in _GIVEAWAY):
        return "istem cevabi ele veriyor"
    return None


def parse_cases(text: str) -> list[dict]:
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    raw = fenced.group(1) if fenced else (re.search(r"\[.*\]", text, re.S) or [None])[0]
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _prompts_from_jsonl(path: Path) -> list[str]:
    return [row.get("prompt") or row.get("soru") or "" for row in read_jsonl(path)]


def load_eval_prompts(root: Path = ROOT) -> list[str]:
    """Every held-out eval prompt; generated training prompts must not overlap them."""
    suite = json.loads((root / "eval/suites/reliability-dev-v1.json").read_text(encoding="utf-8"))
    prompts = [case["prompt"] for case in suite["cases"]]
    prompts += _prompts_from_jsonl(root / "eval/data/toolcall_eval_tr.jsonl")
    prompts += _prompts_from_jsonl(root / "eval/data/quality_safety_tr.jsonl")
    lines = (root / "eval/data/cassiopeia_eval_prompts_tr.txt").read_text(encoding="utf-8").splitlines()
    prompts += [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return [prompt for prompt in prompts if prompt.strip()]


class ContaminationFilter:
    def __init__(self, prompts: list[str], threshold: float = CONTAMINATION_JACCARD):
        self.threshold = threshold
        self._keys = {normalized_key(prompt) for prompt in prompts}
        self._tokens = [tokens(prompt) for prompt in prompts]

    def is_contaminated(self, prompt: str) -> bool:
        if normalized_key(prompt) in self._keys:
            return True
        candidate = tokens(prompt)
        return any(token_jaccard(candidate, other) >= self.threshold for other in self._tokens)


def holdout_prompts(cases_path: Path) -> list[str]:
    """Prompts of the val families in an existing case file: new cases must not near-copy them either."""
    from data.dpo.build import split_of
    return [row["prompt"] for row in read_jsonl(cases_path) if split_of(row["family_id"]) == "val"]


def default_plan(total: int, per_call: int,
                 focus: dict[str, tuple[float, tuple[str, ...]]] = FOCUS) -> list[tuple[str, str]]:
    plan = []
    for category, (weight, focuses) in focus.items():
        calls = math.ceil(total * weight / per_call)
        plan += [(category, focuses[index % len(focuses)]) for index in range(calls)]
    return plan


def _ask(teacher, category: str, focus: str, index: int, per_call: int, model: str) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.replace("{catalog}", catalog_excerpt())},
        {"role": "user", "content": (f"Kategori: {category}\nOdak: {focus}\nVaryasyon no: {index}\n"
                                     f"{per_call} farklı vaka yaz. Senaryoları, hedefleri ve ifadeleri çeşitlendir.")},
    ]
    reply = teacher.chat(messages, model=model, max_tokens=8192, temperature=0.9, thinking=False)
    return [{**case, "category": category, "prompt_model": model} for case in parse_cases(reply.content)]


def generate(teacher, out_path: Path, *, plan: list[tuple[str, str]], per_call: int = 8,
             workers: int | None = None, eval_prompts: list[str] | None = None,
             models: list[str] | tuple[str, ...] = (DEFAULT_MODEL,), id_stem: str = "DPO",
             known_prompts: list[str] | None = None) -> dict:
    """`known_prompts` (e.g. another case file) count as duplicates; `id_stem` keeps ids apart from that file."""
    existing = read_jsonl(out_path)
    contamination = ContaminationFilter(load_eval_prompts() if eval_prompts is None else eval_prompts)
    seen_prompts = [row["prompt"] for row in existing] + list(known_prompts or [])
    seen_keys = {normalized_key(prompt) for prompt in seen_prompts}
    seen_tokens = [tokens(prompt) for prompt in seen_prompts]
    counters = {prefix: 0 for prefix in ID_PREFIX.values()}
    for row in existing:
        prefix, number = row["id"].split("-")[1:3]
        counters[prefix] = max(counters[prefix], int(number))
    stats = {"requests": len(plan), "accepted": 0, "invalid": 0, "duplicate": 0, "contaminated": 0,
             "teacher_errors": 0}
    invalid_reasons: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers or 2 * len(models)) as pool:
        futures = [pool.submit(_ask, teacher, category, focus, index, per_call, models[index % len(models)])
                   for index, (category, focus) in enumerate(plan)]
        for future in as_completed(futures):
            try:
                batch = future.result()
            except Exception as error:  # noqa: BLE001 - one failed request must not stop the run
                stats["teacher_errors"] += 1
                log.warning("istem istegi basarisiz: %s", error)
                continue
            for case in batch:
                case = repair_case(case)
                reason = validate_case(case)
                if reason:
                    stats["invalid"] += 1
                    invalid_reasons[reason.split(":")[0]] += 1
                    continue
                if contamination.is_contaminated(case["prompt"]):
                    stats["contaminated"] += 1
                    continue
                key, case_tokens = normalized_key(case["prompt"]), tokens(case["prompt"])
                if key in seen_keys or any(token_jaccard(case_tokens, other) >= DUPLICATE_JACCARD
                                           for other in seen_tokens):
                    stats["duplicate"] += 1
                    continue
                prefix = ID_PREFIX[case["category"]]
                counters[prefix] += 1
                row = {"id": f"{id_stem}-{prefix}-{counters[prefix]:04d}", "split": "train",
                       **{field: case[field] for field in REQUIRED_FIELDS},
                       "prompt_model": case.get("prompt_model")}
                append_jsonl(out_path, row)
                seen_keys.add(key)
                seen_tokens.append(case_tokens)
                stats["accepted"] += 1
    stats["invalid_reasons"] = dict(invalid_reasons.most_common())
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=1500)
    parser.add_argument("--per-call", type=int, default=8)
    parser.add_argument("--models", default=f"{DEFAULT_MODEL},qwen3.8-flash-next",
                        help="comma-separated teacher models; requests are spread round-robin")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "cases.jsonl")
    parser.add_argument("--plan", choices=sorted(PLANS), default="default")
    parser.add_argument("--id-stem", default="DPO", help="id prefix; use a new one when --out is a new file")
    parser.add_argument("--known-cases", type=Path, action="append", default=[],
                        help="existing case file: its prompts are duplicates and its val families are held out")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    known = [row["prompt"] for path in args.known_cases for row in read_jsonl(path)]
    held_out = [prompt for path in args.known_cases for prompt in holdout_prompts(path)]
    stats = generate(EvrenClient(load_key()), args.out, plan=default_plan(args.total, args.per_call, PLANS[args.plan]),
                     per_call=args.per_call, models=[m.strip() for m in args.models.split(",") if m.strip()],
                     id_stem=args.id_stem, known_prompts=known, eval_prompts=load_eval_prompts() + held_out)
    print(json.dumps(stats, ensure_ascii=False))
    return 0 if stats["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
