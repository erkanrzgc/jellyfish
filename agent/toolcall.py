"""arac blok parse + modele render. Format: ```arac {"arac":..,"parametreler":{..}} ```.
Geri-besleme egitimle birebir: normalize.flatten_tool_messages reuse (tool->user 'ARAC CIKTISI')."""
from __future__ import annotations
from dataclasses import dataclass
import json
import re
from agent.messages import Message
from data.sft.normalize import flatten_tool_messages

_ARAC_RE = re.compile(r"```arac\b\s*(.*?)```", re.S | re.I)
_ARAC_START = re.compile(r"```arac\b", re.I)
_DUSUNCE_RE = re.compile(r"```dusunce\s*.*?```", re.S)


def strip_dusunce(text: str) -> str:
    """Kullaniciya giden metinden ```dusunce``` bloklarini sok (ic muhakeme gizli).

    Blok yoksa metni aynen dondurur; asla cokmez. arac blogu KORUNUR (ayri parse edilir).
    """
    return _DUSUNCE_RE.sub("", text or "").strip()


@dataclass(frozen=True)
class ToolCall:
    name: str
    params: dict


@dataclass(frozen=True)
class ParsedReply:
    calls: list[ToolCall]
    errors: list[str]


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"tekrarlanan alan: {key}")
        result[key] = value
    return result


def parse_tool_reply(text: str, *, validate_catalog: bool = False) -> ParsedReply:
    """Return valid calls AND errors; the runtime never executes a partly invalid reply."""
    from agent.catalog import get_spec, parameter_error
    calls, errors = [], []
    matches = list(_ARAC_RE.finditer(text or ""))
    if len(_ARAC_START.findall(text or "")) != len(matches):
        errors.append("kapanmamis arac blogu")
    for m in matches:
        try:
            d = json.loads(m.group(1), object_pairs_hook=_unique_keys,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, RecursionError):
            errors.append("gecersiz arac JSON'u")
            continue
        if not isinstance(d, dict) or set(d) - {"arac", "parametreler"}:
            errors.append("arac blogu yalniz arac/parametreler alanlarini tasimali")
            continue
        name, params = d.get("arac"), d.get("parametreler", {})
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            errors.append("arac adi bos olmayan bir metin olmali")
            continue
        if not isinstance(params, dict):
            errors.append("parametreler bir JSON nesnesi olmali")
            continue
        if validate_catalog:
            spec = get_spec(name)
            if spec is None:
                errors.append(f"bilinmeyen arac: {name}")
                continue
            issue = parameter_error(spec, params)
            if issue:
                errors.append(issue)
                continue
        calls.append(ToolCall(name=name, params=params))
    return ParsedReply(calls, errors)


def parse_arac_calls(text: str) -> list[ToolCall]:
    """Compatibility scorer/parser: skip malformed blocks without coercing their values."""
    return parse_tool_reply(text).calls


def render_for_model(messages: list[Message]) -> list[dict]:
    """Message listesini modele verilecek dict listesine cevir (tool rolu flatten'lanir)."""
    return flatten_tool_messages([m.to_dict() for m in messages])
