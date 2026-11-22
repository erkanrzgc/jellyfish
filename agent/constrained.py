"""Constrained repair of invalid ```arac``` calls.

The model first generates freely, exactly as trained. Only when a call block is invalid (unknown tool,
unknown parameter key, missing required key, bad JSON, unclosed block) the reply is cut at that block's
opener and the call body is regenerated under an Ollama ``format`` JSON schema that admits only catalog
tools with their own parameter keys. Valid replies and prose replies are never touched, so the repair
can only turn an invalid call into a valid one.

By default only calls whose tool name is already in the catalog are repaired (bad keys, missing required
keys, bad JSON). Repairing an invented name forces the schema to pick SOME real tool, which can turn a
non-executable call on a refusal prompt into a runnable one, so it needs ``allow_unknown_names=True``.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable

from agent.catalog import CATALOG, REQUIRED_PARAMS, TARGET_KEYS, parameter_error

_OPENER = re.compile(r"```arac\b", re.I)
_BLOCK_END = "```"
_NAME = re.compile(r'"arac"\s*:\s*"([^"]*)"')


def _required_keys(name: str, params: tuple[str, ...]) -> list[str]:
    if name in REQUIRED_PARAMS:
        return list(REQUIRED_PARAMS[name])
    targets = [key for key in params if key in TARGET_KEYS]
    # With one network-target key the target is mandatory; with several the model picks one.
    return targets if len(targets) == 1 else []


def call_schema(names: Iterable[str] | None = None) -> dict:
    """JSON schema for one call body; ``names`` limits it to a tool subset (unknown names ignored)."""
    selected = list(CATALOG) if names is None else [name for name in names if name in CATALOG]
    branches = []
    for name in selected:
        params = CATALOG[name].params
        body: dict = {"type": "object", "properties": {key: {"type": "string"} for key in params},
                      "additionalProperties": False}
        required = _required_keys(name, params)
        if required:
            body["required"] = required
        branches.append({"type": "object", "properties": {"arac": {"const": name}, "parametreler": body},
                         "required": ["arac", "parametreler"], "additionalProperties": False})
    return {"anyOf": branches}


def _block_is_valid(block: str) -> bool:
    from agent.toolcall import parse_tool_reply
    parsed = parse_tool_reply(block, validate_catalog=True)
    return bool(parsed.calls) and not parsed.errors


def _names_catalog_tool(block: str) -> bool:
    match = _NAME.search(block)
    return bool(match) and match.group(1) in CATALOG


def repair_prefix(text: str, *, allow_unknown_names: bool = False) -> str | None:
    """Reply text up to and including the first invalid block's opener, or None if it is not repaired."""
    text = text or ""
    for opener in _OPENER.finditer(text):
        end = text.find(_BLOCK_END, opener.end())
        block = text[opener.start():end + len(_BLOCK_END)] if end != -1 else text[opener.start():]
        if end != -1 and _block_is_valid(block):
            continue
        if allow_unknown_names or _names_catalog_tool(block):
            return text[:opener.end()] + "\n"
        return None
    return None


def splice_repaired_call(prefix: str, body: str) -> str:
    return prefix + body.strip() + "\n" + _BLOCK_END


def is_valid_repair(body: str) -> bool:
    """The constrained body must itself be one valid catalog call."""
    try:
        data = json.loads(body)
    except ValueError:
        return False
    if not isinstance(data, dict) or set(data) != {"arac", "parametreler"}:
        return False
    spec = CATALOG.get(data["arac"]) if isinstance(data["arac"], str) else None
    return spec is not None and parameter_error(spec, data["parametreler"]) is None
