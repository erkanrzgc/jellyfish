"""Normalize a single target without doing DNS or network I/O."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit


def normalize_target(value: str):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("hedef tek ve bos olmayan bir metin olmali")
    if any(c.isspace() for c in value) or value.startswith("-"):
        raise ValueError("gecersiz hedef")
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            raise ValueError("gecersiz hedef URL'si")
        if not parsed.hostname:
            raise ValueError("URL host'u eksik")
        _ = parsed.port  # validate the port, including malformed/non-numeric values
        return normalize_target(parsed.hostname)
    if "%" in value:
        raise ValueError("gecersiz hedef host'u")
    try:
        return ipaddress.ip_network(value, strict=False)
    except ValueError:
        pass
    if "/" in value:
        raise ValueError("gecersiz IP/CIDR")
    if ":" in value:
        parsed = urlsplit("//" + value)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("gecersiz host/port")
        _ = parsed.port
        return normalize_target(parsed.hostname)
    domain = value.rstrip(".").encode("idna").decode("ascii").lower()
    if len(domain) > 253 or re.fullmatch(r"[0-9.]+", domain):
        raise ValueError("gecersiz domain")
    if not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part) for part in domain.split(".")):
        raise ValueError("gecersiz domain")
    return domain


def target_in_scope(value: str, scope: list[str]) -> bool:
    try:
        target = normalize_target(value)
    except (ValueError, UnicodeError):
        return False
    for item in scope:
        try:
            allowed = normalize_target(item)
        except (ValueError, UnicodeError):
            continue
        if isinstance(target, str) or isinstance(allowed, str):
            if isinstance(target, str) and target == allowed:
                return True
        elif target.version == allowed.version and target.subnet_of(allowed):
            return True
    return False
