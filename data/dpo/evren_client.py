"""Minimal EVREN LLM gateway client (OpenAI-compatible, stdlib only).

The key comes from EVREN_API_KEY or the Windows user environment. It never appears
in repr, logs or error messages."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

BASE_URL = "https://evren-llmapi.ssyz.org.tr/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_BACKOFF = 120.0


class EvrenError(RuntimeError):
    """Gateway failure after retries; the message never contains the key."""


def _registry_key() -> str:
    if sys.platform != "win32":
        return ""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
            return str(winreg.QueryValueEx(handle, "EVREN_API_KEY")[0])
    except OSError:
        return ""


def load_key(env: Mapping[str, str] = os.environ, registry: Callable[[], str] = _registry_key) -> str:
    key = (env.get("EVREN_API_KEY") or registry() or "").strip()
    if not key:
        raise EvrenError("EVREN_API_KEY tanimli degil (ortam degiskeni veya Windows kullanici ortami)")
    return key


@dataclass(frozen=True)
class Completion:
    content: str
    reasoning: str
    finish_reason: str | None
    usage: dict


@dataclass
class EvrenClient:
    key: str = field(repr=False)
    base_url: str = BASE_URL
    opener: Callable = field(default=urllib.request.urlopen, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    timeout: float = 600.0
    max_retries: int = 6
    # When a model is busy EVREN allows two concurrent requests per account on it
    # (429 fair_share_limit_exceeded), so the client enforces that cap itself.
    per_model_limit: int = 2
    _gates: dict = field(default_factory=dict, init=False, repr=False)
    _gates_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _gate(self, model: str) -> threading.BoundedSemaphore:
        with self._gates_lock:
            return self._gates.setdefault(model, threading.BoundedSemaphore(self.per_model_limit))

    def chat(self, messages: list[dict], *, model: str = DEFAULT_MODEL, max_tokens: int = 4096,
             temperature: float = 0.3, thinking: bool | None = None) -> Completion:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if thinking is not None:
            body["thinking"] = thinking
        with self._gate(model):
            payload = self._post("/chat/completions", body)
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        return Completion(content=message.get("content") or "",
                          reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
                          finish_reason=choice.get("finish_reason"), usage=payload.get("usage") or {})

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        last_error = ""
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.base_url + path, data=data, method="POST",
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            delay = min(5.0 * 2 ** attempt, _MAX_BACKOFF)
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:300]
                last_error = f"HTTP {error.code}: {detail}"
                if error.code not in _RETRYABLE:
                    raise EvrenError(last_error) from None
                retry_after = (error.headers or {}).get("Retry-After")
                if retry_after:
                    try:
                        delay = min(float(retry_after), _MAX_BACKOFF)
                    except ValueError:
                        pass
            except (OSError, ValueError) as error:  # timeouts, resets, bad JSON
                last_error = f"{type(error).__name__}: {error}"
            if attempt < self.max_retries:
                self.sleep(delay)
        raise EvrenError(f"{last_error} ({self.max_retries + 1} deneme)")
