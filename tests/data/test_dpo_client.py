"""EVREN client: key handling, retries and response parsing without network access."""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from data.dpo.evren_client import EvrenClient, EvrenError, load_key


class _Resp(io.BytesIO):
    def __init__(self, payload: dict):
        super().__init__(json.dumps(payload).encode())
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok(content="Merhaba", reasoning="", finish="stop"):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return _Resp({"choices": [{"message": message, "finish_reason": finish}],
                  "usage": {"total_tokens": 12}})


def _http_error(code: int, retry_after: str | None = None, body: str = "busy"):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("u", code, "err", headers, io.BytesIO(body.encode()))


class _Opener:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_load_key_prefers_environment():
    assert load_key({"EVREN_API_KEY": " evren_llm_x "}, registry=lambda: "other") == "evren_llm_x"


def test_load_key_falls_back_to_registry():
    assert load_key({}, registry=lambda: "evren_llm_reg") == "evren_llm_reg"


def test_load_key_missing_raises():
    with pytest.raises(EvrenError):
        load_key({}, registry=lambda: "")


def test_repr_never_contains_key():
    client = EvrenClient("evren_llm_secret123")
    assert "secret123" not in repr(client)


def test_chat_returns_content_reasoning_and_sends_bearer():
    opener = _Opener(_ok("Cevap", reasoning="dusundum"))
    out = EvrenClient("evren_llm_k", opener=opener).chat([{"role": "user", "content": "x"}])
    assert (out.content, out.reasoning, out.finish_reason) == ("Cevap", "dusundum", "stop")
    assert out.usage == {"total_tokens": 12}
    request = opener.requests[0]
    assert request.get_header("Authorization") == "Bearer evren_llm_k"
    body = json.loads(request.data)
    assert body["model"] == "deepseek-v4.1-flash" and body["max_tokens"] >= 4096


def test_thinking_flag_is_forwarded_only_when_set():
    opener = _Opener(_ok(), _ok())
    client = EvrenClient("k", opener=opener)
    client.chat([{"role": "user", "content": "x"}])
    client.chat([{"role": "user", "content": "x"}], thinking=False)
    assert "thinking" not in json.loads(opener.requests[0].data)
    assert json.loads(opener.requests[1].data)["thinking"] is False


def test_retries_429_honoring_retry_after():
    sleeps = []
    opener = _Opener(_http_error(429, retry_after="7"), _ok("tamam"))
    out = EvrenClient("k", opener=opener, sleep=sleeps.append).chat([{"role": "user", "content": "x"}])
    assert out.content == "tamam" and sleeps == [7.0]


def test_retries_timeouts_then_gives_up():
    opener = _Opener(*[TimeoutError("slow")] * 3)
    client = EvrenClient("k", opener=opener, sleep=lambda s: None, max_retries=2)
    with pytest.raises(EvrenError, match="slow"):
        client.chat([{"role": "user", "content": "x"}])
    assert len(opener.requests) == 3


def test_concurrent_requests_per_model_are_capped():
    import threading
    import time

    lock, active, peak = threading.Lock(), {"n": 0}, {"n": 0}

    def opener(request, timeout=None):
        with lock:
            active["n"] += 1
            peak["n"] = max(peak["n"], active["n"])
        time.sleep(0.05)
        with lock:
            active["n"] -= 1
        return _ok()

    client = EvrenClient("k", opener=opener, per_model_limit=2)
    threads = [threading.Thread(target=client.chat, args=([{"role": "user", "content": "x"}],))
               for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak["n"] == 2


def test_non_retryable_error_raises_immediately_without_key():
    opener = _Opener(_http_error(400, body="bad request"), _ok())
    client = EvrenClient("evren_llm_secret", opener=opener, sleep=lambda s: None)
    with pytest.raises(EvrenError) as info:
        client.chat([{"role": "user", "content": "x"}])
    assert "400" in str(info.value) and "secret" not in str(info.value)
    assert len(opener.requests) == 1
