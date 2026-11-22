"""NativeChatModel: a stock instruction model with native tool calling (Ollama /api/chat `tools`).

The harness contract stays the ```arac``` text the loop already parses: native tool_calls coming back are
rendered as ```arac``` blocks, and earlier assistant blocks are turned back into native tool_calls in the
history, with tool results sent as `tool` turns. Loop, policy, approval and executors are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from agent.backends.gguf_model import GenerationResult
from agent.catalog import CATALOG
from agent.messages import Message
from agent.toolcall import _ARAC_RE, parse_tool_reply
from data.sft.persona import OCTOPUS_SYSTEM_PROMPT

DEFAULT_NATIVE_MODEL = "gemma4:e4b"
NATIVE_SYSTEM_PROMPT = OCTOPUS_SYSTEM_PROMPT + (
    "\n\nBir işi yapmak için araç gerekiyorsa verilen araçlardan uygun olanı çağır; araç gerekmiyorsa "
    "(kavram sorusu, yapıştırılan çıktının yorumu, reddedilmesi gereken istek) araç çağırmadan Türkçe yanıt ver.")


def native_tools() -> list[dict]:
    """Every catalog tool as an Ollama native tool with its own string parameters."""
    from agent.constrained import required_keys
    tools = []
    for name, spec in CATALOG.items():
        parameters = {"type": "object", "properties": {key: {"type": "string"} for key in spec.params},
                      "required": required_keys(name, spec.params)}
        tools.append({"type": "function", "function": {
            "name": name, "description": f"{name} ({spec.domain}, risk: {spec.risk})", "parameters": parameters}})
    return tools


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def to_arac_reply(message: dict) -> str:
    """Native tool_calls -> ```arac``` blocks; unparseable arguments stay visibly invalid."""
    parts = [(message.get("content") or "").strip()]
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                parts.append(f"```arac\n{{\"arac\": {json.dumps(function.get('name'))}, {arguments}\n```")
                continue
        if isinstance(arguments, dict):
            arguments = {key: _text(value) for key, value in arguments.items()}
        body = json.dumps({"arac": function.get("name"), "parametreler": arguments}, ensure_ascii=False)
        parts.append(f"```arac\n{body}\n```")
    return "\n".join(part for part in parts if part)


def to_chat_messages(messages: list[Message], system_prompt: str) -> list[dict]:
    """Harness history -> Ollama chat messages with native tool_calls and tool turns."""
    chat = [{"role": "system", "content": system_prompt}]
    last_tool = None
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "assistant":
            calls = parse_tool_reply(message.content).calls
            entry = {"role": "assistant", "content": _ARAC_RE.sub("", message.content).strip()}
            if calls:
                entry["tool_calls"] = [{"function": {"name": c.name, "arguments": c.params}} for c in calls]
                last_tool = calls[-1].name
            chat.append(entry)
        elif message.role == "tool":
            entry = {"role": "tool", "content": message.content}
            if last_tool:
                entry["tool_name"] = last_tool
            chat.append(entry)
        else:
            chat.append({"role": message.role, "content": message.content})
    return chat


@dataclass
class NativeChatModel:
    """Same call interface as GgufModel: messages -> reply text with ```arac``` blocks."""
    model: str = DEFAULT_NATIVE_MODEL
    host: str = "http://localhost:11434"
    system_prompt: str = NATIVE_SYSTEM_PROMPT
    temperature: float = 0.0
    seed: int = 3407
    num_ctx: int = 16384  # the 130-tool schema alone is ~6.7k tokens
    num_predict: int = 1024
    timeout: int = 600

    def __call__(self, messages: list[Message]) -> str:
        return self.generate_result(messages).display_text()

    def generation_settings(self) -> dict:
        return {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx,
                "num_predict": self.num_predict}

    def generate_result(self, messages: list[Message]) -> GenerationResult:
        started = time.perf_counter()
        body = {"model": self.model, "stream": False, "think": False, "tools": native_tools(),
                "messages": to_chat_messages(messages, self.system_prompt), "options": self.generation_settings()}
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        metadata = {"settings": self.generation_settings(), "model": self.model, "host": self.host,
                    "request_sha256": hashlib.sha256(encoded).hexdigest()}
        request = urllib.request.Request(f"{self.host}/api/chat", data=encoded,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return GenerationResult(error=f"HATA: Ollama HTTP {error.code} ({self.model})", metadata=metadata)
        except TimeoutError:
            return GenerationResult(error=f"HATA: Ollama zaman asimi ({self.timeout}s)", metadata=metadata)
        except (urllib.error.URLError, OSError) as error:
            return GenerationResult(error=f"HATA: Ollama'ya baglanilamadi ({error})", metadata=metadata)
        except (ValueError, UnicodeError) as error:
            return GenerationResult(error=f"HATA: gecersiz Ollama yaniti ({error})", metadata=metadata)
        metadata["elapsed_seconds"] = time.perf_counter() - started
        if not isinstance(data, dict) or data.get("error") or not isinstance(data.get("message"), dict):
            return GenerationResult(error=f"HATA: Ollama: {data.get('error') if isinstance(data, dict) else data}",
                                    metadata=metadata)
        metadata["response"] = {key: value for key, value in data.items() if key != "message"}
        return GenerationResult(text=to_arac_reply(data["message"]), done_reason=data.get("done_reason"),
                                truncated=data.get("done_reason") == "length", metadata=metadata)
