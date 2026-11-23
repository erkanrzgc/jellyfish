"""GgufModel: gercek Jellyfish'u Ollama uzerinden harness'e baglar (mock yerine).

Prompt EGITIMLE BIREBIR: kaydedilmis tokenizer'in kendi chat template'i (system rolu +
enable_thinking'i destekler) uygulanir, bastaki cift <bos> siyrilir (ADR 0003 tuzagi),
Ollama'ya RAW modda gonderilir. Ollama'nin kendi gemma2 template'ine GUVENILMEZ."""
from __future__ import annotations

import json
import hashlib
import urllib.error
import urllib.request
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agent.messages import Message
from agent.config import DEFAULT_MODELFILE, DEFAULT_TOKENIZER_DIR, LEGACY_MODEL as DEFAULT_MODEL
from agent.toolcall import render_for_model
from data.sft.normalize import ensure_system
from data.sft.persona import LEGACY_TOOL_SYSTEM_PROMPT

_EOT = "<end_of_turn>"


def load_tokenizer(tokenizer_dir: str):
    """transformers'i LAZY import et (yalniz --gguf yolu; torch weights yuklenmez)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)


def render_prompt(dicts: list[dict], tokenizer, *, bos_token: str | None = "<bos>") -> str:
    """dict listesi -> egitim-birebir prompt metni (tokenize=False, add_generation_prompt=True)."""
    try:
        t = tokenizer.apply_chat_template(
            dicts, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # bazi template'ler enable_thinking kwarg'ini tanimaz
        t = tokenizer.apply_chat_template(dicts, tokenize=False, add_generation_prompt=True)
    # Gemma template metne literal <bos> basar; Ollama raw'da tek BOS yeter -> bastakini siyir.
    if bos_token and t.startswith(bos_token):
        t = t[len(bos_token):]
    return t


@dataclass(frozen=True)
class GenerationResult:
    text: str = ""
    error: str | None = None
    done_reason: str | None = None
    truncated: bool = False
    metadata: dict = field(default_factory=dict)

    def display_text(self) -> str:
        if self.error:
            return self.error
        if self.truncated:
            return "HATA: model yaniti kesildi; gorev tamamlanmadi. Kismi yanit:\n" + self.text.strip()
        return self.text.strip()


@dataclass
class GgufModel:
    """Ollama uzerinden model. __call__ mock ScriptedModel ile ayni arayuz."""
    model: str = DEFAULT_MODEL
    host: str = "http://localhost:11434"
    tokenizer_dir: str = DEFAULT_TOKENIZER_DIR
    system_prompt: str = LEGACY_TOOL_SYSTEM_PROMPT  # araç-farkında: model ```arac``` bloğu basar
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repeat_penalty: float = 1.15
    num_predict: int = 512
    seed: int = 3407
    num_ctx: int = 4096
    stop_tokens: tuple[str, ...] = (_EOT,)
    bos_token: str | None = "<bos>"
    native_tool_roles: bool = False
    modelfile: str = DEFAULT_MODELFILE
    timeout: int = 180
    renderer: Callable[[list[dict]], str] | None = None  # test enjeksiyonu (transformers'i atlar)
    constrain_calls: bool = False  # gecersiz arac blogunu katalog-semasiyla yeniden uret (agent.constrained)
    constrain_tools: tuple[str, ...] | None = None  # onarim semasini bu araclarla sinirla (None = tum katalog)
    repair_unknown_names: bool = False  # uydurma ad onarimi: ret-istemini calisir cagriya cevirebilir, ayri olc
    _tokenizer: object = field(default=None, init=False, repr=False)

    def __call__(self, messages: list[Message]) -> str:
        result = self.generate_result(messages)
        return result.display_text()

    def generation_settings(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty, "num_predict": self.num_predict,
                "seed": self.seed, "num_ctx": self.num_ctx, "stop": list(self.stop_tokens)}

    def generate_result(self, messages: list[Message]) -> GenerationResult:
        started = time.perf_counter()
        dicts = [m.to_dict() for m in messages] if self.native_tool_roles else render_for_model(messages)
        dicts = ensure_system(dicts, self.system_prompt)
        try:
            prompt = self._render(dicts)
        except Exception as e:  # noqa: BLE001 -- dongu asla cokmemeli
            return GenerationResult(error=f"HATA: prompt/tokenizer hazirlanamadi ({type(e).__name__}: {e})",
                                    metadata={"settings": self.generation_settings(),
                                              "elapsed_seconds": time.perf_counter() - started})
        result = self._generate_result(prompt, started)
        if self.constrain_calls and not result.error and not result.truncated:
            return self._repair_calls(prompt, result)
        return result

    def _repair_calls(self, prompt: str, result: GenerationResult) -> GenerationResult:
        """Ilk gecersiz arac blogunu kisitli cozumlemeyle yeniden uret; basarisizsa orijinali koru."""
        from agent.constrained import call_schema, is_valid_repair, repair_prefix, splice_repaired_call
        prefix = repair_prefix(result.text, allow_unknown_names=self.repair_unknown_names)
        if prefix is None:
            return result
        repaired = self._generate_result(prompt + prefix, time.perf_counter(),
                                         fmt=call_schema(self.constrain_tools))
        applied = not repaired.error and not repaired.truncated and is_valid_repair(repaired.text)
        info = {"applied": applied, "original": result.text,
                "elapsed_seconds": repaired.metadata.get("elapsed_seconds")}
        if not applied:
            info["repair_text"] = repaired.text
            info["repair_error"] = repaired.error
            return GenerationResult(text=result.text, done_reason=result.done_reason,
                                    metadata={**result.metadata, "call_repair": info})
        return GenerationResult(text=splice_repaired_call(prefix, repaired.text), done_reason=result.done_reason,
                                metadata={**result.metadata, "call_repair": info})

    def _render(self, dicts: list[dict]) -> str:
        if self.renderer is not None:
            return self.renderer(dicts)
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.tokenizer_dir)
        return render_prompt(dicts, self._tokenizer, bos_token=self.bos_token)

    def _generate(self, prompt: str) -> str:
        result = self._generate_result(prompt, time.perf_counter())
        return result.display_text()

    def _generate_result(self, prompt: str, started: float, fmt: dict | None = None) -> GenerationResult:
        settings = self.generation_settings()
        fingerprint_text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        metadata = {"settings": settings, "model": self.model, "host": self.host,
                    "prompt_sha256": hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()}
        def failure(error, text=""):
            metadata["elapsed_seconds"] = time.perf_counter() - started
            return GenerationResult(text=text, error=error, metadata=metadata)
        payload = {"model": self.model, "prompt": prompt, "raw": True, "stream": False, "options": settings}
        if fmt is not None:
            payload["format"] = fmt
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # URLError alt-sinifi -> ONCE yakala
            if e.code == 404:
                modelfile = self.modelfile if self.model == DEFAULT_MODEL or self.modelfile != DEFAULT_MODELFILE else "<Modelfile>"
                return failure(f"HATA: '{self.model}' Ollama'da yok "
                               f"(once: ollama create {self.model} -f {modelfile})")
            return failure(f"HATA: Ollama HTTP {e.code}")
        except TimeoutError:
            return failure(f"HATA: Ollama zaman asimi ({self.timeout}s)")
        except (urllib.error.URLError, OSError) as e:
            return failure(f"HATA: Ollama'ya baglanilamadi ({e}); 'ollama serve' calisiyor mu?")
        except (ValueError, UnicodeError) as e:
            return failure(f"HATA: gecersiz Ollama yaniti ({e})")
        if not isinstance(data, dict):
            return failure("HATA: Ollama yaniti bir nesne olmali")
        metadata["response"] = {key: value for key, value in data.items() if key != "response"}
        metadata["elapsed_seconds"] = time.perf_counter() - started
        if data.get("error"):
            return failure(f"HATA: Ollama: {data['error']}",
                           data.get("response") if isinstance(data.get("response"), str) else "")
        if not isinstance(data.get("response"), str):
            return failure("HATA: Ollama yanit metni eksik/gecersiz")
        truncated = data.get("done_reason") == "length" or data.get("done") is False
        return GenerationResult(text=data["response"], done_reason=data.get("done_reason"),
                                truncated=truncated, metadata=metadata)
