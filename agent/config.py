"""Runtime defaults shared by demos, inference, and evaluation runners."""

# Product default since 2026-11-22: stock Gemma 4 E4B with native tool calling (agent.backends.native_chat).
# Bake-off: eval/reports/gemma4_e4b_bakeoff_2026-11-22.md.
DEFAULT_MODEL = "gemma4:e4b"
NATIVE_MODELS = frozenset({"gemma4:e4b", "gemma4:12b", "qwen3:8b"})

# Legacy fine-tuned model (raw-prompt ```arac``` path, agent.backends.gguf_model). The tokenizer and Modelfile
# below belong to it: v1.0 was evaluated with the v0.8 training chat template; see the v1.0 reports.
LEGACY_MODEL = "jellyfish-v10"
DEFAULT_TOKENIZER_DIR = "models/jellyfish-v8-tokenizer"
DEFAULT_MODELFILE = "models/Modelfile.v10"


def is_native(model: str) -> bool:
    """True for stock instruction models driven through Ollama native tool calling."""
    return model in NATIVE_MODELS or model.split(":", 1)[0] in {"gemma4", "qwen3"}
