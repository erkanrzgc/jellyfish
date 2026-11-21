"""Explicit local model contracts; selecting a profile never downloads or promotes it."""
from dataclasses import dataclass

from agent.config import DEFAULT_MODEL, DEFAULT_MODELFILE, DEFAULT_TOKENIZER_DIR


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    tokenizer_dir: str
    modelfile: str
    weights_path: str
    eos_token: str
    stop_tokens: tuple[str, ...]
    bos_token: str | None
    native_tool_roles: bool
    source_model: str


_PROFILES = {
    "octopus-v10": ModelProfile(
        "octopus-v10", DEFAULT_MODEL, DEFAULT_TOKENIZER_DIR, DEFAULT_MODELFILE,
        "models/octopus-v10-Q4_K_M.gguf", "<eos>", ("<end_of_turn>",), "<bos>", False,
        "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"),
    "qwen3-4b-instruct-2507": ModelProfile(
        "qwen3-4b-instruct-2507", "qwen3-4b-instruct-2507",
        "models/qwen3-4b-instruct-2507-tokenizer", "models/Modelfile.qwen3-4b-instruct-2507",
        "models/qwen3-4b-instruct-2507-Q4_K_M.gguf", "<|im_end|>", ("<|im_end|>",), None, True,
        "Qwen/Qwen3-4B-Instruct-2507"),
}


def get_profile(name: str) -> ModelProfile:
    try:
        return _PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown model profile: {name}") from None


def create_model(profile_name: str = DEFAULT_MODEL, **overrides):
    from agent.backends.gguf_model import GgufModel
    profile = get_profile(profile_name)
    settings = {key: getattr(profile, key) for key in (
        "model", "tokenizer_dir", "modelfile", "stop_tokens", "bos_token", "native_tool_roles")}
    settings.update(overrides)
    return GgufModel(**settings)
