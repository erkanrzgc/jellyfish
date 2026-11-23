"""Explicit local model contracts; selecting a profile never downloads or promotes it."""
from dataclasses import dataclass

from agent.config import DEFAULT_MODELFILE, DEFAULT_TOKENIZER_DIR, LEGACY_MODEL as DEFAULT_MODEL


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


_NATIVE_OVERRIDES = ("system_prompt", "temperature", "seed", "num_predict", "host", "timeout", "tools_enabled")


def create_model(profile_name: str = DEFAULT_MODEL, **overrides):
    """Profiles of stock models (agent.config.is_native) get NativeChatModel; the rest are GGUF profiles."""
    from agent.config import is_native
    if is_native(profile_name):
        from agent.backends.native_chat import NativeChatModel
        # GGUF-only settings (tokenizer, num_ctx 4096...) do not apply: the native tool schema needs ~16k ctx.
        return NativeChatModel(model=profile_name,
                               **{key: value for key, value in overrides.items() if key in _NATIVE_OVERRIDES})
    from agent.backends.gguf_model import GgufModel
    profile = get_profile(profile_name)
    settings = {key: getattr(profile, key) for key in (
        "model", "tokenizer_dir", "modelfile", "stop_tokens", "bos_token", "native_tool_roles")}
    settings.update({key: value for key, value in overrides.items() if key != "tools_enabled"})
    return GgufModel(**settings)
