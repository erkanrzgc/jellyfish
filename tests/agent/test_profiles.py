import pytest

from agent import profiles


def test_profiles_keep_default_and_separate_templates():
    gemma = profiles.get_profile("octopus-v10")
    qwen = profiles.get_profile("qwen3-4b-instruct-2507")
    assert gemma.model == "octopus-v10"
    assert gemma.tokenizer_dir == "models/octopus-v8-tokenizer"
    assert gemma.bos_token == "<bos>" and not gemma.native_tool_roles
    assert gemma.eos_token == "<eos>"
    assert qwen.eos_token == "<|im_end|>"
    assert qwen.bos_token is None and qwen.native_tool_roles
    assert "<end_of_turn>" not in qwen.stop_tokens
    assert qwen.tokenizer_dir != gemma.tokenizer_dir
    assert gemma.weights_path.endswith("octopus-v10-Q4_K_M.gguf")
    with pytest.raises(ValueError, match="Unknown"):
        profiles.get_profile("v12")


def test_factory_copies_profile_and_accepts_generation_overrides():
    model = profiles.create_model("qwen3-4b-instruct-2507", temperature=0, num_predict=1024)
    assert model.model == "qwen3-4b-instruct-2507"
    assert model.temperature == 0 and model.num_predict == 1024
    assert model.bos_token is None and model.native_tool_roles
    assert model.stop_tokens == ("<|im_end|>",)
    assert model.seed == 3407 and model.num_ctx == 4096
