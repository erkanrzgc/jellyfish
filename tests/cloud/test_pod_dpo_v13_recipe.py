from pathlib import Path

RECIPE = Path("cloud/pod_dpo_v13.sh").read_text(encoding="utf-8")
CODE = "\n".join(line for line in RECIPE.splitlines() if not line.lstrip().startswith("#"))


def test_recipe_stages_both_adapters_for_scp():
    assert 'cp -r "$SFT" "$OUT/' in RECIPE and 'cp -r "$DPO" "$OUT/' in RECIPE


def test_recipe_has_a_self_stop_watchdog_before_any_training():
    assert CODE.index("runpodctl stop pod") < CODE.index("python train/sft_bf16.py")


def test_recipe_runs_the_dpo_preflight_before_training():
    assert RECIPE.index("--check-only") < RECIPE.rindex("train/dpo_bf16.py")


def test_recipe_never_handles_an_hf_token():
    assert "HF_TOKEN" not in CODE and "huggingface-cli login" not in CODE
    assert "upload" not in CODE.lower()


def test_recipe_uses_the_v10_sft_recipe():
    assert "--max-train 0 --max-steps 1570" in RECIPE
