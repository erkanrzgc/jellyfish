"""The baseline runner never executes tools; all model/service access is faked here."""
import hashlib
import json
from dataclasses import replace

import pytest

from agent.backends.gguf_model import GenerationResult
from agent.profiles import get_profile
from eval import reliability_eval as runner


@pytest.fixture
def local(tmp_path):
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"actual local weights")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text('{"chat_template":"local"}')
    (tokenizer / "tokenizer.json").write_text('{}')
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text(f'FROM "{weights}"\n')
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"suite_id": "test-dev", "review_status": "draft", "cases": [
        {"id": "T1", "prompt": "PROMPT ONLY", "reference_answer": "SECRET REFERENCE",
         "rubric": ["SECRET RUBRIC"], "critical": True, "category": "tooling",
         "allow_tools": True, "expected_tools": ["nmap"], "expected_params": {"hedef": "10.0.0.1"}},
        {"id": "T2", "prompt": "SECOND", "critical": False, "category": "fundamentals",
         "allow_tools": False, "expected_tools": [], "expected_params": {}}]}))
    profile = replace(get_profile("jellyfish-v10"), weights_path=str(weights),
                      tokenizer_dir=str(tokenizer), modelfile=str(modelfile))
    return profile, suite, tmp_path / "report.json"


def configure(monkeypatch, profile, outputs):
    seen = []
    class Model:
        system_prompt = "SYSTEM"
        seed = 3407
        def generation_settings(self):
            return {"temperature": 0, "num_ctx": 4096, "num_predict": 1024, "seed": self.seed}
        def generate_result(self, messages):
            seen.append((self.seed, messages))
            return outputs[len(seen) - 1]
    monkeypatch.setattr(runner, "create_model", lambda *args, **kwargs: Model())
    monkeypatch.setattr(runner, "check_service", lambda *args: {"version": "fake", "weight_binding": "verified"})
    return seen


def test_live_report_all_calls_no_reference_leak_and_pending_semantics(monkeypatch, local):
    profile, suite, out = local
    text = '```arac {"arac":"nmap","parametreler":{"hedef":"10.0.0.1"}}```\n```arac {"arac":"httpx","parametreler":{}}```'
    seen = configure(monkeypatch, profile, [GenerationResult(text=text, metadata={"eval_count": 18})])
    assert runner.run_evaluation(profile, suite, out, limit=1) == 0
    report = json.loads(out.read_text())
    result = report["results"][0]
    assert len(result["calls"]) == 2 and result["raw_reply"] == text
    assert result["structural"]["expected_first_tool_match"] is True
    assert result["semantic_scores"] is None and result["review_status"] == "pending-human-review"
    assert [m.content for m in seen[0][1]] == ["PROMPT ONLY"]
    assert report["manifest"]["weights"]["sha256"] == hashlib.sha256(b"actual local weights").hexdigest()
    assert report["manifest"]["system_prompt"]["text"] == "SYSTEM"
    assert report["summary"]["semantic_scores"] is None
    assert report["execution_mode"] == "model-only-no-tools"


def test_manifest_only_never_contacts_service_or_generates(monkeypatch, local):
    profile, suite, out = local
    configure(monkeypatch, profile, [])
    monkeypatch.setattr(runner, "check_service", lambda *args: pytest.fail("service access"))
    assert runner.run_evaluation(profile, suite, out, manifest_only=True) == 0
    report = json.loads(out.read_text())
    assert report["status"] == "manifest-only" and not report["results"]
    assert report["manifest"]["tokenizer"]["files"]


def test_missing_artifact_blocks_no_quality_score(monkeypatch, local):
    profile, suite, out = local
    configure(monkeypatch, profile, [])
    profile = replace(profile, weights_path=str(out.parent / "missing.gguf"))
    assert runner.run_evaluation(profile, suite, out) != 0
    report = json.loads(out.read_text())
    assert report["status"] == "blocked-infrastructure"
    assert report["summary"]["structural"] is None and not report["results"]


def test_service_error_checkpoint_nonzero_and_no_fallback(monkeypatch, local):
    profile, suite, out = local
    configure(monkeypatch, profile, [])
    def unavailable(*args):
        raise OSError("refused")
    monkeypatch.setattr(runner, "check_service", unavailable)
    assert runner.run_evaluation(profile, suite, out) != 0
    assert json.loads(out.read_text())["status"] == "blocked-infrastructure"


def test_error_response_is_checkpointed_without_quality_metrics(monkeypatch, local):
    profile, suite, out = local
    configure(monkeypatch, profile, [GenerationResult(error="HATA: refused")])
    assert runner.run_evaluation(profile, suite, out) != 0
    report = json.loads(out.read_text())
    assert report["results"][0]["error"] == "HATA: refused"
    assert report["results"][0]["structural"] is None
    assert report["summary"]["structural"] is None


def test_repeat_seeds_and_critical_filter_checkpoint_each_response(monkeypatch, local):
    profile, suite, out = local
    seen = configure(monkeypatch, profile, [GenerationResult(text="ok") for _ in range(3)])
    original = runner.atomic_checkpoint
    checkpoints = []
    def checkpoint(path, report):
        original(path, report)
        checkpoints.append(len(json.loads(path.read_text())["results"]))
    monkeypatch.setattr(runner, "atomic_checkpoint", checkpoint)
    assert runner.run_evaluation(profile, suite, out, repeats=3, temperature=0.6, critical_only=True) == 0
    assert [seed for seed, _ in seen] == [3407, 3408, 3409]
    assert all([m.content for m in messages] == ["PROMPT ONLY"] for _, messages in seen)
    assert {1, 2, 3} <= set(checkpoints)


def test_refuse_overwrite_before_any_service_call(monkeypatch, local):
    profile, suite, out = local
    out.write_text("existing")
    monkeypatch.setattr(runner, "create_model", lambda *args, **kwargs: pytest.fail("model creation"))
    with pytest.raises(FileExistsError):
        runner.run_evaluation(profile, suite, out)
    assert out.read_text() == "existing"


def test_actual_service_model_weight_mismatch_blocks(monkeypatch, local):
    profile, _, _ = local
    responses = iter([
        {"version": "0.test"},
        {"models": [{"name": "jellyfish-v10:latest", "digest": "manifest-sha"}]},
        {"modelfile": "FROM /blobs/sha256-" + "a" * 64, "parameters": "", "template": ""},
    ])
    monkeypatch.setattr(runner, "get_ollama_json", lambda *args: next(responses))
    with pytest.raises(ValueError, match="weight"):
        runner.check_service("http://localhost:11434", profile, "b" * 64)


def test_unexpected_generation_exception_preserves_attempt_record(monkeypatch, local):
    profile, suite, out = local
    configure(monkeypatch, profile, [])
    class Broken:
        system_prompt = "S"
        seed = 3407
        def generation_settings(self):
            return {}
        def generate_result(self, messages):
            raise OSError("read failed")
    monkeypatch.setattr(runner, "create_model", lambda *args, **kwargs: Broken())
    assert runner.run_evaluation(profile, suite, out) == 2
    report = json.loads(out.read_text())
    assert report["results"][0]["case_id"] == "T1"
    assert "read failed" in report["results"][0]["error"]


def test_service_binding_success_retains_actual_server_manifest(monkeypatch, local):
    profile, _, _ = local
    digest = "b" * 64
    responses = iter([{"version": "0.test"},
                      {"models": [{"name": "jellyfish-v10:latest", "digest": "tag-manifest"}]},
                      {"modelfile": f'# generated\nFROM "C:\\blobs\\sha256-{digest}"\n'}])
    monkeypatch.setattr(runner, "get_ollama_json", lambda *args: next(responses))
    record = runner.check_service("http://localhost:11434", profile, digest)
    assert record["weight_binding"] == "verified"
    assert record["installed_model"]["digest"] == "tag-manifest"


def test_malformed_call_is_parse_error_not_silent_no_tool_pass():
    calls, errors, metrics = runner.structural_metrics(
        {"allow_tools": False, "expected_tools": []}, '```arac {"arac":"nmap"}')
    assert calls == [] and errors
    assert metrics["parse_valid"] is False
    assert metrics["no_unnecessary_tools"] is False


def test_expected_first_tool_matches_one_acceptable_alternative():
    _, _, metrics = runner.structural_metrics(
        {"allow_tools": True, "expected_tools": ["nmap", "rustscan"], "expected_params": {}},
        '```arac {"arac":"rustscan","parametreler":{}}```')
    assert metrics["expected_first_tool_match"] is True


def test_expected_parameter_subset_checks_relevant_first_call_only():
    _, _, metrics = runner.structural_metrics(
        {"allow_tools": True, "expected_tools": ["nmap", "rustscan"],
         "expected_params": {"hedef": "10.0.0.1"}},
        '```arac {"arac":"nmap","parametreler":{"hedef":"10.0.0.1"}}```'
        '\n```arac {"arac":"httpx","parametreler":{}}```')
    assert metrics["expected_first_tool_match"] is True
    assert metrics["expected_parameter_subset_match"] is True
    assert metrics["call_count"] == 2


def test_first_tool_and_parameters_not_applicable_without_expectation():
    _, _, metrics = runner.structural_metrics(
        {"allow_tools": False, "expected_tools": [], "expected_params": {}}, "explanation")
    assert metrics["expected_first_tool_match"] is None
    assert metrics["expected_parameter_subset_match"] is None
    assert metrics["no_unnecessary_tools"] is True


def test_missing_or_wrong_first_tool_fails_required_parameters():
    case = {"allow_tools": True, "expected_tools": ["nmap"], "expected_params": {"hedef": "10.0.0.1"}}
    for text in ("no call", '```arac {"arac":"httpx","parametreler":{"hedef":"10.0.0.1"}}```'):
        _, _, metrics = runner.structural_metrics(case, text)
        assert metrics["expected_first_tool_match"] is False
        assert metrics["expected_parameter_subset_match"] is False


def test_truncated_response_never_counts_as_success(monkeypatch, local):
    profile, suite, out = local
    configure(monkeypatch, profile, [GenerationResult(text="partial", truncated=True, done_reason="length")])
    assert runner.run_evaluation(profile, suite, out, limit=1) == 2
    report = json.loads(out.read_text())
    assert report["results"][0]["structural"] is None
    assert report["summary"]["truncated"] == 1
    assert report["status"] == "completed-with-invalid-responses"
