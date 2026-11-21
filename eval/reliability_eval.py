"""Auditable model-only development baseline. This module has no tool executor.

Generation metadata follows https://docs.ollama.com/api/generate. Semantic
rubrics always require human review; structural checks are not quality scores.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import sys
import platform
from importlib.metadata import version, PackageNotFoundError
import urllib.request

from agent.config import DEFAULT_MODEL
from agent.backends.gguf_model import GenerationResult
from agent.messages import Message
from agent.profiles import ModelProfile, create_model, get_profile
from agent.toolcall import parse_tool_reply


def file_snapshot(path: Path) -> dict:
    record = {"path": str(path.resolve()), "exists": path.is_file()}
    if not record["exists"]:
        return record
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"Artifact changed while hashing: {path}")
    record.update(sha256=digest.hexdigest(), size_bytes=after.st_size)
    return record


def build_manifest(profile: ModelProfile, model, suite_path: Path) -> tuple[dict, list[str]]:
    weights = file_snapshot(Path(profile.weights_path))
    modelfile = file_snapshot(Path(profile.modelfile))
    if modelfile["exists"]:
        modelfile["text"] = Path(profile.modelfile).read_text(encoding="utf-8")
    directory = Path(profile.tokenizer_dir)
    files = [file_snapshot(path) for path in sorted(directory.rglob("*")) if path.is_file()]
    # Retain the actual template and special-token configuration without loading a tokenizer.
    configs = {path.name: path.read_text(encoding="utf-8") for path in (
        directory / "tokenizer_config.json", directory / "special_tokens_map.json") if path.is_file()}
    blockers = []
    if not weights["exists"]:
        blockers.append(f"Missing local weights: {profile.weights_path}")
    if not modelfile["exists"]:
        blockers.append(f"Missing Modelfile: {profile.modelfile}")
    if not (directory / "tokenizer_config.json").is_file() or not any(
            (directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        blockers.append(f"Missing local tokenizer artifacts: {profile.tokenizer_dir}")
    prompt = model.system_prompt
    packages = {}
    for name in ("transformers", "tokenizers", "sentencepiece", "jinja2"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "profile": asdict(profile), "weights": weights, "modelfile": modelfile,
        "tokenizer": {"path": str(directory.resolve()), "files": files, "configuration": configs},
        "system_prompt": {"text": prompt, "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()},
        "generation_settings": model.generation_settings(), "suite": file_snapshot(suite_path),
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": packages},
        "implementation": [file_snapshot(Path(path)) for path in (
            "agent/profiles.py", "agent/backends/gguf_model.py", "agent/toolcall.py",
            "eval/reliability_eval.py", "uv.lock")],
        "provenance_note": "Hashes identify current local artifacts only; no historical training provenance is inferred.",
    }, blockers


def get_ollama_json(host: str, endpoint: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(host.rstrip("/") + endpoint,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError(f"Invalid Ollama {endpoint} response: {payload}")
    return payload


def check_service(host: str, profile: ModelProfile, weights_sha256: str) -> dict:
    version = get_ollama_json(host, "/api/version")
    tags = get_ollama_json(host, "/api/tags")
    names = {profile.model, profile.model + ":latest"}
    installed = next((item for item in tags.get("models", []) if item.get("name") in names), None)
    if installed is None:
        raise ValueError(f"Model not installed in Ollama: {profile.model}")
    show = get_ollama_json(host, "/api/show", {"model": profile.model})
    # /api/show returns the server's generated Modelfile. Bind its actual weight
    # blob to our independently hashed local GGUF; a tag's manifest digest alone
    # does not establish weight identity.
    match = re.search(r'^FROM\s+"?(.+?)"?\s*$', show.get("modelfile", ""), re.MULTILINE | re.IGNORECASE)
    blob = re.search(r"sha256[-:]([a-fA-F0-9]{64})(?:\s|\"|$)", match.group(1)) if match else None
    if blob is None or blob.group(1).lower() != weights_sha256:
        raise ValueError("Cannot verify served model weight hash against the local profile GGUF")
    return {"version": version, "installed_model": installed, "show": show, "weight_binding": "verified"}


def atomic_checkpoint(path: Path, report: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def structural_metrics(case: dict, text: str) -> tuple[list[dict], list[str], dict]:
    parsed = parse_tool_reply(text)
    checked = parse_tool_reply(text, validate_catalog=True)
    calls = [asdict(call) for call in parsed.calls]
    expected = case.get("expected_tools", [])
    first_matches = bool(parsed.calls and parsed.calls[0].name in expected and not parsed.errors) if expected else None
    return calls, parsed.errors, {
        "call_count": len(calls), "parse_valid": not parsed.errors,
        "catalog_valid": not checked.errors, "catalog_errors": checked.errors,
        "expected_first_tool_match": first_matches,
        "no_unnecessary_tools": not calls and not parsed.errors if not case.get("allow_tools") else None,
        "expected_parameter_subset_match": (
            bool(first_matches and all(parsed.calls[0].params.get(key) == value
                                       for key, value in case["expected_params"].items()))
            if expected and case.get("expected_params") else None),
    }


def summarize(results: list[dict], *, infrastructure_blocked: bool = False) -> dict:
    valid = [row for row in results if row["structural"] is not None]
    metrics = None
    if valid and not infrastructure_blocked:
        metrics = {}
        for key in ("parse_valid", "catalog_valid", "expected_first_tool_match", "no_unnecessary_tools",
                    "expected_parameter_subset_match"):
            values = [row["structural"][key] for row in valid if row["structural"][key] is not None]
            metrics[key] = {"passed": sum(values), "denominator": len(values)}
    return {"responses": len(results), "errors": sum(bool(row["error"]) for row in results),
            "truncated": sum(row["truncated"] for row in results),
            "structural": metrics, "semantic_scores": None, "review_status": "pending-human-review"}


def run_evaluation(profile: ModelProfile, suite_path: Path, out: Path, *, limit: int | None = None,
                   temperature: float = 0, repeats: int = 1, seed: int = 3407,
                   manifest_only: bool = False, critical_only: bool = False,
                   host: str = "http://localhost:11434") -> int:
    suite_path, out = Path(suite_path), Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {out}")
    if repeats < 1 or (limit is not None and limit < 1):
        raise ValueError("repeats and limit must be positive")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = [case for case in suite["cases"] if not critical_only or case.get("critical")]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No cases selected")
    model = create_model(profile.name, tokenizer_dir=profile.tokenizer_dir, modelfile=profile.modelfile,
                         temperature=temperature, num_ctx=4096, num_predict=1024, seed=seed, host=host)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents simultaneous runners from overwriting each other.
    with out.open("x", encoding="utf-8"):
        pass
    report = {"schema_version": 2, "status": "preparing", "execution_mode": "model-only-no-tools",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "suite_id": suite.get("suite_id"), "suite_review_status": suite.get("review_status"),
              "selection": {"case_ids": [case["id"] for case in cases], "limit": limit,
                            "critical_only": critical_only, "repeats": repeats,
                            "seeds": [seed + index for index in range(repeats)]},
              "manifest": {}, "infrastructure_errors": [], "results": [], "summary": summarize([])}
    try:
        report["manifest"], blockers = build_manifest(profile, model, suite_path)
        report["infrastructure_errors"] = blockers
        if manifest_only:
            report["status"] = "manifest-only"
            atomic_checkpoint(out, report)
            return 0
        if blockers:
            raise ValueError("; ".join(blockers))
        report["manifest"]["service"] = check_service(host, profile, report["manifest"]["weights"]["sha256"])
        report["status"] = "running"
        atomic_checkpoint(out, report)
        for repetition in range(repeats):
            model.seed = seed + repetition
            for case in cases:
                started = time.perf_counter()
                try:
                    result = model.generate_result([Message("user", case["prompt"])])
                except Exception as error:
                    result = GenerationResult(error=f"{type(error).__name__}: {error}",
                        metadata={"settings": model.generation_settings(),
                                  "elapsed_seconds": time.perf_counter() - started})
                calls, errors, structural = structural_metrics(case, result.text)
                row = {"case_id": case["id"], "category": case.get("category"), "prompt": case["prompt"],
                       "repetition": repetition + 1, "seed": model.seed, "raw_reply": result.text,
                       "calls": calls, "parse_errors": errors, "error": result.error,
                       "done_reason": result.done_reason, "truncated": result.truncated,
                       "metadata": result.metadata, "structural": structural if not result.error and not result.truncated else None,
                       "semantic_scores": None, "review_status": "pending-human-review"}
                report["results"].append(row)
                report["summary"] = summarize(report["results"], infrastructure_blocked=bool(result.error))
                if result.error:
                    report["status"] = "blocked-infrastructure"
                    report["infrastructure_errors"].append(result.error)
                atomic_checkpoint(out, report)
                if result.error:
                    return 2
        invalid = any(row["error"] or row["truncated"] for row in report["results"])
        report["status"] = "completed-with-invalid-responses" if invalid else "completed-pending-human-review"
        atomic_checkpoint(out, report)
        return 2 if invalid else 0
    except Exception as error:  # Preserve completed responses if infrastructure fails mid-run.
        report["status"] = "blocked-infrastructure"
        report["infrastructure_errors"].append(f"{type(error).__name__}: {error}")
        report["summary"] = summarize(report["results"], infrastructure_blocked=True)
        atomic_checkpoint(out, report)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_MODEL)
    parser.add_argument("--suite", type=Path, default=Path("eval/suites/reliability-dev-v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--critical-only", action="store_true")
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args(argv)
    try:
        profile = get_profile(args.profile)
        code = run_evaluation(profile, args.suite, args.out, limit=args.limit, temperature=args.temperature,
                              repeats=args.repeats, seed=args.seed, manifest_only=args.manifest_only,
                              critical_only=args.critical_only, host=args.host)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Report: {args.out} (exit {code}); semantic review pending")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
