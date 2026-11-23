"""Local conversation UI with advisory and host-approved tool modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from agent.audit import AuditLog
from agent.execution import ExecutionResult
from agent.executor import MockExecutor
from agent.loop import ToolLoopResult, run_tool_loop
from agent.messages import Message
from agent.policy import LabPolicy
from agent.registry import ToolRegistry
from agent.toolcall import strip_dusunce
from data.sft.persona import OCTOPUS_SYSTEM_PROMPT, OCTOPUS_TOOL_SYSTEM_PROMPT


_ADVISORY_RULE = (
    "\nBu oturum danisman modundadir. Yalniz aciklama, komut onerisi ve saglanan kanitin "
    "analizini yap. Hicbir aracin calistigini iddia etme. Gozlenen bulguyu, hipotezi ve "
    "onerilen sonraki kontrolu ayir. Kanit yoksa belirsizligi belirt."
)
ADVISORY_PROMPT = OCTOPUS_SYSTEM_PROMPT + _ADVISORY_RULE
ASSIST_PROMPT = OCTOPUS_TOOL_SYSTEM_PROMPT.replace(
    "Araçları doğrudan çalıştırırsın;", "Arac cagrilari yalniz oneridir; gercek islemler kullanici onayi bekler;"
) + "\nGozlenen bulgu, hipotez ve onerilen sonraki kontrolu ayir. Reddedilen/hata veren islemi tamamlandi sayma."


def prompts_for(profile: str) -> tuple[str, str]:
    """(advise, assist) system prompts: Jellyfish for native stock models, the trained v1.0 text otherwise."""
    from agent.config import is_native
    if not is_native(profile):
        return ADVISORY_PROMPT, ASSIST_PROMPT
    from agent.backends.native_chat import NATIVE_SYSTEM_PROMPT
    from agent.persona import JELLYFISH_SYSTEM_PROMPT
    return JELLYFISH_SYSTEM_PROMPT + _ADVISORY_RULE, NATIVE_SYSTEM_PROMPT + (
        "\nArac cagrilari yalniz oneridir; gercek islemler kullanici onayi bekler. Gozlenen bulgu, hipotez ve "
        "onerilen sonraki kontrolu ayir. Reddedilen/hata veren islemi tamamlandi sayma.")


def approval_prompt(registry: ToolRegistry, *, read=input, write=print):
    def approve(result: ExecutionResult) -> str | None:
        write("\n[ONAY BEKLIYOR] " + result.call.name)
        write(json.dumps({"parametreler": result.call.params, "komut": result.command},
                         ensure_ascii=False, indent=2))
        try:
            answer = read("Yalniz bu islemi onayliyor musun? [evet/hayir]: ").strip().casefold()
            if answer not in {"evet", "yes"} or not result.proposal_id:
                return None
            return registry.approve(result.proposal_id)
        except (EOFError, KeyboardInterrupt, ValueError):
            return None
    return approve


def run_turn(messages, generate, registry, *, mode="advise", approval_handler=None,
             max_steps=10) -> ToolLoopResult:
    if mode == "assist":
        return run_tool_loop(messages, generate, registry, max_steps=max_steps,
                             approval_handler=approval_handler)
    if mode != "advise":
        raise ValueError("bilinmeyen mod")
    try:
        reply = generate(messages)
    except Exception as error:
        return ToolLoopResult(f"HATA: model ({type(error).__name__})", 1, status="failed")
    messages.append(Message("assistant", reply))
    if not isinstance(reply, str) or reply.startswith("HATA"):
        return ToolLoopResult(str(reply), 1, status="failed")
    return ToolLoopResult("[Danisman modu: arac calistirilmadi]\n" + strip_dusunce(reply), 1,
                          status="advisory")


def load_session(path: Path, profile: str) -> list[Message]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("profile") != profile:
        raise ValueError("gecersiz oturum veya profil uyusmazligi")
    if not isinstance(data.get("messages"), list):
        raise ValueError("gecersiz oturum mesajlari")
    messages = []
    for item in data["messages"]:
        if (not isinstance(item, dict) or set(item) != {"role", "content"}
                or item["role"] not in {"user", "assistant", "tool"}
                or not isinstance(item["content"], str)):
            raise ValueError("gecersiz oturum mesaji")
        messages.append(Message(item["role"], item["content"]))
    return messages


def save_session(path: Path, profile: str, messages: list[Message]) -> None:
    if path.exists():
        load_session(path, profile)  # Never clobber an unrelated user file.
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "profile": profile,
               "messages": [m.to_dict() for m in messages]}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".octopus-session-", suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    from agent.config import DEFAULT_MODEL
    from agent.profiles import create_model
    from agent.backends.assistant_executor import AssistantExecutor
    from agent.composite_executor import CompositeExecutor
    parser = argparse.ArgumentParser(description="Jellyfish local conversation")
    parser.add_argument("--mode", choices=["advise", "assist"], default="advise")
    parser.add_argument("--profile", default=DEFAULT_MODEL)
    parser.add_argument("--executor", choices=["mock", "kali", "docker"], default="mock")
    parser.add_argument("--scope", nargs="*", default=[])
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--session", type=Path)
    parser.add_argument("--ask")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    try:
        messages = load_session(args.session, args.profile) if args.session and args.session.exists() else []
        advise_prompt, assist_prompt = prompts_for(args.profile)
        model = create_model(args.profile, system_prompt=advise_prompt if args.mode == "advise" else assist_prompt,
                             num_ctx=4096, num_predict=1024, tools_enabled=args.mode == "assist")
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"HATA: {error}")
        return 2
    security = MockExecutor()
    if args.executor == "kali":
        from agent.backends.real_executor import RealExecutor
        security = RealExecutor()
    elif args.executor == "docker":
        from agent.backends.docker_executor import DockerExecutor
        security = DockerExecutor()
    workspace = str(args.workspace.resolve())
    registry = ToolRegistry(LabPolicy(scope=args.scope, workspace_root=workspace),
                            CompositeExecutor(security, AssistantExecutor(workspace)), AuditLog.default())
    print(f"Jellyfish | profil={args.profile} | mod={args.mode} | guvenlik executor={args.executor}")
    if args.executor == "mock" and args.mode == "assist":
        print("Guvenlik taramalari SIMULASYON; dosya islemleri gercek ve onaylidir.")
    while True:
        try:
            question = args.ask if args.ask is not None else input("\nSen (/quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if not question.strip() or question.strip() == "/quit":
            break
        messages.append(Message("user", question))
        result = run_turn(messages, model, registry, mode=args.mode,
                          approval_handler=approval_prompt(registry), max_steps=args.max_steps)
        print(result.final)
        print(f"[durum={result.status}; calisan cagri={len(result.calls)}]")
        if args.session:
            save_session(args.session, args.profile, messages)
        if args.ask is not None:
            return 0 if result.status in {"completed", "advisory"} else 2
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
