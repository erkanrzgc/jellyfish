"""Model<->arac dongusu (Hermes recursive loop deseni, agentic-model'den uyarlandi).
model uret -> arac blogu var mi? yoksa nihai cevap; varsa (skill katmani acikken) yeni
arac icin once skill enjekte et + regenerate, sonra calistir, sonucu tool mesaji olarak
ekle, tekrarla. max_steps sonsuz-dongu korumasi."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from agent.incident import looks_like_incident
from agent.messages import Message
from agent.registry import ToolRegistry
from agent.skills import SkillLibrary
from agent.toolcall import ToolCall, parse_tool_reply, strip_dusunce
from agent.execution import ExecutionResult

Generate = Callable[[list[Message]], str]

# Mingbird "completion check" analogue: an incident answered without any evidence gets ONE push to call a
# tool itself. Opt-in: on the 2026-11-22 measurement gemma4:e4b RCA went 0/5 -> 5/5, but 3 DPO-val
# "interpret this pasted log" cases (no tool expected) became unnecessary calls.
EVIDENCE_PUSH = ("Henüz hiçbir kanıt toplanmadı. Kullanıcıya komut listesi verme: şimdi verilen araçlardan "
                 "uygun olanı (ör. run_cmd, read_file, grep) KENDİN çağırarak ilk kanıtı topla.")


@dataclass
class ToolLoopResult:
    final: str
    steps: int
    calls: list[ToolCall] = field(default_factory=list)
    attempts: list[ExecutionResult] = field(default_factory=list)
    status: str = "completed"
    pending: list[ToolCall] = field(default_factory=list)


def _inject_tool_skills(
    messages: list[Message], calls: list[ToolCall],
    skills: SkillLibrary, shown: set[str],
) -> bool:
    """Cagrilan araclardan skill'i olan + bu konusmada henuz gosterilmemis olanlar icin
    skill md'sini USER turu olarak ekle (once oku). Cache: her arac en fazla 1 kez.
    En az bir enjeksiyon yapildiysa True (dongu regenerate etsin)."""
    injected = False
    for call in calls:
        if call.name in shown:
            continue
        skill = skills.match_tool(call.name)
        if skill is None:
            continue
        messages.append(Message("user", skills.tool_skill_injection(skill)))
        shown.add(call.name)
        injected = True
    return injected


def run_tool_loop(
    messages: list[Message],
    generate: Generate,
    registry: ToolRegistry,
    *,
    max_steps: int = 10,
    skills: SkillLibrary | None = None,
    approval_handler: Callable[[ExecutionResult], str | None] | None = None,
    evidence_push: bool = False,
) -> ToolLoopResult:
    if max_steps < 1:
        raise ValueError("max_steps en az 1 olmali")
    executed: list[ToolCall] = []
    attempts: list[ExecutionResult] = []
    shown: set[str] = set()
    repairs = 0
    task = next((m.content for m in messages if m.role == "user"), "")
    may_push = evidence_push and looks_like_incident(task)

    def stopped(status, detail, step, pending=None):
        return ToolLoopResult("Gorev tamamlanmadi: " + detail, step, executed,
                              attempts, status, pending or [])

    for step in range(max_steps):
        try:
            reply = generate(messages)
        except Exception as error:
            return stopped("failed", f"model hatasi ({type(error).__name__})", step + 1)
        messages.append(Message("assistant", reply))
        if not isinstance(reply, str) or reply.startswith("HATA"):
            return stopped("failed", str(reply), step + 1)
        parsed = parse_tool_reply(reply, validate_catalog=True)
        if parsed.errors:
            if repairs:
                return stopped("failed", "; ".join(parsed.errors), step + 1)
            repairs += 1
            messages.append(Message("tool", "GECERSIZ CAGRI: " + "; ".join(parsed.errors)
                                    + ". Bir kez gecerli arac/parametreler JSON'u ile duzelt; hicbir arac calismadi."))
            continue
        calls = parsed.calls
        if not calls and may_push and not executed:
            may_push = False
            messages.append(Message("user", EVIDENCE_PUSH))
            continue
        if not calls:
            return ToolLoopResult(final=strip_dusunce(reply), steps=step + 1,
                                  calls=executed, attempts=attempts)
        if skills is not None and _inject_tool_skills(messages, calls, skills, shown):
            continue          # skill eklendi -> model cagriyi duzeltsin diye yeniden uret
        for index, call in enumerate(calls):
            result = registry.invoke_result(call)
            attempts.append(result)
            if result.status == "awaiting_approval" and approval_handler is not None:
                token = approval_handler(result)
                if token:
                    result = registry.invoke_result(call, approval_token=token)
                    attempts.append(result)
            messages.append(Message("tool", result.output))
            if result.status != "executed":
                return stopped(result.status, result.output, step + 1,
                               calls[index:] if result.status == "awaiting_approval" else [])
            executed.append(result.call)
    return stopped("incomplete", "adim sinirina ulasildi", max_steps)
