"""Uctan uca demo/giris. Uc mod, tek dongü:
  (varsayilan) mock  : MockExecutor (sahte cikti, Windows'ta calisir)
  --real             : RealExecutor (Kali WSL'de gercek arac)
  --docker           : DockerExecutor (lab docker aginda container arac -> erisilebilir hedef)
Faz 2'de ScriptedModel yerine gercek GGUF modeli takilir, ayni dongu."""
from __future__ import annotations
import argparse
import re
import sys
from agent.backends.mock_model import ScriptedModel
from agent.config import DEFAULT_MODEL, DEFAULT_TOKENIZER_DIR
from agent.display import render_transcript
from agent.loop import run_tool_loop
from agent.messages import Message
from agent.registry import ToolRegistry


class DemoTranscript(str):
    """Preserve the public string API while carrying the real completion status."""
    def __new__(cls, text: str, status: str = "completed"):
        instance = super().__new__(cls, text)
        instance.status = status
        return instance


def _format_run(msgs: list[Message], result, label: str) -> str:
    """Döngü transcript'ini Claude Code tarzı render et + kısa özet altbilgisi (DRY: tüm demolar)."""
    status = result.status
    footer = f"(durum={status}, adim={result.steps}, cagri={len(result.calls)}) ({label if status == 'completed' else 'gorev tamamlanmadi'})"
    final = "" if msgs and msgs[-1].role == "assistant" and msgs[-1].content == result.final else "\n" + result.final
    return DemoTranscript(f"{render_transcript(msgs, color=sys.stdout.isatty())}{final}\n{footer}", status)

# Docker container-adi grameri: harf/rakam ile baslar, ardindan [A-Za-z0-9_.-]. Enjeksiyon
# kapisi — _docker_container_ip'te WSL komutuna yalniz bu deseni gecen ad girer.
_CONTAINER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

# skills parametresi icin sentinel: uc durum ayirt et — HIC gecilmedi -> SkillLibrary.load()
# (canli beyin yolu AKTIF), None -> acikca kapat (test/hiz), <SkillLibrary> -> belirli kutuphane.
_LOAD_SKILLS = object()


def _scan_demo(executor, target: str, scope: list[str], secenekler: str, label: str) -> str:
    """Tek scripted nmap turu: model arac blogu -> policy -> executor -> geri-besleme -> cevap."""
    from agent.policy import LabPolicy
    from agent.audit import AuditLog
    from agent.chat import approval_prompt
    registry = ToolRegistry(LabPolicy(scope=scope), executor, AuditLog.default())
    model = ScriptedModel([
        f'Yetkili lab hedefini tararim.\n```arac\n'
        f'{{"arac":"nmap","parametreler":{{"hedef":"{target}","secenekler":"{secenekler}"}}}}\n```',
        "Tarama tamamlandi, sonucu yukarida yorumladim.",
    ])
    msgs = [Message("user", f"{target} yetkili lab hedefini tara")]
    result = run_tool_loop(msgs, model, registry, approval_handler=approval_prompt(registry))
    return _format_run(msgs, result, label)


def run_demo(scope: list[str]) -> str:
    """Mock: gercek arac calistirmadan dongu (Windows'ta calisir)."""
    from agent.executor import MockExecutor
    return _scan_demo(MockExecutor(), "10.10.10.5", scope, "-sV", "demo bitti")


def run_real_demo(target: str = "127.0.0.1", scope: list[str] | None = None,
                  distro: str = "kali-linux", port: int = 8000) -> str:
    """GERCEK: RealExecutor Kali WSL'de gercek nmap (hedef Kali'den erisilebilir olmali)."""
    from agent.backends.real_executor import RealExecutor
    scope = scope or ["127.0.0.0/8"]
    return _scan_demo(RealExecutor(distro=distro), target, scope, f"-Pn -sV -p{port}", "gercek demo bitti")


def run_docker_demo(target: str = "octopus-target", scope: list[str] | None = None,
                    network: str = "octopus-lab", port: int = 80) -> str:
    """DOCKER LAB (Model B): arac lab aginda container -> hedefe DNS adiyla erisir."""
    from agent.backends.docker_executor import DockerExecutor
    scope = scope or [target, "172.30.0.0/24"]
    return _scan_demo(DockerExecutor(network=network), target, scope, f"-Pn -sV -p{port}", "docker demo bitti")


def _demo_registry(scope: list[str], executor, role: str | None = None) -> ToolRegistry:
    """Demo registry: role verilirse rol-scoped (YAPISAL containment — rol-disi arac calismaz),
    yoksa tam katalog (bugunku davranis). Olcum (2026-08-17): scoping DOGRULUGU artirmaz; degeri
    guvenlik/containment (opt-in), accuracy-default DEGIL. Rapor: eval/reports/subagent_scoping_*."""
    from agent.policy import LabPolicy
    from agent.audit import AuditLog
    pol = LabPolicy(scope=scope)
    if role:
        return ToolRegistry.scoped(role, policy=pol, executor=executor, audit=AuditLog.default())
    return ToolRegistry(pol, executor, AuditLog.default())


def run_gguf_demo(scope: list[str] | None = None, model: str = DEFAULT_MODEL,
                  executor=None, target: str = "10.10.10.5", label: str = "gguf demo bitti",
                  skills=_LOAD_SKILLS, role: str | None = None,
                  tokenizer_dir: str = DEFAULT_TOKENIZER_DIR) -> str:
    """GERCEK BEYIN: GgufModel (Ollama) dongüyü sürer. executor=None -> MockExecutor eller.
    executor verilirse (ör. DockerExecutor) gercek beyin + gercek eller uctan-uca calisir.
    Ollama yoksa GgufModel net HATA stringi döner -> döngü çökmez, onu nihai cevap alir.
    skills: HIC gecilmezse SkillLibrary.load() ile skill katmani AKTIF (post-call correction);
    None -> kapali (geriye-uyum/test); <SkillLibrary> -> belirli kutuphane enjekte et."""
    from agent.backends.gguf_model import GgufModel
    from agent.chat import approval_prompt
    if executor is None:
        from agent.executor import MockExecutor
        executor = MockExecutor()
    if skills is _LOAD_SKILLS:
        from agent.skills import SkillLibrary
        skills = SkillLibrary.load()
    scope = scope or ["10.10.10.0/24"]
    registry = _demo_registry(scope, executor, role)
    msgs = [Message("user", f"{target} yetkili lab hedefini tara")]
    # Sistem promptu iki kosullu katman ile kurulur (taban EGITIM-BIREBIR korunur, sona eklenir):
    #  1) Skill acikken: egitim-disi araclari (trufflehog/magika/ghunt) KESIF manifesti olarak ekle.
    #  2) Gorev bir sunucu-OLAYI ise: RCA "kanit-once-sonuc" kurali ekle (assemble_incident_aware).
    #     Kural GLOBAL uygulaninca toolcall'i bozar (dogru-arac 75->54) -> yalniz olay dilinde.
    from data.sft.persona import OCTOPUS_TOOL_SYSTEM_PROMPT
    from agent.incident import assemble_incident_aware_prompt
    task = msgs[0].content
    base = OCTOPUS_TOOL_SYSTEM_PROMPT
    if skills is not None:
        from agent.catalog import extension_augmented_system_prompt
        base = extension_augmented_system_prompt(base)
    system_prompt = assemble_incident_aware_prompt(base, task)
    gguf_kwargs: dict = {"model": model, "tokenizer_dir": tokenizer_dir}
    if system_prompt != OCTOPUS_TOOL_SYSTEM_PROMPT:   # taban degistiyse (manifest veya RCA kurali)
        gguf_kwargs["system_prompt"] = system_prompt
    result = run_tool_loop(msgs, GgufModel(**gguf_kwargs), registry, skills=skills,
                           approval_handler=approval_prompt(registry))
    return _format_run(msgs, result, label)


def _docker_container_ip(container: str, distro: str = "kali-linux") -> str | None:
    """Lab container'inin guncel IP'sini docker inspect ile al (None = bulunamadi).
    NEDEN IP: Docker embedded DNS (isimle cozme) WSL'de flaky olabilir; model zaten
    prompt'taki ACIK IP'yi arac blogunda birebir onurlandirir (hostname'i ise kacirir).
    GUVENLIK: shell YOK — argv listesi (wsl.exe -> docker inspect ...) + container adi
    allowlist ile dogrulanir (enjeksiyon kapali)."""
    import subprocess
    if not _CONTAINER_NAME_RE.match(container):     # gecersiz/kotucul ad -> hic komut kurma
        return None
    try:
        out = subprocess.run(
            ["wsl.exe", "-d", distro, "--", "docker", "inspect", container,
             "--format", "{{json .NetworkSettings.Networks}}"],
            capture_output=True, text=True, timeout=30).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    m = re.search(r'"IPAddress":"([0-9.]+)"', out)
    return m.group(1) if m else None


def run_gguf_docker_demo(scope: list[str] | None = None, model: str = DEFAULT_MODEL,
                         target: str = "octopus-target", network: str = "octopus-lab",
                         tokenizer_dir: str = DEFAULT_TOKENIZER_DIR) -> str:
    """UCTAN UCA (Model B): GERCEK model + GERCEK docker-lab eli (container nmap).
    Model arac blogu uretir -> policy -> DockerExecutor lab aginda gercek nmap -> hedef.
    Hedefi IP'ye cevir (DNS-siz): model acik IP'yi onurlandirir, container adini kacirir."""
    from agent.backends.docker_executor import DockerExecutor
    ip = _docker_container_ip(target)
    real_target = ip or target                      # IP alinamazsa isme dus (DNS calisiyorsa olur)
    scope = scope or [real_target, "172.16.0.0/12", "172.30.0.0/24"]
    return run_gguf_demo(scope, model, executor=DockerExecutor(network=network),
                         target=real_target, label="gguf+docker uctan uca bitti",
                         tokenizer_dir=tokenizer_dir)


def run_assistant_demo(workspace: str | None = None, real_cmd: bool = False) -> str:
    """Asistan araclari demo: gercek jailed dosya islemi + policy reddi (mock guvenlik elleri).
    real_cmd=True -> run_cmd izole docker sandbox'ta GERCEKTEN calisir (alpine, --network none)."""
    import tempfile
    from agent.audit import AuditLog
    from agent.backends.assistant_executor import AssistantExecutor
    from agent.composite_executor import CompositeExecutor
    from agent.executor import MockExecutor
    from agent.policy import LabPolicy
    from agent.toolcall import ToolCall
    ws = workspace or tempfile.mkdtemp(prefix="octopus-ws-")
    sandbox = None
    if real_cmd:
        from agent.backends.sandbox_executor import DockerSandboxExecutor
        sandbox = DockerSandboxExecutor()  # None kalirsa AssistantExecutor MockExecutor'a duser
    # ws AYNI olmali: policy guard (workspace_root) ile executor (root) ayni hapishaneyi gormeli.
    execu = CompositeExecutor(security=MockExecutor(),
                              assistant=AssistantExecutor(ws, sandbox=sandbox))
    # real_cmd = kullanicinin acik opt-in'i -> run_cmd (risk=high) sandbox'a ulassin diye
    # allow_high. Uretimde run_cmd onayi per-cagri insan kapisidir; bu DEMO opt-in'i gosterir.
    registry = ToolRegistry(LabPolicy(scope=[], allow_high=real_cmd, workspace_root=ws),
                            execu, AuditLog.default())
    steps = [
        ToolCall(name="write_file", params={"yol": "not.txt", "icerik": "Octópus lab notu"}),
        ToolCall(name="read_file", params={"yol": "not.txt"}),
        ToolCall(name="write_file", params={"yol": "../kacis.txt", "icerik": "x"}),  # reddedilmeli
    ]
    if real_cmd:  # gercek sandbox turunda run_cmd'yi de goster (docker yoksa zarifce HATA doner)
        steps.append(ToolCall(name="run_cmd", params={"komut": "echo octopus-sandbox && id && pwd"}))
    lines = [f"[workspace] {ws}"]
    succeeded = True
    for c in steps:
        result = registry.invoke_result(c)
        if result.status == "awaiting_approval" and result.proposal_id:
            # CLI opt-in authorizes this fixed demo fixture, never arbitrary model calls.
            result = registry.invoke_result(c, approval_token=registry.approve(result.proposal_id))
        lines.append(f"[{c.name}] -> {result.output}")
        expected_status = "denied" if c.params.get("yol") == "../kacis.txt" else "executed"
        succeeded = succeeded and result.status == expected_status
    lines.append("(asistan demo bitti)" if succeeded else "(gorev tamamlanmadi)")
    return DemoTranscript("\n".join(lines), "completed" if succeeded else "failed")


def main() -> int:
    # Windows konsolu (cp1254) ó/Ç basamaz -> UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Octópus agent harness demo")
    ap.add_argument("--scope", nargs="*", default=None, help="izinli IP/CIDR (mod varsayilanini ezer)")
    ap.add_argument("--real", action="store_true", help="RealExecutor: Kali WSL'de gercek nmap")
    ap.add_argument("--docker", action="store_true", help="DockerExecutor: lab docker aginda gercek nmap")
    ap.add_argument("--gguf", action="store_true", help=f"GgufModel: Ollama'da {DEFAULT_MODEL}")
    ap.add_argument("--assistant", action="store_true", help="Asistan araclari demo (jailed file + policy)")
    ap.add_argument("--real-cmd", action="store_true", help="run_cmd'yi GERCEK docker sandbox'ta "
                    "calistir (alpine, --network none, --rm; Docker gerekir). --assistant ile.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model adi (varsayilan {DEFAULT_MODEL})")
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR,
                    help=f"egitim chat-template dizini (varsayilan {DEFAULT_TOKENIZER_DIR})")
    ap.add_argument("--rol", default=None, help="rol-scoped subagent (containment): sadece o "
                    "domain'in araclari calisir (or. recon-scan/web/blue-server). Bos=tam katalog")
    ap.add_argument("--target", default=None, help="hedef (mod varsayilanini ezer)")
    ap.add_argument("--port", type=int, default=None, help="taranacak port (mod varsayilani)")
    args = ap.parse_args()
    if args.assistant:
        output = run_assistant_demo(real_cmd=args.real_cmd)
    elif args.gguf and args.docker:
        output = run_gguf_docker_demo(args.scope, args.model,
                                   args.target or "octopus-target", tokenizer_dir=args.tokenizer_dir)
    elif args.gguf:
        output = run_gguf_demo(args.scope, args.model, role=args.rol,
                            target=args.target or "10.10.10.5", tokenizer_dir=args.tokenizer_dir)
    elif args.docker:
        output = run_docker_demo(args.target or "octopus-target", args.scope, port=args.port or 80)
    elif args.real:
        output = run_real_demo(args.target or "127.0.0.1", args.scope, port=args.port or 8000)
    else:
        output = run_demo(args.scope or ["10.10.10.0/24"])
    print(output)
    return 0 if getattr(output, "status", "completed") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
