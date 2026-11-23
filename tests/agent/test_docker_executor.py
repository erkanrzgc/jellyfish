from agent.backends.docker_executor import build_argv, DockerExecutor


def test_build_argv_nmap_docker_run():
    argv = build_argv("nmap", {"hedef": "jellyfish-target", "secenekler": "-Pn -sV -p80"})
    assert argv == [
        "docker", "run", "--rm", "--network", "jellyfish-lab",
        "instrumentisto/nmap", "-Pn", "-sV", "-p80", "jellyfish-target",
    ]


def test_build_argv_unmapped_tool_none():
    # Haritada imaji olmayan arac -> None (run() 'imaj yok' doner).
    assert build_argv("sqlmap", {"url": "http://x"}) is None


def test_build_argv_custom_network():
    argv = build_argv("nmap", {"hedef": "1.2.3.4"}, network="test-net")
    assert "--network" in argv and "test-net" in argv


def test_build_argv_no_shell_injection():
    argv = build_argv("nmap", {"hedef": "t", "secenekler": "; rm -rf /"})
    assert argv[0] == "docker"  # shell yok, dogrudan docker CLI
    assert ";" in argv and "rm" in argv  # literal token'lar


def test_run_unknown_tool_rejected():
    assert "bilinmeyen" in DockerExecutor().run("yok_boyle", {}).lower()


def test_run_unmapped_tool_message():
    out = DockerExecutor().run("sqlmap", {"url": "http://x"})
    assert "imaj" in out.lower()


def test_build_argv_renders_subcommand_param_not_dropped():
    # BUG FIX: secenekler+hedef disindaki parametreler (or. ghunt 'modul') dusuruluyordu.
    # Imaj-haritasi enjekte ederek positional render'in docker yolundan da gectigini dogrula.
    argv = build_argv("ghunt", {"modul": "email", "hedef": "a@b.com"},
                      images={"ghunt": "jellyfish/ghunt"})
    assert argv is not None
    assert "email" in argv                     # modul artik dusmuyor
    assert argv.index("email") < argv.index("a@b.com")  # alt-komut hedeften once
    assert argv[-1] == "a@b.com"               # ag-hedefi en sona


def test_build_argv_docker_prefix_routes_through_wsl():
    # Opsiyonel yonlendirme: docker baska bir WSL distro'sundan cagrilabilir.
    prefix = ["wsl.exe", "-d", "kali-linux", "--", "docker"]
    argv = build_argv("nmap", {"hedef": "t"}, docker_prefix=prefix)
    assert argv[:6] == [*prefix, "run"]
    assert argv[-1] == "t"


def test_executor_default_calls_docker_directly(monkeypatch):
    import subprocess
    seen: dict = {}

    class _Proc:
        returncode, stdout, stderr = 0, "3000/tcp open", ""

    def _fake_run(argv, **kwargs):
        seen["argv"] = argv
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    out = DockerExecutor().run("nmap", {"hedef": "jellyfish-target", "secenekler": "-Pn -p3000"})
    assert out == "3000/tcp open"
    assert seen["argv"][0] == "docker"
    assert "wsl.exe" not in seen["argv"]


def test_executor_missing_docker_message(monkeypatch):
    import subprocess

    def _missing(*a, **k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", _missing)
    out = DockerExecutor().run("nmap", {"hedef": "t"})
    assert "docker" in out.lower() and "wsl" not in out.lower()
