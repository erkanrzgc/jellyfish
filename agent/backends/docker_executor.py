"""DockerExecutor: araci lab docker aginda CONTAINER olarak calistirir (Model B).

NEDEN: araci hedefle AYNI docker aginda container olarak kostur -> DNS adiyla erisir.
Varsayilan `docker` CLI'i dogrudan cagrilir (Kali WSL gerekmez); `docker_prefix` ile istenirse
baska bir WSL distro'su uzerinden yonlendirilebilir. (RealExecutor Kali-WSL yolu ayridir.)

GUVENLIK: Policy kapisi registry.invoke'da ONCE gecer. Komut argv (shell=False); 'secenekler'
shlex ile token -> shell metakarakteri literal arg. Imaj entrypoint'i = arac (instrumentisto/nmap
gibi) -> arg olarak yalniz bayrak+hedef verilir."""
from __future__ import annotations
import shlex
import subprocess
from dataclasses import dataclass, field
from agent.catalog import get_spec, positional_args, target_value

_DEFAULT_TIMEOUT = 180
_DEFAULT_NETWORK = "jellyfish-lab"
_DEFAULT_DOCKER_PREFIX: tuple[str, ...] = ("docker",)

# Arac -> docker imaji (entrypoint = arac). Genisletilir; haritada olmayan icin
# ileride tek kali-tools imaji (kali-linux-default) fallback'i eklenecek.
TOOL_IMAGES: dict[str, str] = {
    "nmap": "instrumentisto/nmap",
}


def build_argv(tool: str, params: dict, network: str = _DEFAULT_NETWORK,
               docker_prefix: list[str] | None = None,
               images: dict[str, str] | None = None) -> list[str] | None:
    """docker-lab argv: <docker_prefix> run --rm --network <net> <image> <flags...> [target].
    docker_prefix varsayilan ['docker']. Imaj tanimli degilse None (run() 'imaj yok' doner)."""
    prefix = list(docker_prefix) if docker_prefix is not None else list(_DEFAULT_DOCKER_PREFIX)
    images = images if images is not None else TOOL_IMAGES
    image = images.get(tool)
    if image is None:
        return None
    lead = positional_args(get_spec(tool), params)
    flag_args = shlex.split(str(params.get("secenekler", "") or ""))
    argv = [*prefix, "run", "--rm", "--network", network, image, *lead, *flag_args]
    target = target_value(params)
    if target is not None:
        argv.append(target)
    return argv


@dataclass
class DockerExecutor:
    network: str = _DEFAULT_NETWORK
    docker_prefix: list[str] = field(default_factory=lambda: list(_DEFAULT_DOCKER_PREFIX))
    timeout: int = _DEFAULT_TIMEOUT
    images: dict[str, str] = field(default_factory=lambda: dict(TOOL_IMAGES))

    def preview(self, tool: str, params: dict) -> list[str]:
        argv = build_argv(tool, params, self.network, self.docker_prefix, self.images)
        if argv is None:
            raise ValueError(f"'{tool}' icin docker imaji yok")
        return argv

    def run(self, tool: str, params: dict) -> str:
        if get_spec(tool) is None:
            return f"HATA: bilinmeyen arac '{tool}'"
        argv = build_argv(tool, params, self.network, self.docker_prefix, self.images)
        if argv is None:
            return f"HATA: '{tool}' icin docker imaji tanimli degil (kali-tools imaji ileride)"
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return f"HATA: '{tool}' zaman asimi ({self.timeout}s)"
        except FileNotFoundError:
            return f"HATA: '{self.docker_prefix[0]}' bulunamadi (Docker kurulu ve PATH'te mi?)"
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode:
            return f"HATA: '{tool}' exit={proc.returncode}: {out.strip()}"
        return out.strip() or f"({tool} cikti uretmedi, exit={proc.returncode})"
