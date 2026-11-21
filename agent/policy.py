"""Lab-only yetki + risk kapisi. Varsayilan: dis hedef yok, high-risk onay ister.
Hedef IP/CIDR kapsam allow-list'iyle karsilastirilir; kapsam disi -> reddet (modelin
kendi reddiyle cift kilit)."""
from __future__ import annotations
from dataclasses import dataclass, field
from agent.catalog import ToolSpec, TARGET_KEYS, target_value, parameter_error
from agent.targets import target_in_scope


@dataclass(frozen=True)
class Decision:
    allowed: bool
    requires_approval: bool
    reason: str


@dataclass
class LabPolicy:
    scope: list[str] = field(default_factory=list)  # izinli IP/CIDR (bos = dis hedef yok)
    allow_high: bool = False
    workspace_root: str | None = None  # asistan dosya araclari icin hapishane koku (None = fs reddi)

    @classmethod
    def default(cls) -> "LabPolicy":
        return cls(scope=[], allow_high=False)

    def _in_scope(self, target: str) -> bool:
        return target_in_scope(target, self.scope)

    def decide(self, spec: ToolSpec, params: dict) -> Decision:
        issue = parameter_error(spec, params)
        if issue:
            return Decision(False, False, issue)
        if spec.domain == "asistan":
            return self._decide_assistant(spec, params)
        target = target_value(params)
        # FAIL-CLOSED: arac hedef-alan bir param bildiriyorsa ama cagri degerlendirilebilir
        # hedef vermediyse REDDET — yoksa hedef gizlenerek scope kilidi atlanabilir.
        spec_has_target = any(k in spec.params for k in TARGET_KEYS)
        if spec_has_target and target is None:
            return Decision(False, False, f"'{spec.name}' icin hedef parametresi eksik (fail-closed)")
        for key in TARGET_KEYS:
            if key in params and not self._in_scope(params[key]):
                return Decision(False, False, f"hedef '{params[key]}' izinli kapsam disinda (lab-only)")
        if spec.risk == "high" and not self.allow_high:
            return Decision(False, True, f"'{spec.name}' yuksek riskli, acik onay gerekir")
        if "secenekler" in params and params["secenekler"]:
            if spec.name == "nmap":
                from agent.arguments import nmap_flags
                try:
                    nmap_flags(params["secenekler"])
                except ValueError as error:
                    return Decision(False, False, str(error))
            else:
                return Decision(False, False, f"'{spec.name}' icin serbest secenekler dogrulanmadi")
        for key, value in params.items():
            if key in TARGET_KEYS or key == "secenekler":
                continue
            if spec.name == "dig" and key == "tip" and value.upper() in {
                "A", "AAAA", "MX", "TXT", "NS", "SOA", "CNAME", "PTR", "SRV", "CAA", "ANY",
            }:
                continue
            return Decision(False, False, f"'{spec.name}.{key}' icin dogrulanmis adapter yok")
        return Decision(True, False, "izinli")

    def _decide_assistant(self, spec: ToolSpec, params: dict) -> Decision:
        """Asistan-domain araclarini domaine gore fs/cmd/web guard'ina yonlendir (fail-closed)."""
        from agent.guards import cmd, fs, web
        name = spec.name
        if name in ("read_file", "write_file", "edit_file", "list_dir", "grep"):
            return fs.guard(params, self.workspace_root)
        if name == "web_fetch" and not self._in_scope(params["url"]):
            return Decision(False, False, f"hedef '{params['url']}' izinli kapsam disinda (lab-only)")
        if name in ("web_fetch", "web_search"):
            return web.guard(params)
        if name == "run_cmd":
            d = cmd.guard(params)
            if not d.allowed:
                return d
            if spec.risk == "high" and not self.allow_high:
                return Decision(False, True, f"'{spec.name}' yuksek riskli, acik onay gerekir")
            return d
        return Decision(False, False, f"bilinmeyen asistan araci '{name}' (fail-closed)")
