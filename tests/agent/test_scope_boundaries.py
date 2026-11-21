import pytest

from agent.catalog import get_spec
from agent.policy import Decision, LabPolicy


@pytest.mark.parametrize("options", [
    "-sV 192.0.2.1", "-iL targets.txt", "--exclude 10.10.10.5 192.0.2.1",
    "--script http-fetch", "-sV; whoami", "-p80 -- 192.0.2.1", "--resume scan.log",
])
def test_nmap_options_cannot_hide_destinations_or_unvalidated_actions(options):
    decision = LabPolicy(scope=["10.10.10.0/24"]).decide(
        get_spec("nmap"), {"hedef": "10.10.10.5", "secenekler": options},
    )
    assert not decision.allowed


def test_all_target_fields_are_checked():
    decision = LabPolicy(scope=["10.10.10.0/24"]).decide(
        get_spec("nmap"), {"hedef": "10.10.10.5", "url": "https://192.0.2.1"},
    )
    assert not decision.allowed


@pytest.mark.parametrize("target", ["10.10.10.0/16", "10.10.10.5 192.0.2.1", "10.10.10.5\n192.0.2.1", "-iL"])
def test_target_must_be_single_and_entirely_in_scope(target):
    assert not LabPolicy(scope=["10.10.10.0/24"]).decide(get_spec("nmap"), {"hedef": target}).allowed


@pytest.mark.parametrize("scope,target,tool", [
    (["10.10.10.0/24"], "https://10.10.10.5:8443/path", "sqlmap"),
    (["2001:db8:1::/48"], "2001:db8:1::5", "nmap"),
    (["2001:db8:1::/48"], "2001:db8:1::/64", "nmap"),
    (["lab.example"], "https://lab.example:8443/path", "sqlmap"),
    (["lab.example"], "LAB.EXAMPLE.", "nmap"),
])
def test_scope_understands_urls_ipv6_domains_and_contained_networks(scope, target, tool):
    assert LabPolicy(scope=scope, allow_high=True).decide(
        get_spec(tool), {"url" if tool == "sqlmap" else "hedef": target},
    ).allowed


def test_domain_suffix_is_not_an_authorization_match():
    assert not LabPolicy(scope=["lab.example"]).decide(
        get_spec("nmap"), {"hedef": "lab.example.evil.test"},
    ).allowed


@pytest.mark.parametrize("url", [
    "https://outside.example/page", "https://docs.example.evil.test/",
])
def test_assistant_web_fetch_cannot_bypass_target_scope(monkeypatch, tmp_path, url):
    from agent.audit import AuditLog
    from agent.backends.assistant_executor import AssistantExecutor
    from agent.guards import web
    from agent.registry import ToolRegistry
    from agent.toolcall import ToolCall

    # Isolate authorization from DNS and SSRF checks; no network request occurs.
    monkeypatch.setattr(web, "guard", lambda params: Decision(True, False, "public"))
    fetched = []
    executor = AssistantExecutor(str(tmp_path), http_get=lambda value: fetched.append(value) or "body")
    registry = ToolRegistry(LabPolicy(scope=["docs.example"]), executor,
                            AuditLog(tmp_path / "audit.jsonl"))
    result = registry.invoke_result(ToolCall("web_fetch", {"url": url}))
    assert result.status == "denied"
    assert result.proposal_id is None and fetched == []


def test_web_fetch_accepts_encoded_path_but_not_encoded_host(monkeypatch):
    from agent.guards import web
    monkeypatch.setattr(web, "guard", lambda params: Decision(True, False, "public"))
    policy = LabPolicy(scope=["docs.example"])
    spec = get_spec("web_fetch")
    assert policy.decide(spec, {"url": "https://docs.example/a%2Fb?q=c%20d"}).allowed
    assert not policy.decide(spec, {"url": "https://docs%2eexample/a"}).allowed
