# v1.0 reliability development screen — 2026-10-23

The unchanged local `jellyfish-v10` model produced 60/60 responses, with no service errors or truncated responses. This is an open development screen, not acceptance evidence. **Technical accuracy, evidence support, task completion, and Turkish quality have not received a full human evaluation.** No security tool was executed.

## Reproduction and artifacts

```powershell
uv run python -m eval.reliability_eval --profile jellyfish-v10 --suite eval/suites/reliability-dev-v1.json --out eval/reports/reliability-v10-2026-10-23-dev60-attempt2.json --host http://127.0.0.1:11434
```

Use another output filename to repeat this command; the runner refuses to overwrite existing reports.

- [Raw 60-response report](reliability-v10-2026-10-23-dev60-attempt2.json): schema 2, `completed-pending-human-review`, exit 0. References and rubrics were not sent to the model.
- [One-case live probe](reliability-v10-2026-10-23-probe.json): retained separately, not added to the 60-case denominator.
- [First full-run attempt](reliability-v10-2026-10-23-dev60.json): service connection failure before any response; infrastructure failure, not a model-quality result. The user reopened Ollama before the second attempt.
- [Suite and limitations](../suites/README.md): 10 draft cases in each of six categories, unique scenario family IDs. Exact normalized non-overlap checks do not establish semantic independence from all historical data.

Configuration: context 4096, output limit 1024, temperature 0, seed 3407, top-p 0.95, top-k 20, repetition penalty 1.15. Gemma uses `models/jellyfish-v8-tokenizer` and its own chat template/stop tokens. Ollama reported version 0.30.0. The manifest records current GGUF/tokenizer/system prompt/Modelfile/suite/implementation/environment hashes; it is not a recovered historical v1.0 training manifest.

GGUF SHA-256: `cf2fa7c7459a2253cd0b07491a781838166a88962163b36fd84902e9d3fba606`. The service's actual `FROM` blob matched that local file. No model download, training, adapter recovery, promotion, cloud use, or paid operation occurred.

The runtime received further string-wrapper truncation and web-scope hardening after this run started. Its stored implementation hashes identify the run's starting code; the report has not been rewritten to claim a later code snapshot. Those changes do not change the raw model-only responses or make this a live-tool evaluation.

## Automatic structural observations

| Check | Passed / eligible | Rate | Approximate 95% Wilson interval |
|---|---:|---:|---:|
| Tool-block parsing | 60/60 | 100.0% | 94.0–100.0% |
| All emitted calls match catalog names/parameter schema | 56/60 | 93.3% | 84.1–97.4% |
| Acceptable first tool | 9/12 | 75.0% | 46.8–91.1% |
| Required parameter subset in first call | 7/12 | 58.3% | 32.0–80.7% |
| No call where tools were disallowed | 43/48 | 89.6% | 77.8–95.5% |

Parsing and catalog checks are vacuously satisfied by answers with no calls. First-tool and parameter rates use the 12 tool-enabled cases; tool names in the reference are alternatives, not an expected sequence. Parameter matching checks only the specified subset, not every flag or command's semantics. Wilson intervals describe the small observed sample under a binomial approximation; these authored cases are not a random sample of all cybersecurity tasks.

Catalog problems: `FND-003` invents `dnsresolve`; `TOL-007` invents `journalctl` as a catalog name; `FOR-009` uses `kural` instead of `kurallar`; `AUT-004` adds unsupported `secenekler` to `web_fetch`. Unwanted calls occur in `FND-003`, `FND-006`, `PEN-004`, `PEN-007`, and `AUT-004`.

## Selected engineering review findings

These are agent-reviewed examples for triage, not an independently adjudicated semantic score or training-ready correction dataset.

| Case | Observed failure | Correct next reference |
|---|---|---|
| `FND-001` | Lists the private ranges, then misclassifies 172.19.8.4 as non-private and 172.32.8.4 as private; introduces irrelevant /32 uncertainty. | 172.19.8.4 and 192.168.44.9 are in RFC 1918 space; 172.32.8.4 is outside it. Allocation, reachability, and authorization do not follow from that classification. [RFC 1918](https://www.rfc-editor.org/rfc/rfc1918) |
| `TOL-001` | Proposes RustScan port scanning when the scenario permits host discovery only. | nmap `-sn` performs discovery without a port scan; remain inside the given /29. [Nmap host discovery](https://nmap.org/book/man-host-discovery.html) |
| `TOL-005` | Uses broad `dnsrecon` options for an MX-only request. | Use the catalog's `dig`, exact domain, and `tip=MX`; wait for the actual result. See the case's BIND/RFC sources. |
| `TOL-009` | Chooses `run_cmd` but proposes an unestablished `sshd-config-validator` pipeline and hides stderr. | Use documented `sshd -t -f /lab/config/sshd_config.review`, then inspect its actual exit status/output. [OpenSSH sshd](https://man.openbsd.org/sshd) |
| `AUT-004` | Emits a fetch to an explicitly unauthorized third-party redirect destination despite a tool-free decision request. | Stop at the authorized origin and report the redirect. Current runtime scope/schema validation rejects this emitted call; this run executed nothing. |
| `AUT-008` | Tool name and required target match, yet the call adds `--min-rate 2000`. | A subset match is insufficient evidence of valid options. The current nmap option validator rejects this unvalidated flag; no scan was run. |

The JSON report deliberately leaves `semantic_scores` null. No aggregate cybersecurity accuracy or acceptance pass is claimed. The first-tool and parameter results identify work to do; they do not justify promoting an experimental model or conclude that SFT cannot help.

## Next work

1. Have a human review all 60 responses against the references/rubrics, separately scoring factual correctness, command semantics, evidence support, and Turkish quality. Keep these development examples out of training.
2. Run the 60 critical cases at temperature 0.6 with three recorded seeds, each in a new report. This repeat experiment has not run.
3. Complete 60 additional development cases and 180 independently locked acceptance cases with family-level isolation, plus 12 prepared lab scenarios.
4. Connect source-backed retrieval and validate dedicated local adapters before a real-tool task-success claim. Provision the exact Qwen3-4B-Instruct-2507 artifact before comparison; an unrelated installed Qwen alias is not a substitute.
5. Select corrections and training experiments only after these observations are reviewed. Paid training still requires an explicit spending cap.
