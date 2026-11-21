# Reliability development cases

`reliability-dev-v1.json` contains the first **60 newly authored Turkish development cases**: 10 each for fundamentals, tooling, pentest, forensics, incident response, and authorization. The planned 300-case evaluation corpus still needs **60 more dev cases and 180 independently locked acceptance cases** (per category: 20 dev and 30 acceptance in total).

This is an open **dev-only**, **source-backed-draft** corpus. Prompts, references, and rubrics are visible for development. It is not a locked acceptance set and must not be reported as independent acceptance evidence or placed in SFT/training data. No acceptance cases have been authored or opened as part of this increment.

The supplied logs, packet summaries, paths, CTF observations, and incident facts are independently authored synthetic scenarios. Their presence does not mean those files exist or that a scan, exploit, model inference, or laboratory experiment was performed. Official primary documentation was consulted for the technical interpretations and each case links its relevant sources. **Human subject-matter review and laboratory verification remain pending.** The source-backed status must not be promoted to either of those claims.

Each case has a unique scenario `family_id`. All 60 current cases have `critical: true`, yielding a balanced initial repeat subset of 10 per category; this is a development designation, not an externally validated severity label. Preserve family separation when designing future splits or independently locked acceptance cases.

The tool fields describe model-output expectations, not authorization to execute tools during evaluation:

- `allow_tools: false` requires an answer without tool calls. This includes both informational cases and boundary/refusal cases; the rubric distinguishes them.
- `expected_tools` lists acceptable first catalog tool names. An empty list with `allow_tools: true` would impose no exact first-tool requirement.
- `expected_params` is a subset of parameters required in that first call. Other constraints, including command options and scope limits, appear in the rubric. A correct tool name alone does not establish task success or argument correctness.
- `reference_answer` is a concise expected interpretation or next action. `rubric` supplies concrete review criteria; it is not an automatic semantic correctness oracle.
- `sources` contains primary HTTPS documentation. A source supports the technical rule; the prompt supplies the synthetic incident facts. Source links are not live tool outputs.

Run the local integrity checks with:

```powershell
uv run pytest tests/eval/test_reliability_cases.py -q
```

The checks enforce size/balance, unique case IDs/families/normalized prompts, complete fields, dev-only split labels, catalog-valid first tools/parameter keys, and exact normalized prompt non-overlap with existing `eval/data` and `data/sft` JSON/JSONL/text prompts. They do not prove semantic novelty, factual correctness, family separation from every legacy training example, successful tool execution, or model reliability. Do not turn these structural checks into a quality score.
