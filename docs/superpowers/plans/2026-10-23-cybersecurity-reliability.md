# Cybersecurity reliability implementation

Approved by the user on 2026-10-23. Product: Turkish-first cybersecurity knowledge, authorized pentest/CTF assistance, defensive analysis, and evidence-based incident work. Real operations require approval. Local work comes first; paid training requires a separately agreed spending cap.

## Delivery sequence and acceptance

1. Establish an independently authored evaluation baseline; fix reproducible runtime failures before live integration.
2. Provide an advisory/approval-driven conversation interface and source-backed local knowledge.
3. Compare unchanged v1.0 with one small alternative before selecting any training candidate.
4. Prepare auditable training data and resumable, validation-aware training. Do not train from GGUF or pretend the missing v1.0 adapter was recovered.

Targets on the locked acceptance set: technical accuracy >=90%, tool selection >=95%, parameters >=95%, lab completion >=85%, evidence-supported conclusions >=95%, no unnecessary tools >=98%, Turkish quality >=4/5; zero observed unapproved/out-of-scope execution or fabricated CVE/source. Candidate must improve the total by >=5 percentage points without losing more than one case in any category. Report denominators and uncertainty; a passing sample is not a universal guarantee.

## Work packages

- [x] **Runtime failures:** remove execution fallback after a dropped/refused skill-corrected call; robust parsing and at most one repair for invalid/unknown calls; normalized IP/CIDR/URL/domain scope checks; reject hidden additional destinations and unvalidated flags. Test before implementation.
- [x] **Execution accounting and approval:** distinguish proposed, awaiting approval, denied, executed, failed; count only executed calls. Bind one-use host approval to the exact call, policy, workspace, and command. Stop on denial/error; never manufacture a completed result. Preserve compatible string wrappers for older callers.
- [x] **Evaluation first 60:** ten new dev cases in each of fundamentals, tooling, pentest, forensics, incident, authorization. Include reference answers, explicit rubrics, relevant primary sources, family IDs, and review status. Keep draft/reference-backed cases distinct from human/lab verification.
- [x] **Baseline runner:** profile-aware model-only generation, all calls, raw responses, errors/truncation, timing/settings, model/tokenizer/prompt/Modelfile hashes. Defaults: context 4096, output 1024, temperature 0; critical 60 repeated at 0.6 with three recorded seeds. No live tool execution by an evaluation runner.
- [ ] **Full evaluation:** 300 independently authored cases, 50/category (20 dev, 30 locked acceptance). Split scenario families before variants; exclude train/dev/acceptance contamination. Existing tests remain regression evidence. Add 12 prepared end-to-end labs, six red and six blue. Retire an acceptance set if its errors inform tuning.
- [x] **Conversation interface:** separate advisory and approved-tool modes with history and truthful task status. Preserve current demo commands. Reports distinguish observations, hypotheses, and proposed checks.
- [ ] **Validated adapters:** nmap port/service checks; httpx status/headers; ffuf with an approved local wordlist; sqlmap detection-only on a prepared lab; tshark local PCAP; yara local files with pinned rules; read-only Linux diagnostics. Reject unsupported parameters/options. Do not advertise lab validation until it actually ran.
- [ ] **Knowledge:** connect retrieval to the active GGUF path; official vendor/tool docs, NVD/CISA, verified local notes. Store source URL/version/date/hash. CPU multilingual-e5-small plus lexical retrieval, heading-aware 384-token chunks/64 overlap/top four passages. Untrusted document/log text never supplies approval. Local files/logs never leave the machine.
- [ ] **Profiles/comparison:** retain octopus-v10 default; compare Qwen/Qwen3-4B-Instruct-2507 with its own template/EOS. Pick candidates on dev data before locked acceptance. Profile changes must not silently promote models.
- [ ] **Training data:** 600 source/lab-verified corrections (300 knowledge/commands, 300 behavior) and 1200 clean replay examples; per-source/domain/tool distribution and immutable provenance. Group split before augmentation/upsampling. Draft generated examples are not automatically training-ready.
- [ ] **Training process:** assistant-response masking verified at token level, context 2048, validation/checkpoints/resume/truncation reporting, pinned environment and archived adapter/tokenizer/manifest/results. Preflight GPU/VRAM/precision and a 30-step smoke before a full run. No 9B bf16 training on local 8GB or 16GB GPU.
- [ ] **Controlled experiments:** at most three SFT comparisons: replay only; replay+knowledge; replay+knowledge+behavior. Keep the recipe otherwise equal. DPO is a subsequent separate preference-data experiment, not the initial remedy. Colab only trains/evaluates; security tools stay in the local lab. Persist checkpoints outside ephemeral sessions.

## First implementation package

Close the reproduced runtime holes, prepare the 60-case development screen, and attempt a current v1.0 model-only baseline. Missing local services may be started without cloud spending; report infrastructure blockers explicitly. Never replace live output with mock output to claim a baseline.

Keep current work in this checkout on `feat/cybersecurity-reliability`, preserving prior uncommitted edits and ignored artifacts. Update `PROJECT_HANDOFF.md` with exact verification, incomplete work, and the next step at every meaningful stopping point. Do not commit or publish without a user request.

## First-package result (2026-10-23)

Implemented and regression-tested the runtime fixes, exact-call host approval, conversation entry point, artifact-aware baseline runner, and 60 development drafts. A live v1.0 run recorded 60/60 responses without service errors or truncation; [the report](../../../eval/reports/reliability-v10-2026-10-23-dev60.md) distinguishes structural scores from pending semantic review. The final runtime suite passed 435 tests with one platform skip; offline smoke passed 3/3.

The baseline-runner checkbox means the runner is implemented. The temperature-0.6 three-seed repeat experiment has not run. Conversation acceptance is limited to tested runtime behavior, not completed real-tool tasks. Model comparison is configured only; retrieval, dedicated adapters/lab verification, the remaining 240 cases, verified correction/replay data, and training remain open. No weights were changed or promoted.
