# Competitor forensics: 20 public Lexsi 2026 repositories

Audited 2026-11-27, 05:10–06:30 IST, by the ActionProof sprint (branch `stage2-superiority-sprint`, base `7ae863f`).

## Scope and method

**The set.** These 20 repositories come from an earlier public search. They are **not** a verified list of the
Stage 2 finalists. The official page says Stage 1 shortlists "up to 20 teams" and does not publish their names.

**What was inspected, for every repository:**
- A clone of the default branch at the stated HEAD. Blobs above 2 MB were skipped. Repo 01 is a shallow clone of
  60 commits, which holds its whole history of 11 commits.
- The last 30 commit messages, the first commit, and the remote branches.
- The file tree, with counts by file type.
- The README in full, and the report or paper's limitations and negative-results sections where they exist.
- The lists of committed result files.
- Mentions of Cohere models and Lexsi toolkits (Python imports, not just README text).
- Test counts, CI and dependency pinning.
- Hub metadata of the headline model, the main dataset and every Space, read from the public API without
  credentials: SDK, runtime stage, hardware, last modification and visibility.

**What was not done.** No competitor code was run. Their result files were not recomputed. Spaces were not driven
in a browser: their type (static replay or live inference) comes from the Hub's `sdk` and `hardware` fields and
the repository's own Space source. Videos were not watched.

**Inspection depth.** Each profile ends with a confidence line. "High" means the README and the result tables or
limitations were read. "Medium" means the README only.

**Raw notes.** `research_audit/competitors/notes.md` and `profiles/*.txt` (the profiling script is
`research_audit/competitors/profile.sh`).

**Scores.** No score from the earlier snapshot is reused. The rankings are in `COMPETITIVE_THREAT_RANKING.md`.

**Hub status (API, 2026-11-27 ~06:20 IST).**

| Space | SDK | Runtime | Hardware | What it is |
|---|---|---|---|---|
| ashish-soni08/escalation-safe-agent-dashboard | static | RUNNING | none | replay dashboard |
| ToolW/desi-agent-lab | static | RUNNING | none | replay of scored traces |
| dev-willbird1936/drift-monitor | static | RUNNING | none | replay page, plus a live tab that calls the scorer |
| dev-willbird1936/drift-monitor-scorer | gradio | RUNNING | zero-a10g | **live inference** (the monitor) |
| itssaharsh/relapse | static | RUNNING | none | results explorer |
| primce/hinglish-expense-agent | gradio | SLEEPING | none | **live inference** (ZeroGPU; asleep at audit time) |
| primce/hinglish-episodes | static | RUNNING | none | replay gallery |
| lost-in-refusal/lost-in-refusal-demo | static | RUNNING | none | static (last modified Nov 6) |
| Earther/evidencelock-demo | static | RUNNING | none | replay |
| BMEscanor/against-the-safety-trace-viewer | static | RUNNING | none | replay ("the model does not run") |
| Finsafe-team/finsafe | static | RUNNING | none | static |
| qasimjethwa/recovery-agent-taught-by-rules | static | RUNNING | none | replay of predictions |
| *atharvgaur18/actionproof-live (ours)* | gradio | RUNNING | zero-a10g | **live inference** |

Every headline model and dataset checked was public and ungated:
- `ashish-soni08/tiny-aya-global-esa-grpo-safe-v4-1`
- `ToolW/desi-agent-tinyaya-grpo-mt-lora`
- `dev-willbird1936/drift-monitor-tiny-aya`
- `itssaharsh/relapse-tiny-aya-global-unlearn-bio`
- `primce/tiny-aya-hinglish-lora`
- `lost-in-refusal/tiny-aya-global-bactrian-lora-safegrad`
- `Earther/evidencelock-track3-c0`
- `qasimjethwa/aya-expanse-8b-recovery-lora`
- `incharajayaram2020/icsa-grounded-assistant-adapter`
- three datasets

---

## 01 Lost in Injection

- **Project / repository / HEAD / latest update.** `beingamanforever/Lost-in-injection`, `d06a5a7`, 2026-11-24
  (11 commits, first Nov 14). The latest commit fixes a portable env script. The last substantive work was on
  Nov 23: a Manim video restyle, a slides PDF, and AgentDojo in five languages.
- **Track and problem.** Track 2a, problem statements 3 and 4. The problem is indirect prompt injection across
  11 languages, scripts and code-mixes.
- **One-sentence thesis.** Prompt-injection defenses trained with Meta-SecAlign's recipe transfer across languages
  one language at a time. They leak through an input filter that cannot read Swahili, through attackers who keep
  rewording, and through a scorer that reads only the first word of a reply.
- **Pipeline.**
  1. CuratorKIT curates AlpacaFarm tasks and injections.
  2. Machine translation with quality gates.
  3. DPO preference pairs, trained with LoRA through AlignTune.
  4. Attacks per language: language search, equal-budget paraphrase and noise, and a PAIR-style attacker
     (Aya Expanse 8B).
  5. Three scorers: the starts-with rule, "Hacked" anywhere, and an LLM judge (qwen3.7-flash).
  6. AuditKIT for utility.
  7. Figures and a report.
- **Models and toolkits.**
  - Models: Tiny Aya Global, Fire, Earth and Water; Qwen3-4B; Llama 3.1 8B; Llama 3.3 70B; Meta-SecAlign 8B
    and 70B.
  - Toolkit imports: `from curatorkit import Curator`, `from aligntune import create_rl_trainer`,
    `from auditkit.lmeval_engine import run_benchmark`.
- **Experiments.**
  - 208 AlpacaFarm-Hacked cases × 11 languages (18 on most runs), 8 defenses, 7 models, 3 families:
    940,992 replies.
  - 95% case bootstraps (10,000 resamples), paired across models.
  - Three training seeds for Tiny Aya Global's two defenses; one seed for everything else; four single-seed
    ablations.
  - Leak check: word 3-gram Jaccard ≥ 0.8.
- **Best supported positive result.** Against the English-only Tiny Aya defense, trying all 11 languages hijacks
  21.3% of cases (95% CI 15.9–26.9; seeds 20.2, 23.1, 20.7), against 97.1% undefended. Low-resource training
  lowers it to 14.3% (−7.1 [−13.6, −0.8], three seeds).
- **Most important negative result (own words).** "The judge is one LLM. Its agreement with hand labels on its own
  four questions is still being measured, and our hand check of 12 replies found 3 borderline calls." Also:
  "Agent benchmarks sit near their floor … attacks succeed in under 4% of cases even undefended."
- **Benchmark quality.**
  - Strong: held-out languages, the leak check, equal-budget controls, three scoring definitions.
  - Weak: the judge is not validated by humans; the payload is a single benign "Hacked".
- **Demo quality.** No deployed Space. There is an `attack.gif`, a 4-panel findings figure, an 87-second Manim
  video and a slides PDF. A judge watches; they cannot run anything.
- **Communication quality.**
  - Excellent: a logo; a three-line thesis at the top; four numbered findings, each with a number; twelve figure
    types in PNG, SVG and PDF; a 642-line report; a blog article; a submission to an AAAI-27 workshop.
- **Reproducibility.**
  - Bash scripts per stage and three uv environments with vLLM and TRL pinned in `envs/`. `pyproject.toml` is not
    pinned.
  - Needs a 24 GB GPU, the gated Llama, Prompt Guard and Meta-SecAlign models, and an OpenRouter key for the judge.
  - All result CSVs are committed. 69 test functions.
- **Strongest advantage over ActionProof.** Breadth and visual clarity: four findings a judge remembers, each
  with a figure, plus a finished video.
- **Weakest point.** No live demo. The headline judge is unvalidated. Agent-level evidence is at the floor.
- **What ActionProof should learn.** Lead with three or four numbered findings, each tied to one figure. Show the
  scoring-rule sensitivity table pattern: the same episodes under different success definitions.
- **How ActionProof can outperform.** A live, inspectable agent episode (they have none), plus agent-level effects
  well above the floor.
- **Confidence.** High (README, report abstract, splits, limitations, what did not work, and the cost table read).

## 02 Escalation-Safe Tool-Using Aya

- **Project / repository / HEAD / latest update.** `Ashish-Soni08/lexsi-escalation-safe-agent`, `c2a5da5`,
  2026-11-25 (148 commits, 9 `claude/*` branches). Latest work: a soft-reward comparison (Soft-SVeRL), a τ-bench
  pilot, GGUF, and the Stage 2 README.
- **Track and problem.** Track 3. Tool agents meet bot walls, logins, captchas and rate limits. They should take a
  legitimate route or stop and say so, not bypass the block.
- **One-sentence thesis.** With an outcome-only GRPO reward, a 4.8% propensity to escalate (bypass) becomes 99.2% in
  160 steps. A reward in which every exploit-shaped call scores below every honest outcome brings it to 0.0% on the
  test set, with 99% legitimate success, on five Aya models.
- **Pipeline.**
  1. `esa/world.py` simulator plus a rule-based exploit detector.
  2. CuratorKIT hygiene: dedup, `SecretsGate`, `PIIPseudonymizer` (imports).
  3. SFT: clean, then escalation-prone.
  4. AgentTune GRPO, with their own patch to the rollout (`esa/rollout.py`).
  5. AuditKIT `agent_eval`.
  6. CircuitKIT layer ranking for quantization.
  7. A static dashboard.
- **Models and toolkits.** Tiny Aya Global, Fire, Water and Earth, and Aya Expanse 8B. AgentTune, AuditKIT,
  CuratorKIT and CircuitKIT, all imported. Closed models are used only as comparisons (GPT-6 Luna, Claude Haiku 5.5).
- **Experiments.**
  - 320 tasks per set.
  - Sets: test, stress, unseen, fresh, sealed, long conversation, and a real HTTP server (40 tasks).
  - The sealed set was built after the Stage 1 model was frozen.
  - Three GRPO seeds for the recommended recipe; five base models, one seed each; reward ablations (one seed).
  - Quantization: NF4, GPTQ 4- and 3-bit, GGUF.
  - BFCL v3 utility.
  - Paired McNemar tests; 95% intervals.
  - An LLM judge (Qwen2.5-32B) audits the detector: κ 0.87 and 0.84.
- **Best supported positive result.** v4.1 recipe: test escalation 0.0 against 99.2 for the success-only reward.
  Sealed: 16.4 ± 4.7 (mean ± sd over three seeds) against 100.0. McNemar p < 1e-13 on every set. Aya Expanse 8B
  safe model: sealed 0.7 (one seed).
- **Most important negative result (own words).** "Unseen shortcuts are not solved: v4 still escalates on 16 ± 5%
  of blocked sealed tasks and 23 ± 5% of blocked unseen tasks." "The reward depends on a rule-based detector that
  was gamed twice." "There are no human labels yet." "Only sealed is clean for all three seeds."
  "Over-cautious … gives up on 43% of fresh-set tasks whose only route is one the site openly offers."
- **Benchmark quality.**
  - Strong: a sealed set after freeze; a real-server check; adversarial shortcut kinds held out.
  - Weak: about half of the evaluation questions recur in training with a different site; the environment, detector
    and sealed set have one author (they state this designer-bias risk).
- **Demo quality.** A static dashboard: outcomes, training curves, AuditKIT before and after, and an episode
  browser. Replay only, no live inference.
- **Communication quality.** A crisp one-line finding and per-set tables. An ACL-template paper with TikZ figures.
  The README is long (626 lines) but well sectioned.
- **Reproducibility.**
  - Best in class for checking numbers: "Check the numbers (no GPU, about 5 minutes)". It runs 16 tests,
    regenerates every data file byte-identically, fetches about 40 MB of episodes, rebuilds every report, and
    expects `git status` to be empty.
  - Every training command is given as an HF Jobs line with its dollar cost.
  - `requirements.txt` has 6 lines, 2 pinned.
- **Strongest advantage over ActionProof.** It owns "stop and say so". Escalation safety is its core result, with
  three seeds, five models and a sealed set. That is exactly where our H5 is weakest.
- **Weakest point.** Its "escalation" is observable: the block is visible in the tool output. There is no hidden
  state and no case where the agent cannot know. Over-caution is 43% on the fresh set.
- **What ActionProof should learn.** A sealed test built after freeze. Cost per reproduction command. A 5-minute
  check of the numbers on a small download.
- **How ActionProof can outperform.** Show **selective** escalation under *irreducible state ambiguity*, with
  over-escalation measured on degraded-but-resolvable evidence, and a risk–coverage view. That is the
  question they cannot pose in their environment.
- **Confidence.** High.

## 03 Desi Agent

- **Project / repository / HEAD / latest update.** `CyberRik/desi-agent`, `1ae2609`, 2026-11-25 (64 commits,
  15 feature branches, CI `ci.yml`, three contributors tagged [R][K][F]). Latest: a second GRPO seed, Lab updates,
  the training appendix.
- **Track and problem.** Track 3. A multilingual banking support agent (English, Hindi, Bengali, Hinglish,
  Banglish) that must pick the right tool, fill exact arguments, and ask instead of guessing.
- **One-sentence thesis.** A 3.35B Tiny Aya distilled with AgentTune from code-generated tool labels beats its
  Qwen3-14B teacher. RFT and multi-turn GRPO then fix language switches and grounding.
- **Pipeline.** The teacher writes only the customer's words; code writes the tool calls → SFT LoRA → RFT
  rounds 1 and 2 (plus 1,148 grounding rows) → 25-step multi-turn GRPO with `reward_lang` → evaluation against a
  deterministic mock bank → the Lab Space.
- **Models and toolkits.** Tiny Aya Global; Qwen3-14B teacher. Imports: `agenttune.agentic`,
  `auditkit.agent_eval`, `curatorkit.llm.base`.
- **Experiments.**
  - 486 held-out test conversations; 300 language-switch conversations; a reworded OOD set (judge-validated);
    BANKING77 real requests.
  - Two GRPO seeds; McNemar on paired conversations.
  - A blind check of the language detector: 56/60, κ 0.85.
  - The reward was audited against scripted cheating policies before optimization.
- **Best supported positive result.**
  - Task success 45.9% → 97.7% (SFT), 98.6% (GRPO 2), on 486 conversations.
  - Wrong bank actions 20.8% → 1.2–1.8%.
  - Language switch, right action and right language: 34.0% → 78.3%. GRPO 2 fixed 36 and broke 0 against RFT 2.
- **Most important negative result (own words).** "The 381 English grounding rows used in training are BANKING77
  *train* messages … The 0.0% is therefore partly in-distribution." GRPO: "two seeds, one learning rate, no KL
  penalty."
- **Benchmark quality.**
  - Labels are correct by construction (code), the mock bank is deterministic, and there is an OOD reworded set.
  - The test is template-generated from the training distribution, so 97.7% is near the ceiling.
- **Demo quality.**
  - The Lab: a tour (base against fine-tuned on the same message, tool calls shown as bank receipts, and a case
    still wrong); a test explorer over all 486 conversations × models, filterable by language, task and outcome
    ("fixed by fine-tuning / still failing / regressions"); results.
  - A 1080p tour video and a local playground. Replay only, stated plainly.
- **Communication quality.** A banner image with the headline numbers; a dumbbell chart in light and dark modes;
  a short "Stage 2" table; plain-language motivation in the users' own languages.
- **Reproducibility.** Traces and rollouts on the Hub; a CI test suite; `pyproject.toml` not pinned.
- **Strongest advantage over ActionProof.** The judge experience. In 60 seconds a judge can browse every test case
  by outcome, including failures. The visual design is polished and consistent.
- **Weakest point.** The near-ceiling in-distribution test; one base model; a short GRPO run.
- **What ActionProof should learn.**
  - An outcome-filtered explorer ("fixed / still failing / regressions") over the registered test episodes.
  - A tour that shows a failure the team still has.
  - A banner image carrying the two headline numbers.
- **How ActionProof can outperform.** A counterfactual benchmark where the same reply hides different states, so
  success cannot be read from the transcript, plus live inference with state revealed afterwards.
- **Confidence.** High (README in full).

## 04 Drift Monitor

- **Project / repository / HEAD / latest update.** `dev-willbird1936/drift-monitor`, `a258d4c`, 2026-11-25
  (18 commits; recreated on Nov 15; solo). Latest: the Stage 2 paper and video, restyled.
- **Track and problem.** Track 1, problem statement 2: replace a closed-model API. The monitor detects when an agent
  session drifts from what the user authorised.
- **One-sentence thesis.** A fine-tuned Tiny Aya Global emits one digit for drift. A threshold on it decides when to
  call a frontier advisor. This cuts oversight cost and keeps the agent on task.
- **Pipeline.**
  1. A non-Cohere teacher writes 5,417 sessions: 20 categories, 16 drift patterns. Each drift session has a
     near-identical on-course twin that differs only in the user's request.
  2. Plus 1,902 recorded sessions: 90,365 rows in total.
  3. CuratorKIT, then AlignTune 4-bit QLoRA SFT, then AuditKIT (`FunctionScorer`, `PrecomputedModel`).
  4. A Pi agent extension (`driftwatch.py`).
- **Models and toolkits.** Tiny Aya Global; CuratorKIT, AlignTune and AuditKIT, all imported.
- **Experiments.**
  - 770 held-out real sessions.
  - Four unseen work categories.
  - Two frontier models given the same prompt on the same 497 rows.
  - A live closed loop: 256 sessions, 16 tasks, 6 of them unseen.
  - Pooled 95% bootstrap intervals; one training seed (seed 42).
- **Best supported positive result.**
  - AUROC 0.552 → 0.960; false alarms on clean steps 100% → 5.4%.
  - Closed loop: unrequested files 2.48 → 0 per session; task checks 84% → 99%.
  - 1.4 advisor calls per drifting session, against 4.8 for reviewing every turn.
- **Most important negative result (own words).** "Advice can't undo a deletion, an agent told to resist correction
  ignored every note, and drift written inside a requested file can go unseen." "Labels are model-generated." "Sessions are
  written, not executed."
- **Benchmark quality.** Counterfactual twins (the same pattern as our TXN quadruples) and a held-out real split.
  Labels are model-generated; two blind judges agreed with them more than with each other.
- **Demo quality.** Recorded runs are replayed step by step, with the reasoning, tool calls and the drift score side
  by side. A "Try it live" tab runs a real agent with no key, through a ZeroGPU scorer Space and a spend-capped
  shared OpenRouter key. The demo has idle and offline states, and the page wakes the scorer Space if it sleeps.
- **Communication quality.**
  - An "At a glance" block with a "Where it falls short" bullet; a hero screenshot.
  - A 2:48 video with a slides PDF and an .srt file; an EMNLP-template paper.
  - A runnable Python snippet in the README.
- **Reproducibility.** Seeds fixed; data and model public; cost per million steps stated.
- **Strongest advantage over ActionProof.** A live demo with graceful degradation (wake, idle and offline states)
  and a finished video with captions and slides.
- **Weakest point.** One training seed. Labels come from models. Sessions are written, not executed.
- **What ActionProof should learn.** Explicit offline and idle states in the live demo; auto-wake; a cost per
  million operations; a finished 2:48 video with a slides PDF.
- **How ActionProof can outperform.** Executed tools and a database verifier (theirs are written sessions with model
  labels); three seeds; registered tests.
- **Confidence.** High.

## 05 Relapse

- **Project / repository / HEAD / latest update.** `itssaharsh/relapse`, `f4ed78d`, 2026-11-26 23:28 UTC
  (129 commits; very active; self-review "panel rounds" 1–5). Latest: the sixth preregistration's results,
  attestations, the tokenizer survey.
- **Track and problem.** Track 2b. Does unlearning of hazardous biology survive deployment: 4-bit files and the
  tokenizer setting a user's copy loads?
- **One-sentence thesis.** An outdated tokenizer setting overstated RMU unlearning at the default layers in 16 of 16
  runs, but not at layers 3–5 (0 of 16). Four-bit forgetting usually holds (107 of 109), but single files can relapse.
- **Pipeline.** SafeTune RMU on five Tiny Aya variants → evaluation through both tokenizer settings and four
  quantized formats → a public survey of affected copies → a checker tool.
- **Models and toolkits.** Tiny Aya Base, Global, Fire, Water and Earth. SafeTune; CircuitKIT and AuditKIT
  imported; CuratorKIT mentioned in docs.
- **Experiments.**
  - A map of 150 runs.
  - Six preregistrations in `results/PREREGISTRATION*.md`: 25 hypotheses, of which 11 held, 2 held in part,
    11 failed and 1 could not be tested.
  - Exact McNemar with Holm; seeds repeated for relapse cases.
- **Best supported positive result.** At the default layers, 16 of 16 counting runs regain about 47% of the audited
  forgetting through the official tokenizer. At layers 3–5, 0 of 16 do.
- **Most important negative result (own words).** A public correction on Nov 21: "Our Stage 1 results contained a
  serious error, and we have withdrawn two of the claims." The layer effect "is confounded with that damage".
- **Benchmark quality.** WMDP-bio and MMLU are standard. Integrity is the strength: the same weights are scored
  through both settings in the same job.
- **Demo quality.** A static results explorer and a separate article Space. No inference.
- **Communication quality.** The title is long and technical. An "In brief" table leads, and the correction is
  displayed prominently.
- **Reproducibility.** HF Jobs scripts; every job is checked to read only public repositories; references are
  checked with a fetch record.
- **Strongest advantage over ActionProof.** Real-world impact evidence. A survey found the Kaggle copy and
  418 derived Hub repositories (64,619 downloads) still carrying the outdated tokenizer, and they ship a checker.
  Their integrity story (a public correction) is memorable.
- **Weakest point.** The narrowed claim is a tooling artefact rather than a method, and the layer effect is
  confounded.
- **What ActionProof should learn.** A "What we got wrong" line in the brief, and counting registered hypotheses as
  held / held in part / failed / untestable.
- **How ActionProof can outperform.** An end-to-end agent system with a live model and a reusable benchmark. They
  have no system to show.
- **Confidence.** High.

## 06 Hinglish Expense Agent (TinyAya-Hinglish-Agent)

- **Project / repository / HEAD / latest update.** `PRINCE2-AI/TinyAya-Hinglish-Agent`, `36ef775`, **2026-11-18**
  (143 commits). **No commits during Stage 2.** The CI badge points to `lexsi-agentic-hinglish`.
- **Track and problem.** Track 3. A Hinglish shared-expense agent with tools: ledger, currency conversion, settling
  balances.
- **One-sentence thesis.** Tiny Aya learns reliable Hinglish tool calling, checked against the final ledger state.
  A deterministic CPU guard under the model keeps attacker money out of the ledger in 208 of 208 injected episodes.
- **Pipeline.** Generated tasks → LoRA SFT (mixed data) → an integer-paise ledger oracle → AuditKIT re-score → the
  guard (`hinglish_agent/guard.py`).
- **Models and toolkits.** Tiny Aya Global; AgentTune (docs and code, no imports found); AuditKIT through an
  `auditkit_eval` wrapper.
- **Experiments.**
  - Splits: hard-30, hard-90 (three seeds), hard-257 built Nov 13, Devanagari-30, six injection splits (208), and a
    blind guard set (8).
  - Wilson intervals and McNemar; task bootstrap.
- **Best supported positive result.**
  - hard-257: 0/257 → 235/257 (Wilson 87.4–94.3%; 235 fixed, 0 broken, p = 3.6e-71).
  - hard-90: 3/90 → 84/90 (three seeds pooled 93.0%).
  - Tool calls per task 7.03 → 3.57 (the oracle needs 3.57).
- **Most important negative result (own words).** "GRPO ran twice and moved nothing (0/0, p = 1.0); that null is
  reported, not hidden." The blind set scores 5/8: "the guard is a filter, not a proof."
- **Benchmark quality.** Deterministic oracle, with no judge. A table of "six shortcuts a tuned agent could take,
  and the mechanism that blocks each". Small n on the safety splits.
- **Demo quality.**
  - A "Judge's 60-second view": four numbered steps.
  - A frozen-episode toggle (MATCH against MISMATCH).
  - "Try the guard yourself" on CPU, which never touches the GPU quota and includes a published blind spot.
  - A live ZeroGPU Space (asleep at audit); an episode gallery of 90 held-out tasks; an attacks gallery of 72; a
    1:44 narrated video; a before/after GIF.
- **Communication quality.** The best judge-facing README in the set. Every figure names the run directory it
  comes from.
- **Reproducibility.** `verify_all.py --with-tests` runs 31 checks with no GPU and no token. `MANIFEST.md` holds
  the sha256 of 74 summaries, the adapter and 11 evaluation files, and `manifest.py --check` re-hashes them from
  the Hub.
- **Strongest advantage over ActionProof.** Speed to trust. A numbered 60-second judge path, a CPU demo path
  separate from the GPU quota, and a hash manifest checked against the Hub.
- **Weakest point.** Stage 1 only (no Stage 2 updates). A rule-based guard. Small safety splits. One base model.
- **What ActionProof should learn.**
  - A numbered "Judge's 60-second view".
  - A "Why these numbers cannot be gamed" table.
  - A CPU path that never touches the GPU quota.
  - A Hub hash manifest with `--check`.
- **How ActionProof can outperform.** A harder problem (hidden state behind identical replies, not visible injected
  text), three seeds, Stage 2 science, and a live model that is awake.
- **Confidence.** High.

## 07 Lost in Refusal

- **Project / repository / HEAD / latest update.** `lost-in-refusal/lexsi-multilingual-safety`, `2b240ca`,
  2026-11-25 (317 commits; each analysis registered in a commit before its runs are read).
- **Track and problem.** Track 2a/2b. Benign multilingual fine-tuning erodes refusals (Korean, untrained Swahili).
  SafeTune SafeGrad as the repair.
- **One-sentence thesis.** Benign LoRA on English, Korean and Chinese raises Korean attack success more than
  English, and lowers refusals in untrained Swahili. SafeGrad repairs it. All of this holds over three seeds.
- **Pipeline.** Bactrian-X (1,800 rows) curated with CuratorKIT → plain against SafeGrad LoRA → MultiJail
  (291 families × 4 languages), XSTest and held-out tasks → a Qwen3Guard judge → human reading checks.
- **Models and toolkits.** Tiny Aya Global and Earth; IBM Granite 4.1 3B as an outside family; Aya Expanse 8B in
  pilots. SafeTune, AlignTune (as a second training stack), AuditKIT and CuratorKIT imported; CircuitKIT for a
  quantization plan.
- **Experiments.**
  - Three seeds; registration sections 14–19, each added before its runs.
  - Human labels: 120 for judge agreement and 300 for reading checks.
  - Replication on a second machine (an L4); dose-response; NF4; a refusal-direction probe.
  - Bootstrap over prompt families, pooled over seeds.
- **Best supported positive result.** Korean drift passes in 3 of 3 seeds (Korean minus English gap +11.8
  [+6.8, +16.7]). SafeGrad repair pooled −8.4 [−11.3, −5.5].
- **Most important negative result (own words).** "One safety judge. A second judge failed our agreement check."
  "A person's reading of the Korean rise (K2) did not pass … p = 0.22." "SafeGrad does not protect better than the
  simple fix."
- **Benchmark quality.** The strongest validation of human labels against a judge in the set, and a family outside
  Tiny Aya.
- **Demo quality.** A static Space, last modified Nov 6, so it predates Stage 2.
- **Communication quality.** A precise table of checks and results. The prose is dense and long.
- **Reproducibility.** Outputs dataset; pinned per-lane requirement files.
- **Strongest advantage over ActionProof.** Human-label validation and breadth of replication (a second machine, a
  second training stack, a second family).
- **Weakest point.** The headline rests on one automatic safety judge, and the human checks did not pass.
- **What ActionProof should learn.** Put the commit and timestamp of each registration next to the result it
  governs.
- **How ActionProof can outperform.** Our verifier is deterministic, so no human-label gap exists for the factual
  state. We also have a live system.
- **Confidence.** High.

## 08 EVIDENCELOCK

- **Project / repository / HEAD / latest update.** `khamirdesai/evidencelock-track3`, `6c666c1`, **2026-11-16**
  (7 commits; no Stage 2 activity; CI; Colab).
- **Track and problem.** Track 3, problem statements 7 and 9. An invoice agent must verify against a trusted ledger
  before approving or flagging, while vendor documents inject text.
- **One-sentence thesis.** SFT on sandbox-verified trajectories teaches the evidence workflow. A plausible
  decision-level "safety" loss breaks it. Receipt-reward GRPO is a null result, with a loophole that positive-only
  audits miss.
- **Pipeline.** Sandbox with authorization → SFT (C0) → CVT ablation (C1) → GRPO (two seeds) → T3-SFT and
  GRPO-3 follow-ups → AuditKIT re-score → batched serving cost.
- **Models and toolkits.** Tiny Aya Global; AgentTune and AuditKIT.
- **Experiments.** Hard panel of 120; fresh panel of 264; 95% paired group-bootstrap intervals on differences;
  pre-registered follow-ups.
- **Best supported positive result.** Secure completion 33 → 91 of 120 [40.0, 55.8 pp]. Verification before acting
  60/96 → 96/96.
- **Most important negative result (own words).** CVT: 91 → 28 of 120 (−52.5 pp). GRPO-2: 216 → 216 of 264
  ("null"). "Unparseable output (−0.40) cost less than a schema-valid wrong flag (−0.55), and the audit checked only
  positive rewards."
- **Benchmark quality.** A sandbox with authorization; small panels.
- **Demo quality.** A static replay Space.
- **Communication quality.** The best claim labelling in the set: every row says "measured, pre-registered,
  negative / null / not positive".
- **Reproducibility.** `pytest` runs 53 checks of the headline numbers against the evidence files in about one
  second. Colab quickstart.
- **Strongest advantage over ActionProof.** A per-row label column on the headline table, and a one-second test of
  the numbers.
- **Weakest point.** Inactive in Stage 2; GRPO null; small n.
- **What ActionProof should learn.** Add a "Status" column (registered supported / registered not supported /
  descriptive / exploratory) to every headline table.
- **How ActionProof can outperform.** Larger, registered, multi-seed effects; a counterfactual benchmark; live
  inference.
- **Confidence.** High (README in full).

## 09 Safe from Fine-Tune to Phone

- **Project / repository / HEAD / latest update.** `SushovitNanda/Safe-from-Fine-Tune-to-Phone`, `b6fda2d`,
  2026-11-15 (8 commits; no Stage 2 activity).
- **Track and problem.** Track 2: medical LoRA erodes safety; a steering repair that survives GGUF.
- **One-sentence thesis.** Benign medical LoRA doubles attack success (18.2% → 36.3%). A run-time steering vector
  cuts it to 26.3% and survives Q3_K_M.
- **Pipeline.** ClinicSafe-Hi (102 harmful plus 102 benign twins in English, Hindi and Hinglish), with Aya
  Red-teaming and XSTest → LoRA → a steering vector → GGUF on llama.cpp.
- **Models and toolkits.** Tiny Aya Global; SafeTune and CircuitKIT (from the README).
- **Experiments.** A 1,612-prompt frozen test; 95% paired bootstrap; one training run per model (seed 13); a
  prereg with five dated amendments; `MANIFEST.sha256.json`.
- **Best supported positive result.** Drift +18.1 [15.0, 21.3]; steering −10.1 [−12.5, −7.6]; retention at
  Q3_K_M 1.01.
- **Most important negative result (own words).** "One training run per model; bootstrap resamples prompts, not
  training runs." "No demo Space, video or slides accompany this submission."
- **Benchmark quality.** Harmful/benign twins (counterfactual style); judge κ against 300 AI-drafted labels approved
  by the author.
- **Demo quality.** None.
- **Communication quality.** A one-line claim; a 73-page PDF.
- **Reproducibility.** A hash manifest and a Mac lock file.
- **Strongest advantage over ActionProof.** Pairing harmful prompts with benign twins in a safety set.
- **Weakest point.** No demo; one seed; inactive in Stage 2.
- **What ActionProof should learn.** State in one sentence what the bootstrap resamples (prompts, not runs). We
  should do the same for tasks against seeds.
- **How ActionProof can outperform.** A live system, multiple seeds, Stage 2 results.
- **Confidence.** Medium-high.

## 10 Lexsi AI Alignment ("Does safety survive an attack in another language?")

- **Project / repository / HEAD / latest update.** `dhairya-pandya/Lexsi-AI-Alignment`, `d0f2c62`, 2026-11-20
  (98 commits). Latest: an ACL-format paper.
- **Track and problem.** Track 2. Fine-tuning attacks in Bengali bypass English-tested SafeTune defenses.
- **One-sentence thesis.** Defenses hold in English (96–100% refusal) but leave 59–61% of harmful Bengali prompts
  answered. The attack's weight change is low-rank, and removing its top singular direction (k = 1) drops Bengali
  harm from 50% to 14%.
- **Pipeline.** Tiny Aya Fire → SafeTune (Lisa, Booster, Vaccine, RepNoise) → a Bengali poisoned fine-tune →
  evaluation in two languages → post-hoc SVD projection.
- **Models and toolkits.** Tiny Aya Fire; SafeTune.
- **Experiments.** Matched English and Bengali evaluation; MMLU utility. The README states no seeds or intervals;
  details are in the paper.
- **Best supported positive result.** Bengali harm 50% → 14% with `project_dw` k = 1, MMLU preserved.
- **Most important negative result.** Training-time anchoring "didn't work … anchoring on the union made it worse"
  ("repair beats prevention").
- **Benchmark quality.** Not assessed in depth (paper not read).
- **Demo quality.** A Vercel site ("live demo and interactive results", an animated walkthrough, `repair.mp4`) and a
  Space `app.py`.
- **Communication quality.** A strong banner with the finding; a short README (62 lines).
- **Reproducibility.** uv and pytest with a tiny random Cohere2 model for fast tests.
- **Strongest advantage over ActionProof.** A single striking visual contrast ("English looks fine; Bengali is wide
  open").
- **Weakest point.** README-level evidence is thin; seeds are not stated.
- **What ActionProof should learn.** A hero image built around one contrast: here, the same "ok" and two databases.
- **How ActionProof can outperform.** Registered multi-seed evidence and a live agent.
- **Confidence.** Medium (README only).

## 11 Against the Safety

- **Project / repository / HEAD / latest update.** `Bhaveshmeghwal21/against-the-safety`, `c5a093c`, 2026-11-16
  (144 commits; no Stage 2 activity; CI; pinned lock files).
- **Track and problem.** Track 3, problem statement 4 (RAG). Search, cite, and abstain when the evidence is missing.
- **One-sentence thesis.** QLoRA through AgentTune teaches Tiny Aya to answer with citations or abstain. Two
  training-free repairs (fail-closed, self-check) cut unsupported answers.
- **Pipeline.** HotpotQA scripted episodes → QLoRA (AgentTune) → a fail-closed rule and a self-check → a sealed
  test → a trace viewer.
- **Models and toolkits.** Tiny Aya Global (pinned revision); AgentTune 1.1.3 and AuditKIT 1.1.3 (pinned).
- **Experiments.** A sealed test of 938 episodes (about 40% with hidden evidence); a fresh 679-episode test opened
  once; GPT-6 Luna and DeepSeek V4 Flash on 925 shared episodes; seed 13 (one run).
- **Best supported positive result.**
  - Answers when the evidence is hidden: 281/375 → 42/375.
  - Correct cited answers 5.7% → 46.3%.
  - Macro success 58.0% **against 42.0% for refusing everything** (reported explicitly).
- **Most important negative result (own words).** "Every later training change failed its own pre-registered release
  rule." The fresh-test check gave +0.9 [−0.3, +2.1]. Single-passage SQuAD is "close to the refuse-everything score".
- **Benchmark quality.** A sealed test; the refuse-all baseline reported; four measures, because macro success can
  be gamed by refusing.
- **Demo quality.** A static trace viewer ("the model does not run").
- **Communication quality.** A glossary for non-experts; four measures.
- **Reproducibility.** Pinned CPU and GPU lock files.
- **Strongest advantage over ActionProof.** It reports the **refuse-everything** baseline beside abstention metrics.
- **Weakest point.** One run; English only; inactive in Stage 2.
- **What ActionProof should learn.** Report an *always-escalate* policy next to every escalation metric, so a
  higher escalation rate cannot be mistaken for selectivity.
- **How ActionProof can outperform.** Selective escalation measured against always-escalate and never-escalate, with
  a risk–coverage curve.
- **Confidence.** High.

## 12 Same Model, Different Language

- **Project / repository / HEAD / latest update.** `Hydra-Of-Malice/same-model-different-language`, `ea1b7fa`,
  2026-11-24 (156 commits; CI; EMNLP paper).
- **Track and problem.** Track 2b. Per-language safety gaps between Tiny Aya Earth and Global; repairs without
  retraining.
- **One-sentence thesis.** Only the Javanese gap is confirmed under a corrected bar. No repair met all pre-set
  conditions. Circuits located refusal, but circuit-guided edits did no better than edits at random layers.
- **Pipeline.** MultiJail and Aya Red-teaming on held-out prompts → three judge configurations, including Command A →
  weight-space repairs (merge-back, RESTA, steering, circuit merge) → GGUF.
- **Models and toolkits.** Tiny Aya Earth and Global; Command A as a judge; CircuitKIT and others.
- **Experiments.** McNemar with a multiplicity-corrected bar (0.0036); 95% CIs; two judge pairs plus Command A.
- **Best supported positive result.** Javanese gap +9.7 (+4.0 to +15.3), McNemar p = 0.0023; Command A
  confirms +10.8 (p = 0.0013).
- **Most important negative result (own words).** "Thai, the language the repairs targeted, was not confirmed … so
  every repair result is exploratory." "No repair met all pre-set conditions."
- **Benchmark quality.** A corrected multiple-testing bar; judges checked against each other.
- **Demo quality.** A `space/` folder (not inspected live).
- **Communication quality.** An honest but negative headline.
- **Reproducibility.** A reproduction notebook; CI.
- **Strongest advantage over ActionProof.** Command A, Cohere's own judge, as one of the validators.
- **Weakest point.** The headline is mostly null.
- **What ActionProof should learn.** State the corrected significance bar next to each p-value.
- **How ActionProof can outperform.** Positive registered effects plus a reusable benchmark.
- **Confidence.** High (README).

## 13 RewardShield

- **Project / repository / HEAD / latest update.** `shro-2002/rewardshield`, `35a85df`, 2026-11-22 (39 commits; CI
  added Nov 22; `uv.lock`).
- **Track and problem.** Track 3, problem statements 9 and 6. False accepts by SQL execution rewards.
- **One-sentence thesis.** Mutation-guided extra databases remove a small but replicated share of false accepts at
  equal cost. Filtering training data with the better verifier did not change downstream accuracy.
- **Pipeline.** Spider with a `run_sql` tool (AgentTune format) → five verifiers against an independent test-suite
  oracle rebuilt with Zhong et al.'s generator → SFT arms filtered by each verifier.
- **Models and toolkits.** Tiny Aya Global; AgentTune (the verifier runs as a reward callable inside AgentTune's
  rollout scoring); AuditKIT agent evaluation.
- **Experiments.** 1,360 labelled trajectories; schema-disjoint splits; a git-timestamped preregistration; an
  external audit reproduced line by line.
- **Best supported positive result.** False accepts 3.88% → 1.39% (targeted-4) on dev-A. The audit later showed this
  size depended on one random database draw: averaged over 200 draws, the gap is −0.25 [−0.59, −0.00].
- **Most important negative result (own words).** H3 inconclusive: −1.06 [−4.22, +1.85], minimum detectable effect
  about 4.3. Corrections dated: "H2 size (about 7× overstated on dev-A)".
- **Benchmark quality.** The strongest independence in the set. The oracle is independent in code, and an external
  audit reproduced the analysis.
- **Demo quality.** None (a Google Drive link).
- **Communication quality.** A "For judges: one minute" block with Solid / Found / Corrected after the audit.
- **Reproducibility.** A script that re-derives every headline number.
- **Strongest advantage over ActionProof.** An **independent** verifier. Our reference verifier shares a
  specification and an author with the main one.
- **Weakest point.** Tiny effects; no demo.
- **What ActionProof should learn.** State in the README that our reference verifier is a consistency check (same
  author and specification), not independent ground truth.
- **How ActionProof can outperform.** Larger effects, a live system, and verifier mutation tests (if added).
- **Confidence.** High.

## 14 VeriLoop

- **Project / repository / HEAD / latest update.** `AndrewPrakash1/veriloop`, `7f17615`, 2026-11-21 (9 commits).
- **Track and problem.** Track 3. Compliance steps skipped by small agents; rewards from database state plus an
  audit log.
- **One-sentence thesis.** (Planned.) Pay agents only when the database and the append-only audit log show that the
  work happened.
- **Pipeline.** Episode contracts (YAML) → a deterministic reward (hard gate Γ) → a verified generator with frozen
  splits.
- **Models and toolkits.** No Cohere model or Hub artefact found.
- **Experiments.** "This folder is a *plan*, not a report. Nothing in VeriLoop has been trained at the time of
  writing."
- **Best supported positive result.** None.
- **Most important negative result.** It reports a reward-hacking hole found in its own reward engine (commit
  message).
- **Benchmark quality.** Contracts plus a leakage control, as designed.
- **Demo quality.** None.
- **Communication quality.** A clear statement of the problem as a measurement.
- **Reproducibility.** Not applicable.
- **Strongest advantage over ActionProof.** Declarative episode contracts as a specification format.
- **Weakest point.** No results.
- **What ActionProof should learn.** Our TXN postconditions could be published as a machine-readable contract per
  task.
- **How ActionProof can outperform.** Already does: results, models, demo.
- **Confidence.** High (the repository says it has no results).

## 15 Reward Hacking Observatory

- **Project / repository / HEAD / latest update.** `Swateya03/reward-hacking-observatory`, `b1a8cb9`, 2026-11-27
  01:52 IST (75 commits; still active).
- **Track and problem.** Reward hacking in tool agents: 30 tasks × 3 ambiguity levels, 17 tools (MCP, FastAPI,
  SQLite), 5 hack types, 3 rewards.
- **One-sentence thesis.** The same trajectory can score high on a naive reward and low on a robust reward.
- **Pipeline.** A workspace server → a Claude Haiku 5.5 agent through the API → rescoring under three rewards.
- **Models and toolkits.** **No Cohere model.** The agent is `claude-haiku-5-5`. No Hub artefacts.
- **Experiments.** 300 raw runs rescored to schema v8; 396 tests.
- **Best supported positive result.** None published: "Results pending … Earlier figures in this README were taken
  from test fixtures and sample data … and have been removed."
- **Most important negative result.** It withdrew fixture-based numbers.
- **Benchmark quality.** A taxonomy and tests; no published measurement.
- **Demo quality.** `demo.py` locally; nothing deployed.
- **Communication quality.** Badges and a taxonomy.
- **Reproducibility.** Not applicable yet.
- **Strongest advantage over ActionProof.** A reward-hacking taxonomy as a teaching device.
- **Weakest point.** Likely non-compliant (no Cohere model) and no results.
- **What ActionProof should learn.** Nothing material.
- **How ActionProof can outperform.** Already does.
- **Confidence.** High.

## 16 Contextual Pseudonymizer

- **Project / repository / HEAD / latest update.** `wenbinio/lexsi-contextual-pseudonymizer`, `61b5321`,
  2026-11-15 (2 commits).
- **Track and problem.** Track 1, problem statement 5. Extraction of direct and indirect identifiers.
- **One-sentence thesis.** A two-pass QLoRA Tiny Aya extracts identifiers with near-perfect recall on synthetic
  records.
- **Pipeline.** Synthetic records → CuratorKIT → AlignTune QLoRA (two passes) → AuditKIT span metrics.
- **Models and toolkits.** Tiny Aya Global, pinned to revision `af89d219`.
- **Experiments.** 240 reused records plus 120 fresh; one run.
- **Best supported positive result.** Exact entity recall 66.0% → 94.6% on the fresh set; precision 84.6%.
- **Most important negative result.** Precision drops from 99.0% on the reused set to 84.6% on the fresh set.
- **Benchmark quality.** Synthetic English only.
- **Demo quality.** A local `demo_server.py`; nothing deployed.
- **Communication quality.** Short and precise.
- **Reproducibility.** Pinned revision; a notebook.
- **Strongest advantage over ActionProof.** None material.
- **Weakest point.** Small and synthetic.
- **What ActionProof should learn.** Pin the base model by revision in every card. We should check that ours do.
- **How ActionProof can outperform.** Already does.
- **Confidence.** High (short README).

## 17 Tamil Clinical Triage

- **Project / repository / HEAD / latest update.** `SS-360/lexsi-hackathon-tamil-clinical-triage`, `f56334a`,
  2026-11-12 (a single commit).
- **Track and problem.** Tracks 1 and 2a. Clinical triage in Tamil and Tanglish.
- **One-sentence thesis.** Tiny Aya Fire plus a deterministic triage engine is safe on 40 probes.
- **Pipeline.** CuratorKIT → AlignTune → a rule engine → AuditKIT (with a gpt-oss-120b judge).
- **Models and toolkits.** Tiny Aya Fire.
- **Experiments.** 40 held-out probes (10 per category); no intervals, no seeds.
- **Best supported positive result.** Safety pass 52.5% → 97.5% (this includes the rule engine).
- **Most important negative result.** None stated.
- **Benchmark quality.** n = 40.
- **Demo quality.** None deployed.
- **Communication quality.** Badges and a thumbnail.
- **Reproducibility.** Notebooks.
- **Strongest advantage over ActionProof.** None material.
- **Weakest point.** Tiny n; the gains are confounded with the rule engine.
- **What ActionProof should learn.** Nothing material.
- **How ActionProof can outperform.** Already does.
- **Confidence.** High.

## 18 ICSA Grounded Assistant

- **Project / repository / HEAD / latest update.** `Incharajayaram/icsa-grounded-assistant`, `44032d3`,
  2026-11-15 (3 commits).
- **Track and problem.** Track 1. An ICS/OT advisory assistant that cites its source or abstains.
- **One-sentence thesis.** QLoRA teaches citing and abstaining on CISA ICS advisories.
- **Pipeline.** Ingestion → CuratorKIT QA generation (with `cohere_chat/command-r7b-12-2024` as the generator) →
  QLoRA of the core model.
- **Models and toolkits.** **The core model is `Qwen/Qwen2.5-7B-Instruct`, not a Cohere Labs model** (confirmed
  through the Hub: base_model = Qwen/Qwen2.5-7B-Instruct). Cohere appears only as a data generator. Under the
  official rule ("The core submission should use the listed models from Cohere"), it is likely ineligible.
- **Experiments.** 12 held-out cases.
- **Best supported positive result.** All 6 unanswerable questions refused and all 6 answerable ones answered.
- **Most important negative result.** It found the model answering a trivia question and fabricating once, and
  retrained.
- **Benchmark quality.** n = 12.
- **Demo quality.** None.
- **Communication quality.** The write-up is a claude.ai artifact.
- **Reproducibility.** Colab.
- **Strongest advantage over ActionProof.** None.
- **Weakest point.** Eligibility.
- **What ActionProof should learn.** Nothing.
- **How ActionProof can outperform.** Already does.
- **Confidence.** High.

## 19 FinSafe

- **Project / repository / HEAD / latest update.** `dixitkeshav/finsafe`, `8ddaf37`, 2026-11-15 (34 commits; pinned
  lock files; three people).
- **Track and problem.** Track 2b/1. Does finance adaptation erode refusals? Steering as the repair.
- **One-sentence thesis.** On this pilot, finance SFT raised valid JSON and finance accuracy, and did not measurably
  erode safety. Steering at strength 20 over-refuses.
- **Pipeline.** CuratorKIT on 2,500 TAT-QA rows (954 kept) → 100-step QLoRA → SafeTune steering at layer 18.
- **Models and toolkits.** Tiny Aya Global (pinned revision); CuratorKIT, SafeTune, AuditKIT, AlignTune.
- **Experiments.** 200 finance questions; 12 prompts per safety cell; group bootstrap; configuration hashes.
- **Best supported positive result.** Valid JSON 64% → 94%; finance accuracy 7/200 → 14/200.
- **Most important negative result (own words).** "Safety erosion was not observed on this pilot." Strong steering
  raises benign refusal to 41.7%.
- **Benchmark quality.** Tiny safety n (12).
- **Demo quality.** A static Space.
- **Communication quality.** Honest about being a pilot.
- **Reproducibility.** Lock files and hashes.
- **Strongest advantage over ActionProof.** None material.
- **Weakest point.** n.
- **What ActionProof should learn.** A label-sensitivity analysis: rescore under an alternative reading of borderline
  labels.
- **How ActionProof can outperform.** Already does.
- **Confidence.** High.

## 20 Recovery Agent, Taught by Rules

- **Project / repository / HEAD / latest update.** `qasimjethwa/recovery-agent-taught-by-rules`, `4df7da0`,
  2026-11-13 (4 commits; no Stage 2 activity).
- **Track and problem.** Track 3. Failed-payment recovery plans executed as tool calls, taught by a rule engine.
  This is the closest *domain* to ours.
- **One-sentence thesis.** Aya Expanse 8B with LoRA learns to execute a production rule engine's recovery plans.
- **Pipeline.** A port of the Recovery Copilot rule engine → 400 training cases (8 failure reasons) → LoRA SFT →
  128 held-out cases scored against an independent oracle re-implementation → a corrupt-prediction negative
  control.
- **Models and toolkits.** Aya Expanse 8B; AgentTune (the SFT step).
- **Experiments.** 128 held-out cases; one run; no intervals.
- **Best supported positive result.** Exact end-to-end success 0/128 → 47/128 (36.7%); structural 108/128.
- **Most important negative result.** Exact success is still only 36.7%.
- **Benchmark quality.** An independent oracle and a negative control. The failure reason is given in the input:
  there is no hidden state and no ambiguous outcome.
- **Demo quality.** A static replay of predictions.
- **Communication quality.** An `ENTRY-BLUEPRINT.md` that maps every requirement to an artefact.
- **Reproducibility.** The dataset regenerates byte-identically.
- **Strongest advantage over ActionProof.** A grounding in a real production rule engine (a real-world origin).
- **Weakest point.** No hidden execution state; one run; small.
- **What ActionProof should learn.** Name the real-world rule source when one exists. Ours is the idempotency and
  reconciliation practice of payment APIs; cite it.
- **How ActionProof can outperform.** Already does on this problem: ambiguous outcomes, a verifier, multiple seeds,
  a live model.
- **Confidence.** High.
