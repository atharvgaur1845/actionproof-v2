# Novelty positioning (final)

**Date.** 2026-11-27.

**Sources.**
- `docs/research/NOVELTY_AUDIT.md`: 84 entries, verified in earlier sessions.
- `docs/research/NOVELTY_MATRIX.md`.
- Searches and fetches made this session: AgentAbstain, HiL-Bench, EpiSEM-Bench, Evidence-Guided Recovery,
  Stripe's documentation.

**Rules.**
- Only entries whose page was fetched are described in detail.
- Search-snippet-only items are marked as such.
- Arxiv identifiers are as the audit recorded them.

## 1. The closest 15 prior works

| # | Work | Task | Observability | Faults / execution model | Learned policy? | Verifier | What ActionProof adds / does not add |
|---|---|---|---|---|---|---|---|
| 1 | **"Did It Happen?"**, Research Square rs-10730245 (partly verified: title only on the page; design from the audit) | Recovery after a timeout on a side-effecting call | The agent sees an identical timeout; the evaluator knows the hidden commit | Paired hidden-commit units; a status affordance and a stable key as controls | No (evaluation) | Hidden commit state | **The same observation-equivalent design as TXN's quadruples.** ActionProof adds training, `ok` + phantom alongside timeouts, partial and duplicate effects, and the unresolvable regime. The paired design is **not ours**. |
| 2 | **LIMBO**, arXiv 2609.29095 | Where exactly-once safeguards should live | Six services, 12 fault modes, 25,930 episodes, nine models | Duplicates, late commits, read-back possible or not | No (evaluation) | Service state | It already shows read-back and idempotency keys cut duplicates (28% → 4%) and that agents claim success in 90% of duplicate episodes. ActionProof adds trained agents and registered training contrasts. |
| 3 | **Verified Tool Calls**, arXiv 2608.02645 | Non-atomic tools: timeouts after dispatch, delayed visibility, partial updates | A wrapper reads the state | Verify-before-retry, idempotency keys, postconditions | No (explicitly no RL) | Postcondition checks | It is the scripted-wrapper version of our B4 reference. ActionProof's learned agents still trail B4 (TXN VTC 0.724 against 0.864); this sets the bar we report. |
| 4 | **Evidence-Guided Recovery**, arXiv 2610.41453 (fetched) | Recovery under ambiguous outcomes and partial execution | Public evidence; states compatible with it | Failed, committed or in flight | No (planner plus effect adjudicator; prompt-based) | Certified completions; handoff is an explicit outcome | 58 paired units, prompt interfaces. **ActionProof trains a 3.35B model**, measures escalation and over-escalation rates, and runs on a 440-task locked benchmark. |
| 5 | **UndoBench**, arXiv 2610.05622 (abstract) | Task competence against recovery capability | State oracles | Naive retry duplicates in 53% | No | State oracles | The same measurement philosophy; no training. |
| 6 | **RetryBench**, arXiv 2610.15884 | Retry permission under timeouts | Prompt variants | Idempotency-key wording | No | — | Prompt sensitivity only. |
| 7 | **AgentAbstain**, arXiv 2607.10059 (search result; NeurIPS 2026 listing) | Should the agent act at all? | 263 paired tasks, 42 sandboxes | Ambiguity, conflicting constraints, tool failures | No (17 frontier models evaluated; best 59.5% paired accuracy) | Paired labels | Abstention in general; flags "post-hoc abstention" (irreversible action before abstaining). **ActionProof's SEL1 trains** selective escalation and measures abstention *after* a write whose outcome is hidden. |
| 8 | **HiL-Bench**, arXiv 2604.09408 (search result) | When to ask a human | `ask_human()` available | SWE-Bench Pro and BIRD tasks | No | Task success | Models ask correctly in only 4–24% of cases. Not tool-outcome ambiguity. |
| 9 | **EpiSEM-Bench**, arXiv 2610.31650 (fetched) | Safe action when a tool response omits a safety-relevant fact | 300 matched unsafe/safe pairs, six domains | Suppressed facts; a "recovery access" condition | No (ten models evaluated) | Unsafe execution | Its matched pairs and recovery access resemble SEL1's degraded-but-resolvable contrast. No training, no hidden write outcome, no escalation-rate metric. |
| 10 | **SURE-Act** ("Agents' Last Doubt"), arXiv 2610.24194 | Act against ask against escalate | Clone env, force each meta-action | An explicit Commit-Irreversible class | A ranking controller | Critical-error labels | The closest *learned* escalation work. It forks the env to score meta-actions. ActionProof's SEL1 uses demonstrations and tests selectivity against a transient-outage regime held out at test time. |
| 11 | **CASEL**, arXiv 2610.36570 | Runtime mode choice (execute, verify, sandbox, clarify, approve, abstain) | Simulated | Reversibility, evidence freshness | No (runtime rule) | Simulated | The runtime-gateway version of selective escalation (our Candidate C, not pursued). |
| 12 | **Bench2Robust**, arXiv 2608.11977 | Recovery from tool failures | Nine failure modes (three silent) | Retry, switch, *impossible* | Yes (DAPO, a five-phase curriculum) | Task evaluator; assumes no duplicate on retry | Learns retry/switch/impossible with **visible** failures. ActionProof's faults are **hidden behind identical replies**, and duplicates are scored. |
| 13 | **NoisyAgent**, arXiv 2605.27209 | Noisy tools and users | Misleading outputs | Silent misleading outputs | Yes (GRPO, noise curriculum) | An LLM verifier on the final state | Training under silent corruption, scored by an **LLM** verifier. ActionProof's verifier is deterministic. |
| 14 | **Proof-of-Use**, EMNLP 2026 | Decorative tool calls in deep research | Retrieval outputs | Read-only tools | Yes (GRPO) | Citation and perturbation tests | "Information is not use", on read-side tools. ActionProof shows the same for writes (agents read a partial payment, then post). |
| 15 | **Dr. GRPO** (and RLOO) | Unbiased centred advantages | — | — | — | — | Our P3 / AD-C1 removal of the std is Dr. GRPO's advantage. **Not ours.** |

Practice grounding (not a paper). Stripe's API documentation (`docs.stripe.com/error-low-level`,
`docs.stripe.com/api/idempotent_requests`) states the rule that B4 and SEL1's B5 implement:
- retry with the same idempotency key and the same parameters;
- if the original outcome remains unknown, reconcile it before starting another operation;
- if it cannot be established, contact support rather than repeat the operation.

## 2. Competitor overlap

| Competitor | Overlap with ActionProof | Difference |
|---|---|---|
| 02 Escalation-Safe Aya | Learned safe stopping in tool agents (GRPO, five Aya bases) | Its blocks are observable. ActionProof's ambiguity is hidden state behind identical replies, and SEL1 tests transient against permanent evidence outage. |
| 06 Hinglish Expense | Deterministic ledger oracle; deterministic guard | No hidden execution state |
| 08 EVIDENCELOCK | Verify-before-act against a trusted ledger | Visible injected text; GRPO null |
| 13 RewardShield | Verifier reliability; an independent oracle | SQL false accepts, not execution ambiguity |
| 14 VeriLoop | Rewards from DB state plus an audit log | A plan; no results |
| 20 Recovery Agent | Payment-failure recovery | The failure reason is given; no ambiguity |
| 04 Drift Monitor, 09 Safe-to-Phone | Counterfactual twins | Inputs, not hidden state |

## 3. What is already known

1. Acknowledgements can disagree with execution, and agents claim success on duplicates (LIMBO, UndoBench).
2. Verify-before-retry and idempotency keys reduce duplicates (Verified Tool Calls, LIMBO, Stripe practice).
3. The observation-equivalent paired evaluation design ("Did It Happen?").
4. Agents rarely ask or abstain at the right moment (AgentAbstain, HiL-Bench, EpiSEM-Bench).
5. RL against state verifiers can teach recovery from *visible* failures (Bench2Robust, NoisyAgent).
6. Dr. GRPO's centred advantage.
7. Recovery-start curricula (failure-prefix conditioning).

## 4. The smallest genuinely new claims we can support

1. **Trained, registered, multi-seed evidence in the hidden-commit setting.**
   - Recovery demonstrations, then GRPO, raise a 3.35B Cohere agent from 0.513 to 0.819 verified completion on
     450 frozen tasks (P1).
   - The same recipe lifts four other Cohere bases (P5, one SFT seed each).
   - The prior work in this setting is evaluation-only.
2. **A locked counterfactual transaction benchmark (TXN, 440 tasks) with a reference-policy ladder.**
   - B0–B4 plus ablations A1–A6, eight fault classes, three evidence regimes, six OOD groups, and 25 of 25 mutants
     killed.
   - **The design is not new** (item 1). The specific composition and the released episodes are.
3. **Conditional on SEL1 passing its registered tests:** demonstrations can teach *selective* escalation:
   - hand off when a write's outcome is irreducibly unobservable;
   - keep going when the evidence outage is transient;
   - measured against always-escalate and never-escalate references.

   This is the only claim here that would be new against the evaluation-only escalation literature (items 4,
   7–10). Until SEL1 reports, it is a hypothesis.

   **Addendum (2026-11-27 09:57 IST, after task 1812).** SEL1 passed its registered tests (H1, H3, H4 supported;
   H2 passes): escalation on the locked test's unresolvable tasks 0.972 against 0.002 for a matched control,
   selectivity J 0.740 against 0.007, no VTC cost on resolvable tasks. The claim is therefore made, scoped as
   follows. (a) It is taught by supervised demonstrations of a scripted rule; it is not an RL result. (b) Transfer to
   a held-out step is seed-fragile (one seed: 2 of 120). (c) Escalation-Safe Aya (competitor 02) also trains
   stopping; the difference is that ActionProof's outage is hidden behind identical replies and the test separates
   transient from permanent outages. The text above is kept as written.

   **Addendum 2 (2026-11-27 10:40 IST, after the final hostile review; `PREREGISTRATION_SEL1_ADDENDUM1.md`).**
   - Point (c) of addendum 1 is withdrawn. SEL1's trigger is an explicit `service_unavailable` read error, not an
     outage hidden behind identical replies.
   - Every locked-test H1 task could be settled by a same-key resend.
   - Where hand-off is the only safe action, the arm hands off in 164 of 432 episodes, seed-fragile.
   - The claim is therefore narrowed to "supervised demonstrations taught a 3B agent to hand off after repeated
     outage errors, without cost on resolvable work". That is not new against learned stopping or escalation
     controllers (competitor 02; SURE-Act, item 10).
   - Two prior works do train: Bench2Robust (item 12, DAPO) and NoisyAgent (item 13, GRPO). The sentence "The prior work in
     this setting is evaluation-only" is too broad; read it as "most of the prior escalation work".

## 5. Claims that must be abandoned or kept abandoned

- Any new optimizer or credit estimator (FCA dropped; C5 and G dropped; AWRB not better; the P3 no-std change is
  Dr. GRPO's and not supported on D).
- "The counterfactual / paired benchmark design is novel."
- "GRPO teaches verification": ID1 shows that it selects behaviour already sampled.
- "The agent generalises": D fails renamed tools (0.20) and the four-step workflow. P4's gain is at the warm-start
  stage.
- "Safe": H5.
- "The reference verifier independently confirms": same author and specification.
- "Mechanistic circuit discovered."
- "LLM judges can replace the verifier."

## 6. Title and thesis

**Title (submission).** *ActionProof: training small agents to reconcile transactions when "ok" does not mean it
happened.*

**One-sentence thesis (pre-SEL1).** On a locked benchmark where identical "ok" and "timeout" replies hide committed
and uncommitted payments, recovery demonstrations followed by GRPO raise a 3.35B Cohere agent from 0.513 to 0.819
verified completion on 450 frozen tasks. It also duplicates mutations far less often than bounded retry with
idempotency keys (0.024 against 0.191). But it rarely hands off when the evidence cannot settle the outcome. All
of this is in simulation.

The final version is written in `docs/submission/FINAL_STORY_DECISION.md` after SEL1.

## 7. Evidence behind each contribution

| Contribution | Files |
|---|---|
| Recovery training (P1) | `PREREGISTRATION_P1.md`, `results/p1/p1_summary.json`, `scripts/p1_stats.py` |
| Breadth across Cohere bases (P5) | `PREREGISTRATION_P5.md`, `results/p5/p5_summary.json` |
| Warm-start generalisation (P4) | `PREREGISTRATION_P4.md`, `results/p4/p4_summary.json` |
| TXN benchmark and H3 | `PREREGISTRATION_TXN1.md` + addenda, `docs/txn/TXN_SPEC.md`, `results/txn/stats__txn_test.json`, `tests/txn/` |
| H5 escalation gap | `results/txn/stats__txn_test.json` (`H5.per_model`) |
| SEL1 (supported; held-out-step transfer seed-fragile) | `PREREGISTRATION_SEL1.md`, `results/sel1/sel1_summary.json`, `scripts/sel1_stats.py`, `results/sel1_hashes.json` |
