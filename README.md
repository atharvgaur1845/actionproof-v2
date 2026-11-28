# ActionProof

> **Main research paper (2026-11-28): [The Hidden World in the Group: How GRPO Mislearns Costly Verification in
> Agentic RL](paper/main_v4.pdf)** ([source](paper/main_v4.tex), [evidence ledger](paper/EVIDENCE_LEDGER_V4.md),
> [decisions](paper/RESEARCH_DECISIONS_V4.md)).
> When the rollouts of a GRPO group share an unobserved environment state (one fault plan, one database snapshot),
> group-local reward standardisation removes the state-specific stakes: a costly verification decision then follows a
> vote of state-wise signs, not the value of checking. We prove this under stated assumptions, test it in a registered
> held-out testbed of 731 problems (the theory predicts GRPO's decision in 721/721 cells; one registered hypothesis
> fails, one is untestable), and in a registered five-seed study of a 3.35B Cohere tool agent (std-free +0.120 verified
> completion; every GRPO seed ends at the warm start's no-read behaviour, so that arm is degenerate). A registered
> cost test (one run per arm) met all five predictions: GRPO never reads at read costs 0.05 and 0.01 but reads at cost 0,
> while the std-free agent reads more as reads get cheaper. No new optimiser is claimed. The rest of this README describes the **instrument** that paper uses: the ActionProof benchmark, the
> trained recovery agent and the live demo ([practical paper](paper/actionproof_main.pdf)).

**The instrument: training small agents to reconcile transactions when “ok” does not mean it happened.** A payment tool can answer
“ok” to a payment that never happened, or time out on one that did. ActionProof is a benchmark of such ambiguous
writes (TXN, 440 locked tasks) and a trained 3.35B Cohere agent that must tell committed, missing, partial and
ambiguous outcomes apart, then complete, repair, retry or hand off. Outcomes are scored by a deterministic verifier
from the simulator’s final state, and the agent’s report is checked against that state. Everything runs in a
procurement simulator (ProcureSim); nothing touches a real payment system.

> **Judge’s 60-second path**
> 1. **See the failure and the trained agent, live.** Open the
>    [live Space](https://huggingface.co/spaces/atharvgaur18/actionproof-live) and press Run: judge mode is
>    preselected (Task 0004, Reality C, a phantom payment). The left column is a script that trusts the reply: it sees
>    ok and reports done. The right column is a live call to the trained agent: it reads the payment back, finds none,
>    and pays before it reports. The database is revealed after the run. It takes about 70 s once the GPU is warm. This
>    is one greedy run: an illustration, not a rate. If the GPU is busy, the Space says so and replays a recorded run
>    under a `RECORDED MODEL RUN` banner. The [replay Space](https://huggingface.co/spaces/atharvgaur18/actionproof)
>    holds the recorded study.
>    Then pick **Gate demo** (no GPU, about 10 s): the same reply-trusting script runs again, behind the Recovery
>    Gate, an evidence-only controller between the agent and the tools. The gate reads the payment record before it
>    accepts "completed", finds nothing, and hands the order to a human instead of reporting a false success.
> 2. **Check the claim.** The headline tables below each carry their evidence status, and their numbers resolve
>    through `claims/claims.yaml`. `python scripts/smoke_check.py` takes about 15 seconds on a CPU, with no download
>    and no token. `bash scripts/verify_numbers.sh` (about 2 GB) recomputes every registered model analysis.
> 3. **Read what does not work** ([below](#what-does-not-work-registered-or-descriptive-full-list-in-docssubmissionwriteupmd-22)),
>    including the limits of the benchmark itself.
>
> **In one paragraph.**
> - **Training (registered P1, 450 frozen tasks, 3 seeds per arm).** Recovery demonstrations then GRPO raise
>   verified completion from 0.513 to 0.819.
> - **Acknowledgement trust fails.** On the locked TXN test, trusting the reply falsely completes 0.864 of tasks. The
>   trained agent D falsely completes 0.121.
> - **Duplicates.** D leaves far fewer duplicate payments than bounded retry with idempotency keys, both those it
>   could cause (15 of 2430 episodes against 11 of 270) and those the environment injects (54 of 450 against 50 of
>   50). But the registered H3 family is not supported (retries).
> - **The scripted reference still wins.** A hand-written reconciliation script beats every learned agent, with or without the gate.
> - **The benchmark has limits.** Our final hostile review found TXN v1 weaker than designed. A read-free same-key
>   double send reaches 308 of 440 verified completions (D seed 0: 0.724), and its “unresolvable” tasks can be
>   settled by a same-key resend (`PREREGISTRATION_TXN1_ADDENDUM3.md`).
> - **Checks around the agent (registered RG1, sealed split of 440 tasks, D 3 seeds, 3 episodes per task).** The
>   Recovery Gate, an evidence-only controller between the agent and the tools, removes false completion (0.112 to
>   0.000) and raises verified completion on resolvable tasks from 0.824 to 0.892 (+0.068 [+0.046, +0.091]); unsafe
>   episodes there fall from 482 to 34 of 3420. Read with addendum 2: zero false completion is close to guaranteed
>   by design (a consistent read here is a perfect oracle), and the hand-off result (1.000 against 0.011) is mostly the
>   gate relabelling D's honest "failed" reports (468 of 540). Even with the gate, the script B4 completes more
>   (1.000 on resolvable tasks; renamed tools 0.333, unseen workflows 0.289).
> - **A selective gate (registered RG2).** The first gate hands off too early when a record returns after two reads
>   (71 of 120 episodes, predicted in addendum 1). Re-reading up to three times raises selectivity from 0.660 to
>   0.951 with no loss of correct hand-offs; the budget clears this outage length by design.
> - **Hand-off (registered SEL1).** Demonstrations taught the agent to hand off after repeated outage errors, at no
>   cost on resolvable tasks. Where nothing else settles the outcome, it hands off in only 164 of 432 episodes, and
>   the effect is seed-fragile (`PREREGISTRATION_SEL1_ADDENDUM1.md`).
>
> This is transaction-aware recovery under ambiguous execution, in a simulator: not a new RL optimizer, and not a
> claim of safety.

**Terms.**
- **Scores.**
  - VSC (P1) and VTC (TXN): verified completion. Every requested write is in the final state exactly once, and the
    report matches.
  - FCR: false completion (success reported; the state disagrees).
  - DMR: a duplicate mutation left in the final state.
  - URR: an unnecessary retry.
  - STR: safe terminal resolution (verified completion, or a hand-off on an unresolvable task).
  - CER: correct escalation (a hand-off where the task is labelled unresolvable).
- **Scripted baselines.** B0 trusts the reply; B1 checks the final state only; B3 is bounded retry with idempotency
  keys; B4 is the transaction-aware reconciliation script.
- **Agents.** A is recovery-start GRPO. D is recovery SFT then GRPO, and is the recommended agent. Dw is D’s SFT warm
  start. Dw_aug is the P4 randomised warm start. SEL and CTL are SEL1’s two arms.

**Status (2026-11-28).**
- The submission is branch `submission-candidate` (the repository's default branch). The submitted paper is
  `paper/main_v4.pdf`.
- Video: https://github.com/atharvgaur1845/actionproof-v2/releases/tag/video-v4 · slides: `docs/submission/ActionProof_deck.pdf`.
- Internal working notes (`research_audit/`) are not part of the submission branch. Paths to them in older documents
  refer to the git history (commit `055767e`). The clean-room logs moved to `docs/reproducibility/cleanroom/`.
- How to read a status:
  - *registered, supported / not supported*: a test decided by a preregistered rule;
  - *descriptive*: no test;
  - *exploratory* or *post hoc*: decided after the data were seen.
- Every deviation or later finding is a dated addendum; no registration is edited after its outcome.

## Headline results

### P1: recovery demonstrations (registered, `PREREGISTRATION_P1.md`)

| Arm | Sampled VSC | vs A | Holm |
|---|---|---|---|
| A: recovery-start GRPO (reference) | 0.513 | — | — |
| B: A + reconciliation line at inference | 0.483 | −0.030 [−0.041, −0.019] | not supported |
| C: recovery SFT | 0.772 | +0.259 [+0.231, +0.287] | 0.0003, supported |
| D: recovery SFT, then the same GRPO | 0.819 | +0.306 [+0.276, +0.337] | 0.0003, supported |

What GRPO adds after SFT (R7, exploratory, D against its own warm start): +0.140 [+0.120, +0.161] over 450 tasks; 69 tasks improve by at least 0.5 and none gets worse by 0.5 or more. D's most common false success (266 of 976 failed episodes, 260 of them on the unseen four-step workflow) is reporting a goods receipt it skipped: the procedure-replay failure.

### TXN: transaction integrity under ambiguous execution (registered, `PREREGISTRATION_TXN1.md` + addenda 1–2)

The locked `txn_test` split holds 440 tasks: counterfactual quadruples (the same request, tool and arguments; two replies, "ok" and "timeout", each hiding a committed and an uncommitted outcome), eight fault classes, three evidence regimes and six OOD groups. Scripted baselines B0–B4 and ablations share tasks, tools, budgets and the evaluator with the trained and prompted models.

| Agent (descriptive; all episodes) | VTC | FCR |
|---|---|---|
| B0 acknowledgement-trusting | 0.136 | 0.864 |
| B1 final-state only | 0.448 | 0.416 |
| B3 bounded retry + idempotency | 0.491 | 0.509 |
| B4 transaction-aware reconciliation (scripted reference) | 0.864 | 0.000 |
| W (plain SFT warm start) | 0.445 | 0.475 |
| Dw (recovery-SFT warm start) | 0.598 | 0.258 |
| D seed 0 | 0.724 | 0.121 |
| Dn seed 0 (no-std GRPO) | 0.717 | 0.127 |
| Aya Expanse 32B (prompted) | 0.161 | 0.610 |
| Qwen3-235B-A22B (prompted, greedy; supplementary) | 0.261 | 0.641 |
| Dw_aug seeds 0 / 1 / 2 (P4 warm starts) | 0.723 / 0.683 / 0.686 | 0.180 / 0.245 / 0.236 |

VTC: verified transaction completion. FCR: false-completion rate. Scripted rows: one deterministic episode per task. Local models: one greedy plus two sampled episodes per task (T = 1). Qwen: one greedy episode per task.

- **H1 (descriptive).** B0's FCR is 1.000 when the "ok" hides a phantom write and 0.000 when the "ok" is true: acknowledgement-following fails exactly where the acknowledgement and the state disagree.
- **H3 (registered), D (3 seeds) vs B3 and B1, in-distribution tasks (n = 320, 48 template clusters), Holm over 4. Not supported as a family.**

  | Test | D | Comparator | Difference [95% CI] | Holm p | Result |
  |---|---|---|---|---|---|
  | Duplicate mutations vs B3 | 0.024 | 0.191 | −0.167 [−0.271, −0.070] | 0.0012 | supported |
  | Duplicate mutations vs B1 | 0.024 | 0.338 | −0.314 [−0.454, −0.176] | 0.0008 | supported |
  | Unnecessary retries vs B3 | 0.008 | 0.034 | −0.027 [−0.067, +0.003] | 0.123 | not supported |
  | Unnecessary retries vs B1 | 0.008 | 0.103 | −0.095 [−0.199, −0.015] | 0.064 | not supported |

  D's VTC is non-inferior in every test (+0.205 vs B3, +0.289 vs B1).

  *Post hoc (TXN1 addendum 3).* DMR counts every duplicate in the final state, whoever caused it. Two fault classes
  inject the duplicate themselves: `dup_delivery` (a second delivery under one key) and `concurrent` (a colleague's
  write). B3 never reads, so it cannot repair them. D's advantage holds in both parts:
  - injected classes: D leaves 54 of 450 episodes, B3 50 of 50;
  - other classes: D 15 of 2430, B3 11 of 270.
- **H5 (descriptive), unresolvable evidence.** Here the faulted step's read and the audit log stay unavailable.
  - D escalates in 15 of 540 episodes over its three seeds (seed 0: 9 of 180, 0.050) and acts unsafely in 14 of 540;
    mostly it reports failure.
  - Dw escalates in 4 of 180 (0.022).
  - Aya Expanse 32B escalates more (0.222) but acts unsafely in 0.400.
  - B4 escalates on all of them (1.000, unsafe 0.000).

  Escalation never appeared in D's training demonstrations. *Post hoc (TXN1 addendum 3):* all 60 of these tasks
  fault a keyed step (order or pay), which a same-key resend settles. So escalation is one safe disposition here, not
  the only one.

### TXN v1: limits found by the final hostile review (post hoc, `PREREGISTRATION_TXN1_ADDENDUM3.md`)

An independent hostile review of the benchmark found four construct problems. `scripts/audit/txn_validity.py`
(task 1840, `results/audit/txn_validity.json`) sizes them. No registered test or verdict changes.

| Read-free reference policy (one episode per task) | VTC | FCR | DMR |
|---|---|---|---|
| Fire-and-forget: write once, ignore every reply | 211 of 440 | 229 | 82 |
| Blind same-key double send: each keyed write twice, never read | 308 of 440 | 132 | 82 |
| D seed 0 (for comparison) | 0.724 | 0.121 | see H3 |

1. **VTC alone overstates the headroom.** The simulator absorbs a same-key resend, so a policy that never reads gets
   close to D on VTC. Its false completions and duplicates are not close. TXN results are read with FCR and DMR.
2. **"Unresolvable" means unreadable, not unresolvable.** All 60 unresolvable txn_test tasks fault a keyed step. A
   same-key resend makes the final state correct in every hidden reality, and its returned id even reveals that
   reality. Only unkeyed (receive) steps are unresolvable by any action, and txn_test has none.
3. **DMR includes duplicates the environment injects** (see H3 above).
4. **Repeated refunds of one payment are credited.** The verdict checks that one unrefunded payment remains, not net
   money. Re-scoring every recorded episode with a net-money postcondition finds 3 credited-but-net-wrong episodes of
   74,000, none of them D's.

Also recorded:
- The idempotency model is not Stripe's: a mismatched-parameter retry returns the original result, and keys never
  expire.
- "Concurrent write" is a scripted injection, not an interleaving.
- The 25 scoring mutants do not cover ProcureSim's verdict code.

These are v2 work.

### P4: randomised demonstrations reduce procedure replay in the warm start, on the tested surface and workflow (registered, `PREREGISTRATION_P4.md`)

Dw_aug re-renders half of the recovery demonstrations on synonym tool surfaces (none shared with the held-out surface) and adds oracle demonstrations on five new step sequences (none is the test workflow), so it has about 21% more rows than Dw. One held-out tool surface and one new workflow were tested. Three seeds per arm; sampled VSC; paired task bootstrap.

| Test | Dw | Dw_aug | Difference [95% CI] | Seed differences | Result |
|---|---|---|---|---|---|
| H1 renamed tools (`s4_api`, 150 tasks) | 0.137 | 0.741 | +0.604 [+0.540, +0.666] | +0.612, +0.613, +0.587 | supported |
| H2 unseen four-step workflow (160 tasks) | 0.268 | 0.842 | +0.574 [+0.536, +0.610] | +0.545, +0.623, +0.553 | supported |
| H3 no canonical cost (P1 test, 450 tasks; bound > −0.03) | 0.666 | 0.714 | +0.048 [+0.026, +0.072] | +0.027, +0.027, +0.091 | supported |

These are warm starts. GRPO was not run on top of Dw_aug, the earlier attempt to randomise tool names during GRPO (SR1) failed, and the RL agent D itself still scores 0.20 on the renamed tools.

On the TXN benchmark (descriptive), which keeps ProcureSim's training faults and adds four new ones (log conflict, late commit, concurrent write, visibility lag) plus an unresolvable regime, Dw_aug's VTC is above Dw's by +0.126, +0.085 and +0.089 across the three seeds. On TXN's unseen-workflow group, safe terminal resolution is 0.85 for Dw_aug seed 0 against 0.05 for Dw and 0.083 for D seed 0; on renamed tools, 0.60 against 0.18 and 0.30. The cost: Dw_aug never escalates on unresolvable evidence (0 of 180 in every seed), acts unsafely there more often than Dw (11, 24 and 18 of 180, against 8), and retries unnecessarily slightly more (seed 0: +0.019).

### P5: the recipe on four other Cohere bases (registered, `PREREGISTRATION_P5.md` + addendum 1)

The same recovery-SFT data and settings (Dw) applied to other bases; sampled VSC on the 450-task P1 test; paired task bootstrap, Holm over the four models. One SFT seed per model, so this is a task-level test with each model fixed.

| Base model | Untrained | Recovery SFT | Difference [95% CI] | Result |
|---|---|---|---|---|
| Tiny Aya Fire | 0.018 | 0.686 | +0.668 [+0.637, +0.699] | supported |
| Tiny Aya Water | 0.011 | 0.639 | +0.628 [+0.592, +0.664] | supported |
| Tiny Aya Earth | 0.012 | 0.704 | +0.693 [+0.662, +0.724] | supported |
| Aya Expanse 8B | 0.076 | 0.772 | +0.697 [+0.667, +0.727] | supported |

For reference, Tiny Aya Global: 0.01 untrained, 0.679 with the same recipe (seed 0). The unseen four-step workflow stays hard: Tiny Aya Water reaches only 0.017 there.

### Supplementary baselines (B1 addendum 1; not Cohere models)

Greedy VSC on the frozen P1 test splits through HF Inference Providers (the P1 arms above are sampled).

| Split | Qwen3-235B-A22B | gpt-oss-120b |
|---|---|---|
| full episodes | 0.383 | 0.250 |
| partial payments | 0.000 | 0.000 |
| fresh recovery | 0.475 | 0.550 |
| four-step workflow | 0.300 | 0.267 |
| two-fault episodes | 0.067 | 0.067 |

Two large open-weight models, prompted, greedy, with no recovery training and no prompt tuning (gpt-oss at low reasoning effort), complete none of the 90 partial-payment tasks. On fresh-recovery tasks gpt-oss repairs about half of the partial payments it meets (0.52) and Qwen almost none (0.04). Descriptive. These are greedy single runs, so they are not compared with the sampled P1 arms.

### SEL1: selective escalation (registered 2026-11-27, `PREREGISTRATION_SEL1.md`; supported; post-hoc scope in `PREREGISTRATION_SEL1_ADDENDUM1.md`)

**Why.** Under the unresolvable regime, the trained agents rarely hand off: D in 15 of 540 episodes, Dw_aug in 0 of
540.

**Design.**
- Two arms continue from the same Dw_aug parents (3 seeds), with a new LoRA each.
- SEL sees B5 demonstrations that escalate under a permanent evidence outage and re-read through a transient one.
- CTL sees the same number of resolvable demonstrations.
- They are scored on the locked txn_test and on a sealed 240-task split. That split holds an unresolvable outage on a
  held-out step and held-out workflows, a 2-read transient outage (training had 1-read ones; the demonstrated rule
  escalates after 3), and an audit-log-only outage.

**Why selectivity is measured.** On unresolvable tasks an always-escalate policy already scores perfectly. The
registered selectivity statistic J is escalation on unresolvable tasks minus escalation on degraded-but-resolvable
ones. Its scripted reference values are exact by construction: selective B5 1, escalate-on-first-error B4 0.5,
always-escalate 0, never-escalate 0.

**Result (registered, supported; `results/sel1/sel1_summary.json`, task 1812).** Every run finished on its first
attempt. Rates are means over 3 seeds × 3 episodes per task; intervals are 95% cluster bootstrap; p is Holm-adjusted
over H1–H3.

| Test | Tasks | SEL | CTL | SEL − CTL [95% CI] | Rule | Result |
|---|---|---|---|---|---|---|
| H1 escalation, locked txn_test unresolvable | 60 | 0.972 | 0.002 | +0.970 [+0.952, +0.987], p 0.0003 | p < 0.05 and 3/3 seeds > 0 | supported |
| H2 escalation, held-out step and workflows | 80 | 0.624 | 0.006 | +0.618 [+0.486, +0.754], p 0.0003 | same (reported on its own) | passes |
| H3 selectivity J | 200 | 0.740 | 0.007 | +0.732 [+0.625, +0.836], p 0.0003 | same | supported |
| H4 VTC, txn_test resolvable | 380 | 0.960 | 0.958 | +0.002 [−0.006, +0.010] | lower bound > −0.03 | supported |

What the supported tests show, and what they do not (scope from `PREREGISTRATION_SEL1_ADDENDUM1.md`, post hoc):
- **Shown.** SEL learned to call `escalate_to_human` after repeated `service_unavailable` reads of the faulted step,
  including on held-out workflows.
  - It did not escalate on a 2-read outage (0 of 360 episodes); it re-reads and finishes safely in 350 of 360,
    against 264 of 360 for CTL.
  - On txn_test's resolvable tasks it escalates in 13 of 3420, with no VTC cost.
- **What H1 measures.** All 60 locked-test tasks fault a keyed step that a same-key resend settles.
  - SEL's escalations mostly replace an honest "failed" report (CTL: 354 of 540 such episodes).
  - Unsafe actions fall from 12 to 7 of 540.
  - CTL's 173 verified completions there were same-key resends (168), not luck.
- **Not shown.**
  - The audit-log-only outage cannot fire: the demonstrated rule reads the audit log only after its primary reads
    fail, so the 0 of 360 there is not evidence of selectivity.
  - The 2-read outage sits inside the demonstrated 3-read budget.
- **Where hand-off is the only safe action** (unkeyed receive steps, 48 sel1_test tasks): SEL escalates in 164 of 432
  episodes (by seed: 55, 106, 3 of 144) and is unsafe in 215, against 293 for CTL. Partial and seed-fragile.
- **Supervised, from a scripted rule.** J is 0.740 against the scripted B5's 1. This is not an RL result, and not a
  claim that the agent is safe.

## What does not work (registered or descriptive; full list in `docs/submission/WRITEUP.md` §22)

- **Hand-off where it is the only safe action.** SEL1's arm hands off in 164 of 432 such episodes, seed-fragile (one
  seed: 3 of 144). Without SEL1, D escalates in 15 of 540 unresolvable cases, and Dw_aug never does (unsafe in 53 of
  540).
- **Retries.** Fewer unnecessary retries than scripted baselines is not established: the H3 family is not supported.
- **The RL agent’s transfer.** D fails renamed tools (0.20) and the unseen four-step workflow. Randomising tool names
  during GRPO (SR1) did not help. Healthcare transfer is negative.
- **Optimizer changes.** No-std GRPO (P3) is not supported: 0.813 against 0.819.
- **The scripted reference wins.** B4 reaches 0.864 TXN verified completion with no false completion and no duplicate.
  It completes every task outside the counterfactual quadruples; no learned agent does (renamed tools: SEL 139 and
  CTL 124 of 180 episodes). B4, like every scripted policy, always calls the canonical tool names, also in the
  renamed-tool groups, so those groups do not test it.
  - D reaches 0.724.
  - SEL1's SEL arm reaches 0.830, below B4.
  - Its CTL arm (0.871) passes B4's overall VTC only on the quadruples, by settling unresolvable tasks with same-key
    resends where B4 hands off. Both still falsely complete (0.020 and 0.022). Descriptive.
- **The benchmark's own limits.** Read-free policies get close on VTC; "unresolvable" tasks can be settled; DMR mixes
  injected and caused duplicates; repeated refunds are credited. See [TXN v1: limits](#txn-v1-limits-found-by-the-final-hostile-review-post-hoc-preregistration_txn1_addendum3md).
- **LLM judges are not verifier-grade.** The best judge with a database snapshot reaches 0.989 accuracy; the
  committee 0.897.

## Shortcuts: what blocks them, and what does not

| Shortcut an agent could take | What blocks it (and what does not) | Where |
|---|---|---|
| Report success without doing the work | Success is the database state against the requested postconditions, read by a deterministic verifier; the report is checked against it | `actionproof/integrity/sim/truth.py`, `actionproof/txn/truth.py` |
| Read the answer from the task | Fault plans and realities live in `task.meta`; the model sees the request, the tool schemas and the tool replies only | `scripts/p1_eval.py` (`start`, `tool_text`) |
| Escalate everything to look safe | Partly blocked. Always-escalate scores no completions, and SEL1's J is 0 for it. But it gets perfect correct-escalation on unresolvable tasks, where a same-key resend would also be safe | `actionproof/txn/truth.py` (STR), `scripts/sel1_stats.py`, TXN1 addendum 3 |
| Pay twice “to be sure” | Partly blocked. A second payment under a new key is a duplicate and counts. A same-key double send is absorbed by the simulator and not penalised: a read-free version reaches 308 of 440 verified completions | TXN metrics (`docs/txn/TXN_SPEC.md` §7), `results/audit/txn_validity.json` |
| Tune on the test | Every test split is hashed and registered before training; each tag and split is evaluated once (the scripts refuse to overwrite) | `scripts/freeze_*.py`, `results/*_hashes.json` |
| A buggy scorer | Partly blocked. All 25 mutants of the TXN scoring code (`actionproof/txn/`) are killed by its tests; there are goldens and a reference verifier (same author and spec, so only a consistency check). ProcureSim's verdict code is not mutation-tested, and it credits repeated refunds (3 such episodes of 74,000) | `tests/txn/`, TXN1 addendum 3 |

## Public release

| What | Link |
|---|---|
| Live Space (the recommended agent, running) | https://huggingface.co/spaces/atharvgaur18/actionproof-live |
| Replay Space (recorded episodes, no model calls) | https://huggingface.co/spaces/atharvgaur18/actionproof |
| Recommended agent D (merged, seed 0) | https://huggingface.co/atharvgaur18/actionproof-agent-d |
| D adapters (seeds 0 to 2) and the Dw adapter | https://huggingface.co/atharvgaur18/actionproof-agent-d-adapters |
| Recovery-SFT warm start Dw | https://huggingface.co/atharvgaur18/actionproof-warmstart-dw |
| Reference agent A (recovery-start GRPO) | https://huggingface.co/atharvgaur18/actionproof-agent-a |
| Dn, the no-std GRPO variant (P3, not supported), for reproducibility | https://huggingface.co/atharvgaur18/actionproof-agent-dn |
| Dw_aug warm starts (P4, seeds 0 to 2) | https://huggingface.co/atharvgaur18/actionproof-warmstart-dwaug |
| SEL1 adapters (SEL and CTL, seeds 0 to 2; registered SEL1) | https://huggingface.co/atharvgaur18/actionproof-sel1-adapters |
| Episode traces, frozen splits, judge cases | https://huggingface.co/datasets/atharvgaur18/actionproof-traces |
| **Submitted paper** (main research paper) | `paper/main_v4.pdf` (source `paper/main_v4.tex`, `paper/appendix_v4.tex`) |
| Video (3:18: real live run on the L4 Space, then the research result; captions burned in) | https://github.com/atharvgaur1845/actionproof-v2/releases/tag/video-v4 (source and `.srt` in `docs/submission/video/`) |
| Slides (PDF, 12 slides) | `docs/submission/ActionProof_deck.pdf` |
| Supporting technical report on the instrument (benchmark, agent, demo) | `paper/actionproof_main.pdf` |
| Superseded drafts (not the submission) | `paper/archive/` |
| Write-up | `docs/submission/WRITEUP.md` |
| Final-round deck (PDF/PPTX export) | source `docs/submission/deck/`, mirrored by `docs/submission/SLIDES.md` |
| Code | https://github.com/atharvgaur1845/actionproof-v2 |
| Every artefact with its revision and sha256 | `release/ARTIFACT_MANIFEST.json` |

All released models derive from `CohereLabs/tiny-aya-global` (D as a merged model, the rest as LoRA adapters) and carry CC-BY-NC-4.0: research use only, no production payments. P5's SFT models on four other Cohere bases are not released. Every result is on a simulator (ProcureSim). Arms C and D are not nested, P3 is not supported (Dn slightly below D), and D itself does not generalise to renamed tools or to the four-step workflow (P4's gain is at the warm-start stage); each model card says so next to its numbers.

### Toolkits actually used

Checked against the code by import, not by intent.

| Toolkit | What it does in the pipeline | Where |
|---|---|---|
| CuratorKIT | Exact and MinHash deduplication run on the 554 oracle recovery demonstrations before SFT (0 removed: the oracle episodes were already distinct) and on the Stage 1 acknowledgement corpus | `actionproof/p1/build.py`, `actionproof/bench/naturalistic_acks.py` |
| AgentTune | The tool-call format: system-prompt block, schema builder and `extract_tool_calls_from_text` parser, shared by training and evaluation; `TrlAgenticGrpo` trains every GRPO agent (A, D, Dn) | `actionproof.rollout.episode._agenttune`, `scripts/p1_eval.py`, `actionproof/stage3/train.py`, `actionproof/p3/train_nostd.py` |
| Deterministic verifier (ours) | Scores every episode against the database; the authority for all results | `actionproof/integrity/sim/truth.py` |
| AuditKIT | `TaskCompletion` LLM-judge scores of the P1 episodes, secondary and descriptive; it never decides an outcome | `scripts/p1_auditkit.py`, `actionproof/eval/judge_audit.py` |

Not used for any result: AlignTune and SafeTune (never imported). CircuitKIT was attempted in the exploratory A16 analysis, failed in the job environment, and no result is claimed from it (`docs/upgrade/TOOLKITS.md`).

### Citations

The toolkits and models above (arXiv IDs and titles checked on 2026-11-28):

- AgentTune: R. E. Z. M. Lyngkhoi, A. Gupta, V. Vats et al. *AgentTune: A Unified Training Lifecycle for Tool-Using LLM Agents.* arXiv:2610.13335.
- AuditKIT: S. Arora, U. Avaiya, Z. Lyngkhoi et al. *AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems.* arXiv:2610.22511.
- CuratorKIT: S. Bhattacharjee, K. Sharma, V. K. Sankarapu et al. *CuratorKIT: Data Curation and Synthetic Data Generation for LLM Post-Training.* arXiv:2606.21631.
- CircuitKIT (attempted only; no result): P. Seth, H. Gosalia, A. Kasliwal et al. *CircuitKIT: Circuit Discovery, Evaluation, and Application Toolkit for Mechanistic Interpretability.* arXiv:2607.19317.
- Tiny Aya (the base model of every trained agent): A. R. Salamanca, D. Abagyan, D. D'souza et al. *Tiny Aya: Bridging Scale and Multilingual Depth.* arXiv:2603.11510.
- Aya Expanse (8B and 32B, untrained comparison agents): J. Dang, S. Singh, D. D'souza et al. *Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier.* arXiv:2412.04261.

## Reproduce

| What | Command | Needs | Time |
|---|---|---|---|
| Smoke test (**not** result reproduction): benchmark regenerates, scripted baselines, tests, the number ledger | `pip install -e ".[dev]" && python scripts/smoke_check.py` | CPU, no download, no token | about 15 s |
| Every registered model analysis from the released episodes | `pip install numpy scipy pyyaml huggingface_hub && bash scripts/verify_numbers.sh` | CPU, about 2 GB download, `git` | about 10 min |
| Run the trained agent (inference) on the TXN dev split (use `--split txn_test` with a new tag for the locked test) | `python scripts/txn_eval.py --model atharvgaur18/actionproof-agent-d --tag mine_D --split txn_dev --samples 2` | 1 GPU with 24 GB or more, vLLM, the GPU stack below | about 10 min |
| Train the registered arm D (recovery SFT, then GRPO) | `slurm/tasks/1153_p1_dwarm.sh`, then `slurm/tasks/1154_p1_d_s0.sh` | 1 H100/H200, the pinned stack | about 4–5 h |

**What each route proves.**
- `verify_numbers.sh` verifies the stored results: it recomputes every registered statistic from the released
  episodes, and `git diff --exit-code results/` must be empty.
- Inference and training reproduce the episodes themselves. On different hardware, sampled decoding can differ.

**The GPU stack.**
- Python 3.12, `bash slurm/setup_env.sh`.
- `requirements.lock`: torch 2.11.0+cu130, transformers 5.22.1, trl 1.7.1, vLLM 0.23.0, peft 0.21.2.
- AgentTune `d4db4c12d523`, AuditKIT `71316da34578`, CuratorKIT `dd9e835301fc`.

**Common failures.**
- *"results/ has uncommitted changes"*: use a fresh clone.
- *Gated base model*: accept the Tiny Aya terms on the Hub and set `HF_TOKEN`.
- *A float difference below 1e-9 in the TXN statistics JSON* (set order): run with `ALLOW_FLOAT_NOISE=1`.

## Repository map

| Path | What |
|---|---|
| `actionproof/txn/` | TXN benchmark: proc3 env, evaluator, invariants, scripted policies B0–B4 and A1–A6, task generator |
| `actionproof/sel1/` | SEL1: transient-outage env, B5 policy, splits, demonstration builder |
| `actionproof/integrity/sim/` | ProcureSim and its deterministic verifier |
| `actionproof/p1/`, `actionproof/p4/`, `actionproof/stage3/` | Recovery demonstrations, randomised demonstrations, GRPO training |
| `scripts/` | Evaluation (`p1_eval.py`, `txn_eval.py`, `sel1_eval.py`), registered analyses (`*_stats.py`), checks (`smoke_check.py`, `verify_numbers.sh`, `s2_numcheck.py`), freezes |
| `space_live/` | The live Space (real model inference) |
| `demo/` | Sources of the replay site (`demo/static/index.html` is the built page) |
| `release/` | Model and dataset cards, artefact manifest |
| `research_audit/` | This round’s audits (competitors, branch, stale surfaces, red team) |
| `docs/submission/` | Write-up, deck source, video script, captions, run of show, story decision |
| `docs/HISTORY.md` | Earlier studies (Stages 1–4) and the original reproduction notes |
| `PREREGISTRATION_*.md` | 54 registrations and addenda (including the two post-hoc addenda of 2026-11-27) |

## License

- Code: Apache-2.0 (`LICENSE`).
- Models and adapters derived from Cohere Aya: CC-BY-NC-4.0, with Cohere attribution; research use only.
