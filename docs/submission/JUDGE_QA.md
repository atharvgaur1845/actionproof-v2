# Judge Q&A

Short answers to the questions a judge is likely to ask. Each number names the file it comes from. Most numbers are
also entries in `claims/claims.yaml`, which `python scripts/s2_numcheck.py` checks against the result files.

**Status words.**
- *Registered*: a test decided by a rule written before the outcome.
- *Descriptive*: a measured number with no test.
- *Post hoc* or *exploratory*: decided after the data were seen.

**Scope.** Every result comes from ProcureSim, an in-memory procurement simulator. Nothing touches a real payment
system.

---

## 1. What is genuinely new here?

Less than the topic suggests. We say what is not ours first (`research_audit/NOVELTY_POSITIONING_FINAL.md`, 15
closest works).

**Not ours.**
- The paired "same reply, different hidden outcome" test design. "Did It Happen?" uses it.
- Read-back and idempotency keys as fixes. LIMBO, Verified Tool Calls and Stripe's own documentation describe them.
  Verified Tool Calls is a scripted wrapper, close to our reference script B4.
- Removing the standard deviation from GRPO's advantage. That is Dr. GRPO's centred advantage.
- Learned stopping or escalation. Competitor 02 (Escalation-Safe Aya) and SURE-Act train it.

**What we can support.**
1. **Registered, multi-seed training evidence where the write's outcome is hidden.**
   - Recovery demonstrations, then GRPO, raise a 3.35B Cohere agent from 0.513 to 0.819 sampled verified completion.
   - The test has 450 frozen tasks, 3 seeds per arm and 4 sampled episodes per task (P1, `results/p1/p1_summary.json`).
   - The same SFT recipe lifts four other Cohere bases (P5, one SFT seed each).
   - Most prior work in this setting only evaluates. Two works do train (Bench2Robust, NoisyAgent), but their
     failures are visible.
2. **A locked benchmark built from these parts, with released episodes.**
   - TXN has 440 tasks: counterfactual quadruples, eight fault classes, three evidence regimes and six OOD groups.
   - Eleven scripted reference policies run under one evaluator (B0–B4 and the ablations A1–A6).
   - The tests kill all 25 scoring mutants.
   - The design is not new; this composition and the data are.
3. **A narrow escalation result (registered SEL1).** Supervised demonstrations taught a 3B agent to hand off after
   repeated outage errors, at no cost on resolvable tasks. That is not new against learned stopping in general.

## 2. Why is this not just a database verifier?

The verifier is the scorer, not the contribution. It reads the simulator's final state after the episode. The agent
never sees it. The questions are whether an agent, seeing only tool replies and its own reads, can decide what
happened and act on that, and whether training can teach this.

What the verifier alone does not give you:
- **It does not act.** After a phantom "ok", something has to read the record, notice the payment is missing and pay.
  After a partial payment, it has to refund and repay, not pay again.
- **"Just check the final state" is a measured baseline, and it is weak.** B1 writes, reads each record once and
  writes any missing or partial step again under a new key. On the locked test it reaches VTC 0.448 (197 of 440) and falsely completes 0.416 (183
  of 440) (`results/txn/stats__txn_test.json`).
- **The agent's report is scored too.** A run counts as a verified completion only if every requested write is in
  the state exactly once and the report matches.

The scripted reference B4 shows what a correct procedure can reach. Our trained agent D trails it (question 4).

RG1 (registered, sealed 440 tasks, D 3 seeds): behind the Recovery Gate, D's false completion falls from 0.112 to 0.000 and its verified completion on resolvable tasks rises from 0.824 to 0.892 (+0.068 [+0.046, +0.091]); unsafe episodes there fall from 482 to 34 of 3420. The hand-off result (1.000 against 0.011) is mostly the gate relabelling honest "failed" reports, and zero false completion is close to guaranteed by design (PREREGISTRATION_RG1_ADDENDUM2.md). B4 still completes more (1.000 on resolvable tasks).

## 3. Why not simply use idempotency keys?

Keys make one thing safe: resending after a lost acknowledgement. They do not tell you whether the first write
happened. ProcureSim (`actionproof/integrity/sim/env.py`, `_write`) shows the gaps.

- **A phantom "ok".** Nothing was committed, so an agent that trusts the reply has nothing to resend. The script
  that trusts the reply (B0) falsely completes 380 of 440 tasks (0.864).
- **A partial commit.** The key stores the "ok" (`env.py:177-178`). A same-key retry returns that cached reply
  (`env.py:159-163`), and the short payment stays. The repair is refund, then repay under a new key.
- **A duplicate made by the system.** A double delivery inserts two payments under one key. Keys cannot remove it.
- **Some writes take no key.** Goods receipts and refunds have none (`env.py:281`, `env.py:314`).

**Measured.** B3 is bounded retry with idempotency keys, and it never reads.
- VTC 0.491 (216 of 440); false completion 0.509 (224 of 440) (`results/txn/stats__txn_test.json`).
- It leaves the duplicate in all 50 of 50 injected-duplicate episodes, against 54 of 450 for D
  (`results/audit/txn_validity.json`).

**An honest qualification (post hoc, TXN1 addendum 3).** In this simulator a same-key resend is always absorbed. A
policy that never reads but sends each keyed write twice reaches 308 of 440 verified completions. It still falsely
completes 132 tasks and leaves 82 with a duplicate. So TXN results must be read with false completion and duplicates
beside VTC.

## 4. Why does the scripted reference B4 beat the trained agent?

**The gap.** On the locked test, B4 reaches VTC 0.864 (380 of 440) with no false completion (0 of 440) and
escalates on all 60 unresolvable tasks. The trained agent D (seed 0) reaches 0.724 (956 of 1,320 episodes) and
falsely completes 0.121 (160 of 1,320) (`results/txn/stats__txn_test.json`, `results/txn/D_s0__txn_test.summary.json`).

**Why the script has an advantage.**
- **It is written for the benchmark's fault taxonomy.** It reads back after every write, retries with the same key
  and repairs a partial payment by refund and repay. It escalates as soon as a record cannot be read
  (`actionproof/txn/policies.py`, `_step`). It is a reference ceiling, not a general agent.
- **It always sees the canonical tool names.** Scripted policies run with `surface=False`
  (`actionproof/txn/policies.py`, `run_policy`). The models face renamed tools and arguments in the `ood_alias` and
  `ood_schema` groups.
- **D was never trained on TXN's new faults** (log conflict, late commit, concurrent write, visibility lag) or on
  escalation.

**Where D loses** (seed 0, from `results/txn/D_s0__txn_test.jsonl`):
- 160 false completions: 142 in the standard regime, 10 in the delayed regime and 8 in the unresolvable regime.
- On the 180 unresolvable episodes, it reports "failed" without handing off in 155.
- 32 episodes end without a report: 19 with no tool call and 13 at the turn limit.

**What closes part of the gap.** SEL1's arms were trained on tasks from the TXN generator (disjoint from txn_test).
SEL reaches 0.830, still below B4. CTL reaches 0.871, but only by settling unresolvable tasks with same-key resends
where B4 hands off. Both still falsely complete (0.020 and 0.022) (`results/impact/impact_txn.json`, descriptive).

RG1 (registered, sealed 440 tasks, D 3 seeds): behind the Recovery Gate, D's false completion falls from 0.112 to 0.000 and its verified completion on resolvable tasks rises from 0.824 to 0.892 (+0.068 [+0.046, +0.091]); unsafe episodes there fall from 482 to 34 of 3420. The hand-off result (1.000 against 0.011) is mostly the gate relabelling honest "failed" reports, and zero false completion is close to guaranteed by design (PREREGISTRATION_RG1_ADDENDUM2.md). B4 still completes more (1.000 on resolvable tasks).

## 5. What does RL add beyond SFT?

Less than the size of the P1 gap suggests.
- **The registered P1 arms are not nested.** C (SFT) and D (SFT then GRPO) differ by more than GRPO, so the gap from
  C at 0.772 to D at 0.819 is not GRPO's contribution.
- **D against its own SFT warm start (R7, exploratory).** +0.140 [+0.120, +0.161] on the 450 P1 test tasks. 69 tasks
  improve by at least 0.5, and none gets worse by 0.5 or more (`results/audit/r7_grpo_increment.json`).
- **The reward's information source matters (A12, registered, Stage 2).** With a database reward, read-back after a
  phantom write becomes the default (1.00 in each of 5 seeds). With an acknowledgement reward it never does (0.00 in
  each of 5). See `docs/submission/WRITEUP.md` §2 and `results/s2/a12_summary.json`.
- **GRPO selects behaviour; it does not create it (ID1, registered).** From a warm start with no verifying
  demonstrations, the agent reads back after 0 of 164 phantom payments. Training never samples a read in 200 steps,
  and read-back stays at 0 of 153 in both seeds (`results/id1/id1_summary.json`).
- **The optimizer change did not help (P3, registered, not supported).** No-std GRPO scores 0.813 against D's
  0.819: −0.007 [−0.013, −0.001] (`results/p3/p3_summary.json`).

## 6. Why does P4 improve the warm start but not D?

D was never trained from the P4 warm start.
- **P4 (registered, supported).** Randomised demonstrations (Dw_aug) beat the plain warm start Dw (3 seeds each) in
  all three tests (`results/p4/p4_summary.json`):
  - renamed tools (150 tasks): 0.137 → 0.741, +0.604 [+0.540, +0.666];
  - an unseen four-step workflow (160 tasks): 0.268 → 0.842, +0.574 [+0.536, +0.610];
  - no cost on the P1 test (450 tasks): +0.048 [+0.026, +0.072].
- **D's lineage.** D is GRPO on top of Dw, the non-randomised warm start. GRPO was never run on Dw_aug, so P4 says
  nothing about D. D scores 0.202 on the renamed tools (`results/p2/p2_summary.json`).
- **An earlier attempt failed.** Randomising tool names during GRPO instead of during SFT (SR1) did not help.
- **The cost.** Dw_aug never escalates on unresolvable evidence (0 of 180 episodes in each seed). It acts unsafely
  there in 53 of 540 episodes (`results/txn/Dwaug_s*__txn_test.summary.json`).

## 7. Why is TXN H3 not supported?

The registered H3 family has four tests: duplicates and unnecessary retries, each against B3 and B1. The family is
supported only if all four pass. Only the two duplicate tests pass.
- **Setting.** D has 3 seeds; the 320 in-distribution tasks form 48 template clusters; clustered bootstrap; Holm over
  4 (`results/txn/stats__txn_test.json`).

| Test | D | Comparator | Difference [95% CI] | Holm p | Result |
|---|---|---|---|---|---|
| Duplicates vs B3 | 0.024 | 0.191 | −0.167 [−0.271, −0.070] | 0.0012 | pass |
| Duplicates vs B1 | 0.024 | 0.338 | −0.314 [−0.454, −0.176] | 0.0008 | pass |
| Unnecessary retries vs B3 | 0.008 | 0.034 | −0.027 [−0.067, +0.003] | 0.123 | fail |
| Unnecessary retries vs B1 | 0.008 | 0.103 | −0.095 [−0.199, −0.015] | 0.064 | fail |

**A later qualification (post hoc, TXN1 addendum 3).** The duplicate metric counts every duplicate in the final
state, including those the environment injects. D's advantage holds in both parts (`results/audit/txn_validity.json`):
- injected classes: D 54 of 450 episodes, B3 50 of 50;
- other classes: D 15 of 2,430, B3 11 of 270.

## 8. What happens when the evidence is unresolvable?

In the unresolvable regime, the faulted step's record and the audit log stay unavailable.

**Without SEL1, the trained agents rarely hand off** (`results/txn/*__txn_test.summary.json`):
- D (3 seeds) escalates in 15 of 540 episodes and acts unsafely in 14 of 540. It mostly reports "failed" (155 of 180
  for seed 0).
- Dw_aug escalates in 0 of 540 and acts unsafely in 53 of 540.
- B4 escalates on all 60 tasks, with no unsafe action.

**With SEL1 (registered, supported)**, two arms continue from the same Dw_aug parents (3 seeds)
(`results/sel1/sel1_summary.json`):
- escalation on the locked test's 60 unresolvable tasks is 0.972, against 0.002 for the matched control;
- selectivity J is 0.740, against 0.007;
- verified completion on resolvable tasks is unchanged: +0.002 [−0.006, +0.010].

**What this does not show (post hoc, `PREREGISTRATION_SEL1_ADDENDUM1.md`):**
- All 60 locked-test tasks fault a keyed step, which a same-key resend settles. So escalation there is one safe
  choice, not the only one.
- On unkeyed receive steps, hand-off is the only safe action. There SEL escalates in 164 of 432 episodes (by seed:
  55, 106 and 3 of 144) and acts unsafely in 215 (`results/audit/txn_validity.json`).
- The trigger is an explicit `service_unavailable` error, not an inference from identical replies.

**Verdict.** Safe abstention is not solved.

RG1 (registered, sealed 440 tasks, D 3 seeds): behind the Recovery Gate, D's false completion falls from 0.112 to 0.000 and its verified completion on resolvable tasks rises from 0.824 to 0.892 (+0.068 [+0.046, +0.091]); unsafe episodes there fall from 482 to 34 of 3420. The hand-off result (1.000 against 0.011) is mostly the gate relabelling honest "failed" reports, and zero false completion is close to guaranteed by design (PREREGISTRATION_RG1_ADDENDUM2.md). B4 still completes more (1.000 on resolvable tasks).

## 9. Can the model inspect hidden state?

No. The checks below enforce this.
- **What the model receives.** The system prompt with the tool schemas, the request, and the tool replies
  (`scripts/p1_eval.py`, `start` and `tool_text`).
- **Where the hidden data live.**
  - Fault plans, realities and regime labels are in `task.meta`, which the episode loop never renders.
  - Data only the evaluator uses (staleness, commit times, colleague writes) go into the trace and attempt records,
    never into a reply (`actionproof/txn/env.py`, module docstring).
- **Leak tests** (`tests/txn/test_leak.py`):
  - no label, fault name or seed appears in the system prompt, tools, user prompt or any reply of the whole dev split
    run by B4;
  - no label appears in the prompts and tool surfaces of every locked test task;
  - the prompt is identical across the four hidden realities of a quadruple.
- **Identical replies** (`tests/txn/test_env.py::test_ack_text_identical_across_hidden_realities`). The reply text
  is byte-identical whether the "ok" hides a commit, a phantom or a concurrent write, and likewise for the five
  timeout realities.
- **The live Space** reveals the database only after the run ends.
- **LLM judges** (J1) receive no hidden-state fields.

## 10. Why not use an LLM judge?

We measured judges against database truth, and they are not good enough to be the outcome measure (J1,
`results/j1/j1_summary.json`; 660 main cases; an abstention counts as wrong).
- **Transcript only.** The best judge (Gemma 4 31B) reaches 0.576 accuracy at 0.58 coverage.
- **With a database snapshot.** The best judge reaches 0.989, and the majority committee 0.897. The committee is no
  better than its best member. The registered test (accuracy of at least 0.90 by the Wilson lower bound, for 3 of 4
  judges) is not supported.
- **A systematic bias.** In 4 snapshot cases all four judges contradict the verifier, and the verifier is right: a
  refunded duplicate nets to one payment.
- **Caution comes from the prompt (exploratory addendum 1).** Remove one rubric sentence, and transcript-only judges
  accept unverifiable failures in up to 120 of 120 episodes.

So the deterministic verifier decides every outcome. The judges are a measured, secondary instrument.

**Not done yet.** The 76 human calibration labels are still pending: there is one annotator, the author.

## 11. What is the measured practical impact, and what is only a projection?

**Measured, on the locked test.** Rates of false completion, duplicates, false refunds and verified completion per
agent, with numerators and denominators (`results/txn/`).

**Projected.** Money (`scripts/impact_txn.py`, `results/impact/impact_txn.json`). The measured rates are priced
with the cited assumptions of `docs/upgrade/ROI.md` §3. The moderate scenario assumes:
- a 2% fault incidence;
- $5,000 per transaction;
- 60% of the money recovered.

| Agent | Per 1,000 workflows that meet an ambiguous fault | Per 1,000 workflows in all |
|---|---|---|
| B0: trust the reply | $1,493,182 | $29,864 |
| B3: bounded retry with idempotency keys | $621,773 | $12,435 |
| D: trained agent (3 seeds) | $145,111 | $2,902 |
| SEL1 SEL arm (3 seeds) | $42,939 | $859 |
| B4: scripted reconciliation checks | $8,182 | $164 |

**How to read the table.**
- The dollars are projections, not observed savings.
- The honest message: against trusting the reply, the trained agent removes about nine-tenths of the projected
  exposure, and the scripted checks remove more than any learned agent.
- The value of ActionProof is in finding which checks matter and in training a second line of defence. It is not in
  replacing the checks.

RG1 (registered, sealed 440 tasks, D 3 seeds): behind the Recovery Gate, D's false completion falls from 0.112 to 0.000 and its verified completion on resolvable tasks rises from 0.824 to 0.892 (+0.068 [+0.046, +0.091]); unsafe episodes there fall from 482 to 34 of 3420. The hand-off result (1.000 against 0.011) is mostly the gate relabelling honest "failed" reports, and zero false completion is close to guaranteed by design (PREREGISTRATION_RG1_ADDENDUM2.md). B4 still completes more (1.000 on resolvable tasks).

## 12. What can a reviewer reproduce without a GPU?

All of the stored results. Not the episodes themselves. Commands and run times are in
`docs/submission/REPRODUCE.md`.
- **Smoke check** (`python scripts/smoke_check.py`). CPU only, no download, no token, about 15–30 s. It checks four
  things:
  - the 152 TXN and SEL1 tests pass, including the 25 scoring mutants;
  - all six frozen splits regenerate byte-identically from their seeds;
  - all 11 scripted baselines reproduce their numbers on txn_test;
  - the claims ledger resolves.
- **Every registered analysis** (`bash scripts/verify_numbers.sh`). CPU, about 2 GB of anonymous downloads, about 6
  minutes in our clean rooms. It recomputes P1, P2, P3, P4, P5, TXN1, SEL1 and the impact projection from the
  released episodes, and requires `git diff --exit-code results/` to be empty.
- **The post-hoc validity analysis** (`python scripts/audit/txn_validity.py`).

Two clean-room runs from a fresh clone, with an empty Hugging Face cache and no token, reproduced every result
byte-identically (`docs/submission/REPRODUCIBILITY_FINAL.md`). Inference and training need a GPU, and were not rerun
in a clean room.

## 13. Which Cohere models and official toolkits were actually used, and where in the code?

**Cohere models** (model ids as they appear in `scripts/` and `slurm/tasks/`).

| Model | Role |
|---|---|
| `CohereLabs/tiny-aya-global` (3.35B) | The base of every trained policy: A, C, D, Dn, Dw, Dw_aug, the SEL1 arms. The live Space runs it as the untrained option (`space_live/core.py:27`) |
| `CohereLabs/tiny-aya-fire`, `-water`, `-earth`, `CohereLabs/aya-expanse-8b` | P5: the recovery-SFT recipe on four other bases (`slurm/tasks/1720_p5_fire.sh` and siblings) |
| `CohereLabs/aya-expanse-32b` | Stage 1 judge; prompted TXN baseline (`slurm/tasks/1709_txn_aya32b.sh`); prompted P1 baseline (`slurm/tasks/1702_b1_aya32b.sh`) |
| `CohereLabs/aya-expanse-8b` | Also a prompted baseline (`slurm/tasks/1701_b1_aya8b.sh`) |

**Toolkits** (checked by import, not by intent).

| Toolkit | Use | Call sites |
|---|---|---|
| AgentTune | `TrlAgenticGrpo` trains every GRPO agent | `actionproof/stage3/train.py:42` (import), `:60` (constructor; D and A); `actionproof/p3/train_nostd.py:10` (Dn); `actionproof/train/grpo.py:75`, `actionproof/train/grpo_proc.py:45` (earlier stages) |
| AgentTune | Prompt block, schema builder and tool-call parser, shared by training and evaluation | `actionproof/rollout/episode.py:23-26` (`_agenttune`); used by `scripts/p1_eval.py:95-98`, and through it by `txn_eval.py` and `sel1_eval.py` |
| CuratorKIT | Exact and MinHash deduplication of the 554 recovery demonstrations before SFT (0 removed) and of the Stage 1 acknowledgement corpus | `actionproof/p1/build.py:180`, `:212-213`; `actionproof/bench/naturalistic_acks.py:45`, `:63` |
| AuditKIT | `TaskCompletion` LLM-judge scores of P1 episodes, secondary and descriptive; `AgentEvalRunner` for recorded-episode checks | `actionproof/eval/judge_audit.py:95-97`; `scripts/p1_auditkit.py:74-75`; `actionproof/eval/auditkit_bridge.py:34`, `:57-59` |

**One caveat.** AgentTune cannot be installed on ZeroGPU. So `_agenttune()` falls back to a verbatim copy of the
format, `actionproof/rollout/lite_format.py` (`episode.py:27-29`), and the live Space vendors that copy. Tests check
that the copy's prompt text is identical and that its parser gives the same output on sample strings.
`scripts/check_lite_parity.py` compares the two parsers on real model outputs on the cluster. That check was not
re-run for the current Space (`space_live/VENDORED.md`).

**Not used for any result.** AlignTune and SafeTune were never imported. CircuitKIT was tried in exploratory A16; it
failed to run in the job environment, and no result is claimed from it.

## 14. How does the paper relate to the hackathon product?

It is a separate, labelled companion. The submission does not rest on it.
- **What it is.** `paper/main_v3.pdf` (source `paper/main_v3.tex`) is a theory note: "When Group Rollouts Share a
  Hidden State: Group Normalisation Cannot See the Stakes of Verification". It analyses GRPO when a group's rollouts
  share one hidden fault state. Its own registered evidence is TH1 and AD-C1.
- **Why it is separate.**
  - The theory concerns costly verification. In ProcureSim, reads are free.
  - The registered test of the paper's fix on the practical agent (P3) found no gain: −0.007 [−0.013, −0.001].
  - The fix itself is Dr. GRPO's centred advantage. It is not a new optimizer.
- **The product's evidence** is the TXN benchmark, the trained agent (P1, P4, P5), SEL1 and the live and replay
  Spaces. None of it depends on the theory.

---

**Standing caveats** (these apply to every answer above):
- The P1 arms C and D are not nested.
- P4's gains are at the warm-start stage; D still fails renamed tools.
- P5 has one SFT seed per base.
- TXN H3 is not supported as a family.
- Safe escalation is not solved.
- B4 is a scripted reference ceiling.
- LLM judges are not verifier-grade, and the human labels are pending.
- The theory paper is not a new optimizer.
- Everything is simulated.
- The reference verifier was written from the same specification as the main verifier, so their agreement is a
  consistency check.
