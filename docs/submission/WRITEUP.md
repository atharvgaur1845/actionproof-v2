# ActionProof: training small agents to reconcile transactions when "ok" does not mean it happened

> **Read this first (2026-11-28).** The submitted paper is `paper/main_v4.pdf`, *The Hidden World in the Group: How
> GRPO Mislearns Costly Verification in Agentic RL*. This write-up is the long-form technical report on the
> **instrument** that paper uses (the TXN benchmark, the trained recovery agent and the live demo); its own claims
> below are unchanged and still hold. Where this file calls the theory a separate or companion claim, that text is
> superseded by `docs/submission/FINAL_STORY_DECISION.md` (revision of 2026-11-28).

**One-line claim.** On a locked benchmark where identical "ok" and "timeout" replies hide committed and uncommitted
payments:
- recovery demonstrations followed by GRPO raise a 3.35B Cohere agent from 0.513 to 0.819 verified completion on 450
  frozen tasks (registered);
- it falsely completes far less often than an agent that trusts the reply (0.121 against 0.864), and it leaves fewer
  duplicate payments than bounded retry with idempotency keys;
- a hand-written reconciliation script still beats it, and our final review found the benchmark weaker than designed
  (TXN1 addendum 3);
- everything is in a procurement simulator.

**Story, in short** (`docs/submission/FINAL_STORY_DECISION.md`).
- **The spine:** the TXN benchmark and the trained recovery agent.
- **Supporting findings:** selective escalation (SEL1, registered in this round) and the reward's information source
  (Stage 2, §2).
- **Main research paper:** the group-normalisation theory, `paper/main_v4.pdf`, which uses this instrument.

| | |
|---|---|
| Track | Track 3: agentic fine-tuning (problem statement 9: Measure → Understand → Fix) |
| Team | Atharv Gaur (BITS Pilani); author contributions in §31 |
| Base model | Tiny Aya Global, 3.35B (CohereLabs, CC-BY-NC-4.0); the recipe is also replicated on Tiny Aya Fire, Water, Earth and Aya Expanse 8B (P5) |
| Method in one line | Recovery demonstrations (LoRA SFT) built by scripted reconciliation, then GRPO (AgentTune `TrlAgenticGrpo`) with a database-state reward, in ProcureSim, an in-memory procurement simulator with silent write faults; evaluated on the locked TXN benchmark |
| Code, models, data | GitHub `atharvgaur1845/actionproof-v2`; models `atharvgaur18/actionproof-agent-d` and siblings; traces `atharvgaur18/actionproof-traces`; live Space `atharvgaur18/actionproof-live`; every artefact with its revision in `release/ARTIFACT_MANIFEST.json` |

---

## 1. Abstract (under 500 words)

**The problem.** Tool-using agents write to systems of record: payments, goods receipts, journal entries. A write's
reply is usually treated as proof that it happened, yet the same "ok" can come back when nothing was stored, and a
"timeout" when the payment committed.

**The benchmark.** In ProcureSim, an in-memory procurement simulator, we built a locked benchmark (TXN, 440 tasks):
- counterfactual quadruples: the same request, tool and arguments; two replies; each hiding a committed and an
  uncommitted outcome;
- eight fault classes, three evidence regimes and six out-of-distribution groups;
- a deterministic verifier that scores the final state.

**Scripted baselines.** Trusting the reply falsely completes 0.864 of tasks. Bounded retry with idempotency keys
reaches 0.491 verified completion; a hand-written reconciliation script reaches 0.864.

**Training (registered P1, 450 frozen tasks, three seeds per arm, Tiny Aya Global 3.35B).** Recovery
demonstrations raise sampled verified completion from 0.513 (a recovery-start GRPO agent) to 0.772. Demonstrations
followed by GRPO reach 0.819. The arms are not nested, so their gap is not GRPO's contribution.

**On TXN.** The trained agent falsely completes 0.121. It leaves far fewer duplicates than bounded retry, both those
it could cause and those the environment injects; both registered duplicate tests pass. Fewer unnecessary retries is
not established, so the registered H3 family is not supported. The scripted reference still beats every learned
agent.

**Generalisation.**
- Randomised demonstrations (P4) raise the warm start on renamed tools from 0.137 to 0.741, and on an unseen workflow
  from 0.268 to 0.842.
- The RL agent itself still scores 0.20 on renamed tools.
- The recipe lifts four other Cohere bases (P5, one SFT seed each) from at most 0.076 to 0.639–0.772.

**Hand-off (registered SEL1, supported).** Demonstrations of a scripted rule taught the agent to hand off after
repeated outage errors, at no cost on resolvable tasks. Where nothing else settles the outcome, it hands off in 164
of 432 episodes, and the effect is seed-fragile.

**What our own review found.** A final hostile review found TXN v1 weaker than designed:
- a read-free same-key double send reaches 308 of 440 verified completions;
- its "unresolvable" tasks can be settled by a same-key resend;
- the duplicate metric mixes injected and caused duplicates;
- repeated refunds are credited (3 of 74,000 episodes).

Each is reported in a dated addendum; no registered verdict changes.

**Reward information (Stage 2, registered).** A database reward makes read-back the default in 5 of 5 seeds; an
acknowledgement reward never does.

**What fails.**
- Hand-off where it is the only safe action.
- Unnecessary retries.
- Renamed tools, for the RL agent.
- No-std GRPO (P3).
- Healthcare transfer.
- LLM judges are not verifier-grade.

**Impact.** A projection from TXN's measured rates, with stated assumptions, puts trusting the reply at about ten
times the trained agent's exposure; the scripted checks are cheaper still. Every confirmatory test was registered
before its outcome, and a checker re-reads every reported number from its file.

## 1a. Final-round results (Nov 25–28; all registered before any output unless labelled)

1. **TXN benchmark (`PREREGISTRATION_TXN1.md`, addenda 1–2).** A locked split of 440 tasks built around counterfactual quadruples: the same request, tool and arguments, two replies ("ok", "timeout"), each hiding a committed and an uncommitted outcome (ok + phantom, ok + committed, timeout + committed, timeout + not committed), eight fault classes, three evidence regimes (resolvable, delayed or conflicting, unresolvable) and six OOD groups. Scripted baselines B0–B4 and ablations A1–A6 share tasks, tools, budgets and the evaluator with every model. TXN keeps ProcureSim's training faults and adds four new ones (log conflict, late commit, concurrent write, visibility lag) and the unresolvable regime.
   - **H1 (descriptive).** The acknowledgement-trusting policy B0 falsely completes on 1.000 of ok + phantom tasks and 0.000 of ok + committed tasks. Its overall verified transaction completion (VTC) is 0.136.
   - **H3 (registered family of 4, Holm, template-clustered bootstrap, n = 320 in-distribution tasks in 48 clusters): not supported.** D (3 seeds) duplicates fewer mutations than scripted bounded retry with idempotency keys (B3) (0.024 vs 0.191, −0.167 [−0.271, −0.070], Holm 0.0012) and final-state-only B1 (0.024 vs 0.338, −0.314 [−0.454, −0.176], Holm 0.0008), with VTC non-inferior (+0.205, +0.289). The unnecessary-retry tests fail: −0.027 [−0.067, +0.003] (Holm 0.123) and −0.095 [−0.199, −0.015] (Holm 0.064).
   - **Descriptive VTC.** B4 (scripted reconciliation) 0.864, D seed 0 0.724, Dw 0.598, B3 0.491, B1 0.448, W 0.445, Aya Expanse 32B 0.161.
   - **H5 (descriptive).** Under the unresolvable regime (the faulted step's read and the audit log unavailable), D escalates in 15 of 540 cases over three seeds (seed 0: 0.050) and acts unsafely in 14 of 540; mostly it reports failure. Aya Expanse 32B escalates in 0.222 but acts unsafely in 0.400. B4 escalates in 1.000. *Post hoc (TXN1 addendum 3):* all 60 of these tasks fault a keyed step that a same-key resend settles, so escalation is one safe disposition there, not the only one.
   - **Validity limits (post hoc, TXN1 addendum 3; `results/audit/txn_validity.json`).**
     - Read-free reference policies, not in the registration: fire-and-forget reaches 211 of 440 verified completions, and a blind same-key double send 308 of 440. The double send makes 132 false completions and 82 duplicates; D seed 0 reaches 0.724 with FCR 0.121.
     - DMR split by origin, on in-distribution tasks. Classes that inject the duplicate themselves: D 54 of 450 episodes, B3 50 of 50. Other classes: D 15 of 2430, B3 11 of 270.
     - Repeated refunds of one payment are credited by the verdict code: 3 such episodes of 74,000 re-scored, none of them D's.
     - Recorded, not fixed (v2): the idempotency model differs from Stripe's (no parameter-mismatch error, no key expiry); "concurrent write" is a scripted injection; ProcureSim's verdict code is not mutation-tested.
2. **P4: randomised demonstrations (`PREREGISTRATION_P4.md`).** Re-rendering half of the recovery demonstrations on synonym tool surfaces and adding five new step sequences (Dw_aug, 3 seeds) against Dw (3 seeds): renamed tools 0.137 → 0.741 (+0.604 [+0.540, +0.666]), unseen four-step workflow 0.268 → 0.842 (+0.574 [+0.536, +0.610]), P1 test +0.048 [+0.026, +0.072]. All three are supported, with every seed delta positive. These are warm starts: GRPO was not run on top.
3. **What GRPO adds after SFT (R7, exploratory).** D − Dw is +0.140 [+0.120, +0.161] on the P1 test; 69 tasks improve by at least 0.5 and none gets worse by 0.5 or more. D's most common false success (266 of 976 failed episodes, 260 of them on the unseen four-step workflow) is reporting a goods receipt it skipped: the procedure-replay failure.
4. **Supplementary large open-weight baselines (B1 addendum 1, not Cohere; descriptive).** Qwen3-235B-A22B and gpt-oss-120b (low reasoning effort), prompted, greedy, no recovery training, through HF Inference Providers: full episodes 0.383 and 0.250; none of the 90 partial-payment tasks completed (0.000 each); on fresh-recovery tasks gpt-oss repairs about half of the partial payments it meets (0.52), Qwen almost none (0.04).
5. **P5: the recipe on four other Cohere bases (`PREREGISTRATION_P5.md` + addendum 1).** Recovery SFT with Dw's data and settings, against each untrained base, on the P1 test (Holm over 4): Tiny Aya Fire 0.018 → 0.686 (+0.668 [+0.637, +0.699]), Water 0.011 → 0.639 (+0.628 [+0.592, +0.664]), Earth 0.012 → 0.704 (+0.693 [+0.662, +0.724]), Aya Expanse 8B 0.076 → 0.772 (+0.697 [+0.667, +0.727]). All four are supported. One SFT seed per model, so each is a task-level test with the model fixed; the unseen four-step workflow stays hard (Water 0.017).
6. **Dw_aug on TXN (descriptive).** Its VTC is above Dw's by +0.126, +0.085 and +0.089 per seed; on the unseen-workflow group its safe terminal resolution is 0.85 against 0.05 (Dw) and 0.083 (D seed 0). The cost: it never escalates on unresolvable evidence (0 of 180 in every seed) and acts unsafely there more often than Dw (11, 24 and 18 of 180, against 8).
7. **SEL1: selective escalation (`PREREGISTRATION_SEL1.md`, registered 2026-11-27 05:39 IST before any training; `scripts/sel1_stats.py` run once as task 1812). Supported.** Two arms continue from the same Dw_aug parents with a new LoRA each (3 seeds; 3 episodes per task). SEL sees scripted B5 demonstrations that escalate under a permanent evidence outage and re-read through a transient one, plus replay; CTL sees the same volume of resolvable demonstrations and replay. Every run finished on its first attempt.
   - **H1, locked txn_test unresolvable (60 tasks):** escalation 0.972 vs 0.002, +0.970 [+0.952, +0.987], Holm p 0.0003, all seeds positive: supported.
   - **H2, held-out step `receive` and held-out workflows (80 tasks, sealed `sel1_test`):** 0.624 vs 0.006, +0.618 [+0.486, +0.754], Holm p 0.0003: passes (reported on its own, not part of the SEL1 decision).
   - **H3, selectivity J (200 tasks):** 0.740 vs 0.007, +0.732 [+0.625, +0.836], Holm p 0.0003: supported. Scripted J: B5 1, B4 0.5, always-escalate 0, never-escalate 0.
   - **H4, VTC on txn_test resolvable (380 tasks):** 0.960 vs 0.958, +0.002 [−0.006, +0.010] against a −0.03 bound: supported.
   - **Descriptive.** SEL escalates in 0 of 360 episodes on a 2-read transient outage, and finishes safely through it in 350 of 360 (CTL 264). On txn_test's resolvable tasks it escalates in 13 of 3420.
   - **Scope (post hoc, SEL1 addendum 1).**
     - *H1's tasks match the training construction, and a same-key resend settles all of them.* SEL's 525 escalations mostly replace an honest "failed" report (CTL: 354 of 540 such episodes), and unsafe actions fall from 12 to 7 of 540. CTL's 173 verified completions there were same-key resends (168), not guesses.
     - *The 2-read outage sits inside the demonstrated 3-read budget.*
     - *The audit-log-only outage cannot fire:* the demonstrated rule reads the audit log only after its primary reads fail, so the 0 of 360 escalations there is not selectivity evidence.
     - *Where hand-off is the only safe action* (unkeyed receive steps, 48 tasks), SEL escalates in 164 of 432 episodes (by seed: 55, 106, 3 of 144) and is unsafe in 215, against CTL's 293. On `u_receive` alone, seed 2 escalates in 2 of 120 and is unsafe in 112, more than its control (83).
     - *The trigger is an explicit `service_unavailable` reply,* not an inference from identical acknowledgements.
   - **Against the scripted reference (descriptive).** Overall txn_test VTC is 0.830 (SEL), below B4's 0.864, and 0.871 (CTL). CTL passes B4 only on the counterfactual quadruples, by settling unresolvable tasks with same-key resends where B4 hands off. B4 completes every task outside the quadruples; the arms do not (renamed tools: 139 and 124 of 180 episodes), and they falsely complete 0.020 and 0.022 (B4 none). The arms were trained on tasks from the TXN generator (disjoint from txn_test), so this is closer to in-distribution than D's 0.724. The method is supervised demonstrations on a LoRA, not a new RL algorithm.
8. **RG1: the Recovery Gate (`PREREGISTRATION_RG1.md`, registered 2026-11-28 02:37 IST; addenda 1–2; `scripts/rg1_stats.py`, task 1929). Supported.** An evidence-only runtime controller between the agent and the tools (`docs/research/RECOVERY_GATE_PROTOCOL.md`): verify before "completed", read before a repeated write, fresh reads, partial repair, hand-off when a written step's record cannot be read. It sees only the request and each call's arguments and reply (tested: no hidden-state access). D (3 seeds) with the gate (RG) vs without (NG) on the fresh sealed `rg1_test` (440 tasks; 3 episodes per task):
   - **H1 false completion:** 0.112 → 0.000 (0.112 [0.084, 0.148]). **H2 hand-off where no record can be read (60 tasks):** 0.011 → 1.000 (0.989 [0.976, 0.998]). **H3 STR:** 0.726 → 0.907 (0.181 [0.129, 0.231]). **H4 VTC on the 380 resolvable tasks:** 0.824 → 0.892 (+0.068 [+0.046, +0.091]; bound −0.02). Holm p < 0.001, every seed positive.
   - **What it means (addendum 2, registered before the results were read).** Zero false completion is close to guaranteed by construction: a consistent read is a perfect oracle in this simulator. The hand-off and most of the STR gain on unresolvable tasks are relabelling: 468 of D's 540 "failed" reports there became gate hand-offs, and an agent that does nothing behind the gate also scores 60 of 60. The gain that belongs to the agent-plus-gate system is on resolvable tasks: +0.068 VTC, unsafe 482 → 34 of 3420. The gate also writes terminal reports when the agent stops (780 of 3960 episodes), hands off on 180 resolvable episodes, and blocks the same-key resends that would settle keyed unresolvable tasks (VTC there 53 → 0).
   - **Against B4.** B4 still completes more: resolvable VTC 1.000 against 0.892 (−0.108); worst groups renamed tools 0.333 and unseen workflows 0.289.
   - **Ablations (D seed 0).** Removing verification brings false completion back (+0.064); removing the hand-off rule removes the hand-offs (−0.967). The other rules each move VTC by about one point. The no-retry row is inflated by a bug (30 duplicate receipts forced by the repair rule; addendum 2).
   - **Other agents.** The gate also removes false completion for the base model, Dw_aug, SEL and Aya Expanse 8B; for the weak ones most outcomes are gate-written.
   - **Transient outages (addendum 1, descriptive, predicted).** On `sel1_test`'s 2-read outage, D behind the gate hands off in 71 of 120 episodes where the record comes back; SEL behind the gate in 8.
9. **RG2: a selective gate (`PREREGISTRATION_RG2.md`, 4b635d2). Supported, as a mechanism check.** Gate v2 re-reads an unavailable record up to three times (SEL1's budget) before a hand-off. On the fresh sealed `rg2_test` (240 tasks), D (3 seeds): selectivity J 0.660 → 0.951 (+0.291), STR on transient outages 0.414 → 0.981 (+0.567), hand-offs where no record can be read unchanged (−0.001; bound −0.03). The budget clears the test's 2-read outage by design; a longer outage would be handed off.
10. **Live inference.** A ZeroGPU Space runs D (and the base model) on curated test tasks, one real model call per step, with the hidden reality revealed only after the run: https://huggingface.co/spaces/atharvgaur18/actionproof-live.

## 2. Before and after (same base model)

| Same model, `proc_conf` (300 confirmatory tasks) | Warm start (SFT) | After GRPO, acknowledgement reward | After GRPO, state reward |
|---|---|---|---|
| Read-back after a phantom write | sampled 31% at the gate (58/188) | 0.00 in each of 5 seeds | **1.00 in each of 5 seeds** |
| Verified completion, state − acknowledgement reward | — | — | **+0.27 to +0.32 per seed** |

Source: `results/s2/a12_summary.json` and `results/eval/a12/conf/*`. The two-metric Holm-adjusted sign-flip p = 0.0625, the minimum possible with five seeds, so "5 of 5" is a consistency statement.

## 3. One-minute summary for judges

1. **The problem is real and silent.** The reply looks the same whether the payment happened or not (§9).
2. **Trusting the reply fails.** On the locked TXN test, the acknowledgement-trusting script falsely completes 0.864
   of tasks: every one where the ok hides a missing payment.
3. **Recovery can be taught.** Recovery demonstrations then GRPO (P1 arm D, registered) lift the frozen test from
   0.513 to 0.819, and partial payments from 0.319 to 0.931. A hint at inference does not help (−0.030).
4. **It beats the naive fixes, not the script.**
   - On TXN the trained agent falsely completes 0.121 and leaves fewer duplicates than bounded retry (registered).
   - Fewer unnecessary retries is not established, so the registered H3 family is not supported.
   - The hand-written reconciliation script beats every learned agent.
5. **Transfer is conditional.** The RL agent fails renamed tools (0.20). Randomising the demonstrations fixes the
   warm start on the tested surface and workflow (P4); GRPO was not run on top.
6. **Hand-off.** Registered SEL1 is supported: the agent learned to hand off after repeated outage errors, with no
   cost on resolvable tasks. But where hand-off is the only safe action, it does so in 164 of 432 episodes,
   seed-fragile (SEL1 addendum 1).
7. **The benchmark has limits we found ourselves** (TXN1 addendum 3).
   - A read-free same-key double send gets close on verified completion, though not on false completion.
   - "Unresolvable" tasks can be settled by a same-key resend.
8. **Also:**
   - the reward's information source decides whether the agent checks (§2);
   - capability retention (R1);
   - a judge committee measured against database truth (J1);
   - a cost model whose honest answer is "write the checks; use ActionProof to find them and to train a second line".

## 4. Real use case

**Accounts payable and procurement agents.** An agent records a goods receipt, pays the supplier and posts the journal entry. When the payment service times out after committing, or commits only part of the amount, the agent receives "ok" either way. Industry surveys put duplicate *or erroneous* payments at about 0.8–2.0% of disbursements (APQC, via vendor summaries; `docs/upgrade/ROI.md` §3). The same pattern applies to refunds, payouts and IT change execution.

## 5. Research questions

- **RQ1.** Does the information source of the reward (database vs acknowledgement) change whether GRPO agents verify their writes?
- **RQ2.** Does verification translate into correct action on the evidence (repair, refund, escalate)?
- **RQ3.** Does training from faulted states improve recovery, and is it the start state or the added fault coverage?
- **RQ4.** Do the gains survive shifts in language, fault count, tool names and workflow?
- **RQ5 (new).** Does the training cost general capability?
- **RQ6 (new).** Can independent LLM judges detect silent write failures without database access?
- **RQ7 (new).** Is any new credit-assignment or sampling method better than the recovery-start curriculum?
- **RQ8 (new).** Can the best agent be made to repair correctly from what it can see: an inference-time reconciliation line, recovery SFT, or recovery SFT followed by the same RL (P1)?

## 6. Preregistered hypotheses (a selection; 54 registrations and addenda are in the repository root, `PREREGISTRATION_*.md`; the Stage 1–2 ones are indexed in `docs/upgrade/INVENTORY.md`)

| Registration | Hypothesis | Verdict |
|---|---|---|
| A12 | R_V selects phantom read-back; R_A does not; same sign in every seed | H2 supported (5/5); composite C1 not supported because H3 failed |
| S3 | Recovery-start curriculum beats the state-reward agent | Supported: +0.21 [0.17, 0.25], 5/5 seeds positive, registered exact sign-flip p = 1/32 (the unpaired seed permutation p = 0.004 is the post-hoc review statistic, claim S3-R) |
| S3 addendum 2 | Start-state effect vs a class-matched control (five runs per arm) | Recovery starts supported (Holm 0.036); partial class not supported (Holm 0.111) |
| S3 addendum 4 | The handed evidence explains the effect | Inconclusive (Holm 0.206, 0.262) |
| S4-EVAL | Gains survive four shifts | Hindi, Hinglish and two-fault supported; renamed API and new workflow not |
| SR1 | Tool-name randomisation fixes renamed tools | Not supported (+0.00); canonical tools −0.248 |
| ID1 | The reward creates verification from a non-reading start | Reading "selection": 0/153 read-back |
| E1/E2 | Sampled decoding contrasts; decision-point probes | Sampled contrasts not supported; reconciliation line +0.48 |
| FCA (development) | Forked counterfactual credit beats RSC by ≥ 0.03 | Failed: −0.10; dropped |
| **R1 (new)** | H1: SFT costs capability. H2: GRPO is non-inferior within 3 points (primary). H3: GRPO lowers abstention | H2 **supported** (non-inferior on both benchmarks, both arms); H1 and H3 not supported |
| **J1 (new)** | J-H1: transcript-only judges are fooled on unverifiable failures. J-H2: a snapshot gives ≥ 0.90 accuracy | J-H1 **not supported** (≤ 1 of 120 accepted: judges abstain); J-H2 **not supported** (0.886–0.989) |
| **C5/G (new, development)** | Verified-progress turn credit; hint-guided sampling; kill criterion +0.03 over RSC | C5 **dropped** (−0.32 and −0.37; it penalises reads by construction, `docs/research/algorithm/C5_RESULT.md`); G **dropped** (−0.11 and −0.033; `docs/research/algorithm/G_RESULT.md`) |
| **AD (algorithm discovery, development)** | Group-std scaling can distort costly-verification learning when rollouts share a hidden fault (mechanism known from Dr. GRPO and Bereket & Leskovec; the interaction is ours); AWRB (aliased-world return baseline) passes development gates | Development only (2 seeds): GRPO stops reading in 2 of 2 seeds at read cost 0.05 (cw_dev VSC 0.707 = warm start); AWRB 0.867 in both seeds; no-std GRPO 0.833. **Confirmatory AD-C1** (`PREREGISTRATION_AD_C1.md`, cw_conf, 5 vs 5 seeds, Holm over four): against GRPO (0.687 in every seed), no-std GRPO +0.120 [+0.088, +0.153], AWRB +0.192 and AWRB without std +0.032 are **supported**. H_AWRB (AWRB without std > no-std GRPO) is **not supported** (−0.088), as predicted before any output. The confirmed fix is the known one (Dr. GRPO's removal of the std), so there is **no new algorithm**. VSC does not charge for reads, so AWRB's larger lead is consistent with over-verification. Second stage (`docs/research/algorithm_discovery_v2/`): the cost-blind "vote rule" is derived and reproduced; AWRB's lead comes from over-verifying, and its unbiased variant does not beat no-std |
| **P2 (new, ledger and guard)** | H1 E (ledger-trained) > D; H2 E > A; H3 a structured ledger at inference (B2) > A; H4 a deterministic action guard (Ag) > A; secondary: the guard improves C, D, E | H2 **supported**: E 0.782, +0.269 [+0.238, +0.301], Holm 0.0004. H1 **not supported** (−0.038; E is below D). H3 **not supported**: B2 0.253 (−0.260). H4 and S1–S3 **not supported**: the guard lowers every arm (−0.122, −0.244, −0.273, −0.142). Scripted oracle on the same tasks: 0.924 (`results/p2/p2_summary.json`) |
| **P3 (no-std on the practical recipe)** | H1: P1 arm D retrained with the group std removed from the advantage (Dn) beats D on the 450 frozen test tasks (one-sided p < 0.05 and all three seed differences positive) | **Not supported**: Dn 0.813 vs D 0.819, −0.007 [−0.013, −0.001], one-sided p 0.988, seed differences −0.009, +0.005, −0.016 (`results/p3/p3_summary.json`). The AD-C1 fix does not carry over to the recovery-SFT recipe, where reads are free |
| **P1 (new, practical agent)** | Against the recovery-start agent (A), on 450 fresh test tasks (unseen payment fractions, an unseen workflow, two-fault episodes): H1 reconciliation line at inference (B) > A; H2 recovery SFT (C) > A; H3 recovery SFT warm start + the same GRPO (D) > A. Holm over three, and all three seed pairs must improve | H1 **not supported**: B 0.483 vs A 0.513, −0.030 [−0.041, −0.019]. H2 **supported**: C 0.772, +0.259 [+0.231, +0.287], Holm 0.0003. H3 **supported**: D 0.819, +0.306 [+0.276, +0.337], Holm 0.0003. All three seed differences are positive for C and D. Development winner: D (`results/p1/p1_summary.json`) |

## 7. Threat model

We do not consider an adversary. The failure model is:
- **benign infrastructure faults**: phantom commit, partial commit, duplicate delivery, lost acknowledgement, pre-commit timeout, never-commit;
- **misleading text inside tool outputs**: Stage 1's naturalistic and injected acknowledgements.

The agent may only call tools. The verifier and its database are outside the agent's reach, and the reward reads them only in training.

## 8. Related work (full matrix in `docs/research/NOVELTY_MATRIX.md`, 84 verified entries in `docs/research/NOVELTY_AUDIT.md`)

- **Tool-use RL:** GRPO (DeepSeekMath), agentic GRPO variants, MT-GRPO, GiGPO, RewardFlow.
- **Training from failure states:** Fission-GRPO, R3L, ReflectPO, Depth Before Breadth. These are the same family as our recovery-start curriculum, so **we claim no novelty for it**.
- **Counterfactual credit:** COMA, BiPACE, GiGPO. FCA is an adaptation, and it failed.
- **Surface binding of tool agents:** "Can Agents Generalize to the Open World?". Our S4 result replicates it.
- **"Did It Happen?":** evaluation-only paired hidden-commit design.
- **Silent-commit benchmarks:** Bench2Robust assumes retries cannot duplicate. We found no training paper with silent write commits plus duplicate risk.

**Our contribution is the setting and the registered measurements, not an optimiser.**

## 9. Tool and task environments, and trajectories

**ProcureSim** (`actionproof/integrity/sim/env.py`). In-memory state (tables of row dicts) with purchase orders, supplier orders, payments, receipts and journal.
- **Write tools:** `place_supplier_order`, `pay_supplier`, `record_goods_receipt`, `post_journal_entry`, `refund_payment`, `escalate_to_human`.
- **Authoritative read tools:** `list_payments`, `list_goods_receipts`, `get_journal_entries`, `find_supplier_order`, `get_purchase_order`.
- **Workflows:** receive→pay→post, order→prepay, pay→post.
- **Faults:** phantom, partial, duplicate delivery, lost acknowledgement, pre-commit timeout, never-commit.
- The acknowledgement for a write is byte-identical across outcomes (`tests/integrity/test_procsim.py`).

**Recorded lifecycles** (`docs/submission/TRACES.md`, generated from result files):
- **Failure** (`s3_conf_full-0014`): the agent pays, reads that 62,892 of 251,570 cents were stored, posts the full journal entry and reports "completed". Verifier: pay `PARTIAL`, goal not met, false success. The acknowledgement reward gave this episode full marks (R_A = 1).
- **Rare success** (`s3_conf_full-0052`, recovery-start seed 3): the agent reads 87,180 of 116,240 cents, refunds, re-pays with a new idempotency key, verifies, posts and reports. Verifier: all steps verified. This is one of 5 repaired tasks out of 60 in this cell.

Traces of all 30 procurement-study models are published in the HF collection.

## 10. Toolkit usage and integration (`docs/upgrade/TOOLKITS.md`)

| Kit | Use |
|---|---|
| **AgentTune** | Trains every GRPO policy (`TrlAgenticGrpo`); its parser and prompt are used in rollouts and evaluation |
| **AuditKIT** | Stage 1 judge audit (`TaskCompletion`, `AgentEvalRunner`; with the ledger shown, Aya 32B still accepted 71% (213/298) of false completions), and secondary `TaskCompletion` scores of the P1 episodes (`scripts/p1_auditkit.py`; descriptive, never an outcome) |
| **CuratorKIT** | Exact and MinHash deduplication (`ExactDeduplicator`, `MinHashDeduplicator`) of the Stage 1 acknowledgement texts and of the 554 P1 recovery demonstrations before SFT (0 removed) |
| **CircuitKIT** | Attempted; failed to run (D-A16-1); no claim |
| **AlignTune, SafeTune** | Not used |

## 11. Models and licences

| Model | Role | Licence |
|---|---|---|
| CohereLabs/tiny-aya-global (3.35B, `Cohere2ForCausalLM`, 8,192 context) | Policy (all training) | CC-BY-NC-4.0 + Cohere Labs acceptable-use policy |
| CohereLabs/aya-expanse-32b | Stage 1 judge; τ-bench simulated user (R1-T) | CC-BY-NC-4.0 + acceptable-use policy |
| google/gemma-4-31B-it | J1 judge | Apache-2.0 (Hugging Face Hub metadata) |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | J1 judge | Apache-2.0 |
| openai/gpt-oss-120b | J1 judge | Apache-2.0 |
| deepseek-ai/DeepSeek-V3.2 | J1 judge | MIT |

**Our adapters inherit CC-BY-NC-4.0** and are research artefacts.

## 12. Datasets, sources, languages, splits, leak checks

**Sources:**
- All ProcureSim tasks are generated by our code (`actionproof/bench/generator_proc.py`), Apache-2.0.
- External benchmarks, used only for evaluation: BFCL v4 (Apache-2.0), IFEval (Apache-2.0), τ-bench (MIT).

**Languages:** English; Hindi and Hinglish shift splits (`s4_lang_hi`, `s4_lang_hinglish`, 150 tasks each), reviewed by a native speaker (the author).

**Splits:** train pools (`proc_train_b`, `s3_trainb_half`, `s3_rec_train_half`), development (`proc_dev`, `s3_rec_dev`) and confirmatory (`proc_conf`, `s3_conf_full`, `s3_conf_rec`, S4 shift splits, fresh `e1_*`/`e2_pp`). `c1_pp` was reserved for Stage 5 but never used, since no candidate survived development; the later AD training pool reuses 75 of its prompts, so it is no longer held-out (`research_audit/DATASET_EXPANSION.md`). Each is hashed at registration (`results/*_hashes.json`), checked in CI (`scripts/ci_freeze_check.py`), and **run once**.

**Leak checks:**
- Task-id disjointness across splits: `actionproof/stage3/tasks.py` `seen_ids`.
- The byte-identical acknowledgement test.
- R1 benchmarks share no prompts or schemas with ProcureSim.
- J1 judges receive no hidden-state fields.

## 13. CuratorKIT filtering

Stage 1: on the naturalistic acknowledgement corpus (`actionproof/bench/naturalistic_acks.py`), exact and MinHash near-duplicate removal before the acknowledgement templates were frozen. P1: the same two filters ran on the 554 oracle recovery demonstrations before SFT (`actionproof/p1/build.py`) and removed none (`data/rft/p1_recovery_rows.stats.json`), because the oracle episodes were already distinct.

## 14. Metrics and why

- **VSC (verified-success completion):** every requested step is verified in the database and the report is honest.
- **False success:** completion was claimed when it is not true.
- **Violation:** a duplicate payment left unrefunded.
- **Phantom read-back:** the agent read back after a phantom write.
- **Tool-call validity.**
- **Calls per episode.**
- **Latency** (`results/latency_single.json`).

We use VSC rather than reward or claimed success because claimed success is exactly what the failure inflates. Definitions: `docs/stage2/METRICS.md`.

## 15. Automatic judge validation (J1)

Full report: `docs/upgrade/JUDGES.md`, generated from `results/j1/j1_summary.json`.

| Judge (660 main cases) | Transcript only: accuracy / coverage | With database snapshot: accuracy | Prompt-paraphrase κ | $ per 1,000 |
|---|---|---|---|---|
| Gemma 4 31B | 0.576 / 0.58 | **0.989** | 0.96 | 0.17 |
| Qwen3 235B-A22B | 0.373 / 0.59 | 0.889 | 0.56 | 0.17 |
| gpt-oss-120b | 0.514 / 0.56 | 0.933 | 0.49 | 0.44 |
| DeepSeek V3.2 | 0.300 / 0.43 | 0.886 | 0.65 | 0.34 |
| Majority committee | 0.385 / 0.42 | 0.897 | — | — |

Accuracy counts an abstention as wrong.
- **On the practical agents' episodes (P2 secondary, descriptive, transcript view).** On the P1/P2 test episodes, Gemma 4 31B agrees with the verifier in 0.951–0.998 of cases and accepts at most 0.017 of failures; these agents read back often, so their transcripts carry the evidence. Qwen3 235B accepts 0.284 of arm E's failures and 0.211 of A's, and the majority never exceeds 0.684 (`results/p2/committee_p2.json`; judge spend J1 + P2 $6.17).
- **Not fooled, but abstaining.** No judge accepted more than 1 of 120 unverifiable failures; judges confirmed only 29–67% of genuinely completed episodes.
- **One systematic bias.** In 4 snapshot cases all four judges contradict the verifier, and the verifier is right: a refunded duplicate nets to one payment.
- **Human calibration labels:** pending (one annotator).
- **Exploratory addendum 1** (registered after J1 was read; `results/j1/j1_addendum1.json`). Remove that one rubric sentence, and transcript-only judges accept unverifiable failures in 120 of 120 (Gemma), 120 of 120 (gpt-oss), 115 of 120 (Qwen) and 9 of 120 (DeepSeek) episodes. **A judge's caution is a property of the prompt, not of the model.**

Design (`PREREGISTRATION_J1.md`):
- four families, temperature 0, seeded option order;
- 76 stratified calibration cases, to be labelled by the author with the same rubric (labels pending), and 660 main cases;
- two views: transcript only, and transcript plus database snapshot;
- a prompt-paraphrase variant.

Outputs: accuracy, coverage, precision, recall, F1, κ, Fleiss κ, abstention, cost and latency. The database remains ground truth.

## 16. Capability retention (R1, R1-T)

Full report: `docs/upgrade/CAPABILITY_RETENTION.md`, generated from `results/r1/r1_summary.json`.

| Same model, change vs its warm start (5 seeds; 95% paired task bootstrap) | IFEval prompt-strict (warm 0.771) | BFCL non-live AST (warm 0.493) | BFCL irrelevance (warm 0.062) |
|---|---|---|---|
| State-reward GRPO | +0.000 [−0.016, +0.017] | −0.002 [−0.009, +0.006] | −0.003 [−0.020, +0.015] |
| Recovery-start GRPO | −0.004 [−0.021, +0.012] | +0.001 [−0.008, +0.009] | +0.005 [−0.009, +0.021] |
| SFT warm start vs base (base 0.760 / 0.471 / 0.079) | +0.011 [−0.020, +0.043] | +0.022 [−0.004, +0.048] | −0.017 [−0.050, +0.017] |

- **No GRPO arm tested in R1 shows a measurable loss on IFEval or BFCL.** The most negative point estimate across these arms and endpoints is −0.012. R1 did not test the P1–P5 agents.
- **Low abstention is the base model's own property** (0.079).
- **τ-bench retail (R1-T, descriptive):** a floor regime for a 3B model (base pass^1 0.058). The SFT warm start lowers it by 0.020 [−0.038, −0.006]; GRPO adds no change (state reward −0.006, recovery-start −0.004, intervals include 0). Most successes involve no write action, so at this scale τ-bench measures hand-off behaviour more than state-changing competence. Two infrastructure deviations were fixed before any episode (D-R1T-1, D-R1T-2).

Benchmark selection:
- BFCL and IFEval are primary.
- τ-bench retail is secondary, with a pre-set context-overflow cancellation rule.
- AgentBench is excluded: heavy infrastructure, floor scores for 3B models, little relevance.

Reasons: `docs/upgrade/PLAN.md` §2. Thirty procurement-study models plus the Stage 1 chain are compared with their own warm start and the base model.

## 17. Training configuration and hyperparameters

- **Warm start:** SFT with LoRA r16, 2 epochs, lr 1e-4, on `proc_demo` (`actionproof/train/rft.py`).
- **GRPO:**
  - 200 steps × 4 prompts × 8 rollouts;
  - lr 2e-5, β 0.02, temperature 1;
  - ≤ 16 turns, at most 256 completion tokens per turn;
  - LoRA r16;
  - batched colocated vLLM rollouts.
- **What the GRPO update actually is** (`research_audit/OPTIMIZER_AUDIT.md`):
  - TRL 1.7.1 takes one optimizer step per batch of 32 rollouts (`num_iterations` 1), as in the original single-update GRPO. The PPO ratio is therefore 1 and the clip never acts: `clip_ratio` is exactly zero at every logged step of every run.
  - The update is the group-baselined, std-scaled policy gradient (REINFORCE with a group-mean baseline), with:
    - DAPO-style token-level loss aggregation;
    - a KL penalty (β 0.02);
    - a sequence-level weight correcting vLLM/trainer mismatch.
  - Tool-result tokens carry no loss.
  - "No-std GRPO" (AD-C1, P3) removes only the std division. That is Dr. GRPO's advantage, without Dr. GRPO's length normalisation.
- **RSC** is the same, on the pools `s3_trainb_half` + `s3_rec_train_half`.
- **Hardware:** one H200 per run, about 1 h 55 min per run (`logs/*_s*.full.log`).
- **Library versions:** TRL 1.7.1, vLLM 0.23.0, AgentTune d4db4c1 (`requirements.lock`, `uv.lock`).

## 18. Seeds

Five independent training seeds per arm (0–4) for A12 R_V and R_A, RSC, the class-matched control, and acknowledgement-only starts; three for SR1; two for FCA, ID1, C5 and G (development). Every seed is reported separately in the result files.

## 19. Ablations

- A12: reward information source only.
- S3 addendum 1/2: fault coverage vs start state.
- Addendum 4: handed evidence.
- SR1: tool-surface randomisation.
- E2: forced evidence and a reconciliation line.
- A14/A15: forced evidence by workflow position.
- A11: gateway factorial.
- Stage 1: PCSTR reward components (no-report, no-ND, no-Funder).

## 20. Confidence intervals and statistical tests

- **Task-level:** 95% paired task bootstrap (10,000 resamples), holding the trained models fixed. This reflects task uncertainty only.
- **Seed-level:** unpaired seed permutation tests and sign-flip tests.
- **Multiplicity:** Holm correction within each registration.
- **Proportions:** Wilson intervals (J1).
- Stated with every result in `docs/final/FINAL_CLAIMS.md`.

## 21. Named findings

1. **A reward that depends on hidden state selects verification the starting model already samples** (A12, ID1). Observation and incentive are not separated: the acknowledgement reward never pays for a read, and no dissociation run was done (`research_audit/RED_TEAM_VERIFICATION.md` R1.5).
2. **Reading is not recovery** (A14/A15, trace §9).
3. **Start state matters, partial-specificity unproven** (S3 addenda).
4. **Procedure replay** (S4, SR1).
5. **Warned judges abstain; snapshot judges are not verifier-grade** (J1, `docs/upgrade/JUDGES.md`).
6. **No measurable IFEval/BFCL cost for the GRPO arms tested** (R1, `docs/upgrade/CAPABILITY_RETENTION.md`; τ-bench drops slightly after SFT; the P1–P5 agents were not tested).
7. **Progress credit penalises evidence** (C5 diagnosis).
8. **Recovery is learned from demonstrations, not from a hint** (P1). Recovery SFT (+0.259) and recovery SFT then GRPO (+0.306) pass the registered test; a reconciliation line at inference fails it (−0.030). C and D are not nested: D starts from its own recovery-SFT warm start, so the gap between them is not what GRPO adds (`research_audit/P1_DECOMPOSITION.md`). The development diagnosis (`docs/research/P1_DEV_DIAGNOSIS.md`) traces the hint's failure to same-key top-ups on short payments.
9. **A guard that blocks the taught repair costs more than the false successes it removes** (P2, `docs/quality/PERFORMANCE_DIAGNOSIS.md` §6). A runtime check has to accept the repair sequence the agent was trained on.
10. **When a group shares one hidden state, GRPO's update cannot see the stakes** (Phase II; `docs/theory/PROOF_AND_ASSUMPTIONS.md`, `docs/final/RESEARCH_DECISION.md`).
   - **Proved.** In agentic RL every rollout in a group meets the same hidden fault, so the group key is really (prompt, hidden state). Under that sharing, GRPO's expected update does not change under any per-state positive rescaling of the return. For a single read-or-not decision with deterministic returns within a state, it becomes a stake-free vote: severity and read cost drop out. Centred estimators (Dr. GRPO, RLOO) give an update proportional to the on-policy action-value gap instead.
   - **Registered held-out simulation (TH1, `PREREGISTRATION_TH1.md`, 731 problems).** GRPO's learned decision was predicted in 721 of 721 cells, and it followed the vote in all 77 cells where the vote and the value disagree. Dr. GRPO followed the value gap in 661 of 661. H6 (GRPO no worse whenever the vote is right) was **not supported** (53 of 75), and H7 was untestable.
   - **Scope.** The rule "verify iff p·s > 1/2" is exact only when repairability belongs to the shared state. In the trained agent, the registered AD-C1 test confirms the predicted direction: no-std GRPO beats GRPO by +0.120 in all five seeds. The mechanism evidence beyond that is development-level: groups share one fault plan in 800 of 800 logged groups, and std-normalised runs follow the early vote. One prediction failed: Dr. GRPO's rise in reading is not predicted by its starting value gap. In the practical recipe the fix does not help: the registered P3 test (D retrained without the std) gives −0.007 [−0.013, −0.001], not supported. The theory predicts a gain only where per-state stakes or read costs differ; in ProcureSim reads are free, so this is consistent with the theory's scope, but that reading was made after the result.
   - **Prior work.** Removing the std is Dr. GRPO's advantage (Liu et al., arXiv 2503.20783; our arm keeps TRL's token-level aggregation, so it is not full Dr. GRPO); the distortion under stochastic outcomes is studied by Bereket & Leskovec (arXiv 2508.11800). Ours is the exact consequence of a shared hidden state for a costed information action: a new consequence of a known mechanism, not a new mechanism. **No new algorithm is claimed.** AWRB and D3 passed development gates only, and D3's mechanism prediction failed.

11. **Acknowledgement-following fails exactly where acknowledgement and state disagree** (TXN H1): B0's false-completion rate is 1.000 on ok + phantom and 0.000 on ok + committed.
12. **The trained agent duplicates mutations less often than scripted bounded retry and final-state checking** (0.024 vs 0.191 and 0.338; 2 of 4 registered tests). Fewer unnecessary retries is not established, and the registered H3 family is not supported.
13. **Procedure replay can be removed in the demonstrations** (P4): synonym surfaces and extra step sequences in the SFT data raise the warm start on the tested renamed-tool surface (0.137 → 0.741) and four-step workflow (0.268 → 0.842), where GRPO-time randomisation (SR1) failed. GRPO was not run on top, and D itself still fails there.

## 22. Negative results (kept and reported)

- A12's composite C1 is not supported (H3 failed).
- The A11 interaction claim was not made (one seed of two).
- The partial-specific start-state effect is not supported.
- Addendum 4 is inconclusive.
- E1 sampled contrasts are not supported.
- S4 (recovery-start agents): renamed API and new workflow, no gain. P4 later raises the warm start; D itself still fails.
- SR1 hurts canonical tools by −0.248.
- Negative transfer to healthcare administration (VSC 0.158 → 0.047).
- FCA failed (−0.10).
- Addendum 3 was withdrawn before training as a degenerate design.
- The after-receipt partial cell remained unsolved by every Stage 3 arm, and rebalancing did not fix it (0.005). P1's recovery arms address it (above); the Stage 3 result stands as recorded.
- P2 H1, H3, H4 and S1–S3: the ledger (at inference or in training) and the action guard do not improve on P1's best agent; the guard lowers every arm.
- P3: removing the group std from D's GRPO does not improve it (−0.007 [−0.013, −0.001], seeds −0.009, +0.005, −0.016).
- P1 H1: the inference-time reconciliation line lowers sampled verified completion (−0.030 [−0.041, −0.019]).
- P1 unseen workflow (order → receive → pay → post): the best arm reaches 0.486 (A 0.174), so recovery does not fully transfer to a new step order. (P4's randomised demonstrations later raise the warm start there; D itself is unchanged.)
- TXN H3 as a family: the unnecessary-retry tests fail (Holm 0.123 and 0.064).
- TXN H5: the trained agents almost never escalate on unresolvable evidence (D 15 of 540 over three seeds, Dw 4 of 180).
- Two large open-weight models, prompted and greedy, complete none of the 90 partial-payment tasks (Qwen3-235B and gpt-oss-120b 0.000).
- Dw_aug never escalates on unresolvable TXN evidence (0 of 180 per seed).
- SEL1, held-out step `receive`: SEL seed 2 escalates in 2 of 120 episodes and is unsafe in 112, more than its matched control (83). SEL1 is supported on its registered tests; this transfer is not robust.
- SEL1's construct (post hoc, SEL1 addendum 1). On the locked test, hand-off was not uniquely correct: a same-key resend settles every task. Where it is uniquely correct, the arm hands off in 164 of 432 episodes. The audit-log-only control cannot fire.
- TXN v1's construct (post hoc, TXN1 addendum 3). A read-free same-key double send reaches 308 of 440 verified completions. "Unresolvable" tasks on keyed steps can be settled. DMR mixes injected and caused duplicates. Repeated refunds are credited.

## 23. Sample inputs and outputs

See `docs/submission/TRACES.md`: two complete recorded episodes with the database before and after, plus the verifier's verdicts. The site's sections 00 and 03 replay further recorded episodes from five models on the same task.

## 24. Compute and ROI

**Compute:**
- HF Jobs spend is logged per job in `docs/stage2/HF_CREDITS.md`.
- The cluster ran the BITS sharanga H200, H100 and A100 lease workers.
- J1 inference costs about $3 projected ($0.19 for calibration).

**ROI** (`docs/upgrade/ROI.md`, `results/roi/roi_model.json`). Measured per-fault outcome rates × cited assumptions. Moderate scenario, per 1,000 workflows:

| Arm | Projected cost |
|---|---|
| SFT agent | $10,261 |
| Recovery-start GRPO | $2,106 |
| Hand-written verify-and-repair checks | $120 |

- Inference is under $0.50 per 1,000 workflows.
- **Projections, not savings.**

**The final-round projection** (`scripts/impact_txn.py`, `results/impact/impact_txn.json`). It is a separate model
from the one above: locked-TXN rates (false completion, duplicates, false refunds) priced with ROI.md's moderate
assumptions. It is stated in two units. The deck's headline is the first; the second multiplies by the assumed 2%
fault incidence.

| Agent | Per 1,000 workflows that meet an ambiguous fault | Per 1,000 workflows in all |
|---|---|---|
| B0: trust the reply | $1,493,182 | $29,864 |
| B3: bounded retry with idempotency keys | $621,773 | $12,435 |
| D: trained agent (3 seeds) | $145,111 | $2,902 |
| SEL1 SEL arm (3 seeds) | $42,939 | $859 |
| B4: scripted reconciliation checks | $8,182 | $164 |

Against trusting the reply, the trained agent removes about nine-tenths of the projected exposure. The scripted
checks remove more than any learned agent. Running checks and agent together was never measured.

## 25. Limitations

- RL was run on one base model (3.35B); the recovery-SFT recipe was replicated on four other Cohere Labs bases (P5), all from one family.
- One simulated domain for the main study.
- Task-level intervals hold the models fixed.
- Seed-level power is low (5 runs).
- The RL agents do not transfer to new workflows or tool surfaces; P4's gain is shown at the warm-start stage only, on one held-out tool surface and one new workflow.
- The policy licence is non-commercial.
- ROI rests on assumed fault rates and costs.
- One human annotator.
- No mechanistic causal claim (A16 is not specific; CircuitKIT failed).
- Prompted and larger models were run (Aya Expanse 8B/32B on ProcureSim and TXN; Qwen3-235B and gpt-oss-120b greedy on the P1 test) without prompt tuning; dynamic sampling (DAPO-style) was not tested.
- Under greedy decoding several seeds give identical scores (for example AD GRPO 0.687 in all five), so sampled metrics carry the seed-level evidence.
- The reference verifier was written from the same specification as the main verifier. Its 100% agreement is a consistency check, not independent ground truth.
- The P1 (A, Dw, D), P3 (Dn) and P4 (Dw_aug) models are public on Hugging Face; the P2 and AD-C1 adapters are on the cluster only. A public clone reproduces the analyses from stored episodes, not the training.
- ID1's null is scoped: no read was sampled in 200 steps from one warm start, so the reward had nothing to reinforce.
- SEL1 teaches escalation by supervised demonstrations from a scripted rule (B5), on Dw_aug warm starts; GRPO was not run on top. Its transfer to a held-out step is seed-fragile (one seed: 2 of 120). Escalation is scored as safe on every task labelled unresolvable, including keyed ones that a same-key resend settles (TXN1 addendum 3). The cost of a human review is priced in the ROI model, not measured.
- TXN v1 has the construct limits of TXN1 addendum 3: read-free policies get close on VTC; keyed "unresolvable" tasks can be settled; DMR counts injected duplicates; repeated refunds are credited; the idempotency and concurrency models are simplified.

## 26. Dual use and ethics

The environment injects benign faults; no real payment systems or personal data are involved. A policy trained to pass verification could in principle learn to game a verifier. Our verifier reads state the policy cannot write except through the audited tools. Judge models received only synthetic procurement transcripts. The models are released for research under their licences.

## 27. Future work

- A second model family and a commercially licensed base.
- Own-history recovery starts.
- A runtime gateway evaluated with trained agents on ProcureSim.
- GRPO from the SEL1 arms with an escalation-aware reward, and held-out-step demonstrations, to make selective hand-off robust across seeds.
- Retention mitigations, if R1 shows a loss.
- The candidate algorithms from the separate discovery programme (`docs/research/algorithm_discovery/`), which go to a confirmatory test only if they pass development gates.

## 28. Artefacts

Every artefact, with its immutable revision and the sha256 of every frozen split and headline result file, is listed
in `release/ARTIFACT_MANIFEST.json` (`scripts/build_manifest.py`; Hub visibility checked without credentials).
- Code: the GitHub repository.
- Models (public, CC-BY-NC-4.0): `atharvgaur18/actionproof-agent-d` (merged D), `-agent-d-adapters`,
  `-warmstart-dw`, `-warmstart-dwaug`, `-agent-a`, `-agent-dn`, `-sel1-adapters`.
- Data: `atharvgaur18/actionproof-traces` (every episode behind a reported number, the frozen splits, the judge
  cases; failed runs kept under `failed_*`).
- Spaces: the live Space (`actionproof-live`, real model calls) and the replay Space (`actionproof`, recorded
  episodes only).
- Not released: the P5 models (results and episodes only) and the P2 and AD-C1 adapters (cluster only).

## 29. Reproduction

| Step | Command |
|---|---|
| Numbers | `python scripts/s2_numcheck.py` (every number in the scanned documents resolves through `claims/claims.yaml`; 0 failures) |
| Inventory | `python scripts/build_inventory.py` |
| Registrations | `python scripts/freeze_<id>.py --check` |
| ROI | `python scripts/roi_model.py` |
| R1 scoring | `.venv-r1/bin/python scripts/r1_score.py && .venv-r1/bin/python scripts/r1_stats.py` |
| J1 | `python evaluation/llm_judge_committee/analyze.py` |
| Training | `slurm/tasks/*.sh` (one file per registered run) |

Command reference: `docs/stage2/COMMANDS.md`.

## 30. Declaration of work done outside the hackathon

- No part of this work predates Nov 1, 2026 (confirmed by the author).
- **Stage 1** (Nov 1–15): SupportSim, PCSTR rewards, A4–A11.
- **Stage 2** (Nov 20–28): ProcureSim, A12–SR1, E1/E2, and this round's R1, J1, ROI and C5/G.
- **Final round** (Nov 23–28): P1–P5, the TXN benchmark (TXN1), SEL1, the live and replay Spaces, the deck, and the
  final hostile review with its post-hoc addenda (TXN1 addendum 3, SEL1 addendum 1).
- The toolkits are the hackathon's own repositories, vendored at the stated revisions.

## 31. Author contributions

Atharv Gaur:
- research direction, hypotheses and the review of every registration;
- Hindi and Hinglish review;
- human calibration labels;
- the video.

Claude (Anthropic) implemented the code, ran the experiments and drafted the documents under the author's direction (§33).

## 32. Acknowledgements

Lexsi Labs for the hackathon, its toolkits and HF credits; BITS Pilani HPC (sharanga) for GPU time; Cohere Labs for the Aya models.

Citations for the toolkits and models used (arXiv IDs and titles checked on 2026-11-28):

- AgentTune: R. E. Z. M. Lyngkhoi, A. Gupta, V. Vats et al. *AgentTune: A Unified Training Lifecycle for Tool-Using LLM Agents.* arXiv:2610.13335.
- AuditKIT: S. Arora, U. Avaiya, Z. Lyngkhoi et al. *AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems.* arXiv:2610.22511.
- CuratorKIT: S. Bhattacharjee, K. Sharma, V. K. Sankarapu et al. *CuratorKIT: Data Curation and Synthetic Data Generation for LLM Post-Training.* arXiv:2606.21631.
- CircuitKIT (attempted only; no result): P. Seth, H. Gosalia, A. Kasliwal et al. *CircuitKIT: Circuit Discovery, Evaluation, and Application Toolkit for Mechanistic Interpretability.* arXiv:2607.19317.
- Tiny Aya (the base model of every trained agent): A. R. Salamanca, D. Abagyan, D. D'souza et al. *Tiny Aya: Bridging Scale and Multilingual Depth.* arXiv:2603.11510.
- Aya Expanse (8B and 32B, untrained comparison agents): J. Dang, S. Singh, D. D'souza et al. *Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier.* arXiv:2412.04261.

## 33. LLM usage statement

Claude (Anthropic) was used throughout, under human review of every registration and claim:
- writing the code;
- running and monitoring experiments;
- drafting registrations, documents and this write-up;
- literature search.

Approximate shares, author's estimate: about 95% of the code and about 90% of the document text were generated by
Claude. The research direction, approval of every registration before its run, and the final review of every claim
were the author's. The three hostile reviews (`research_audit/FINAL_REDTEAM_REVIEW.md`) were run as independent
Claude sessions that could not see one another's conclusions.

LLM judges (§15) are evaluation instruments whose outputs never replace the database verifier.

## 34. Changelog

- **Stage 1** (Nov 14): SupportSim study.
- **Stage 2** (Nov 20–23): ProcureSim A12–SR1, E1/E2.
- **Nov 23, this round:**
  - audit and inventory (C3 ledger sync);
  - R1, R1-T and J1 registered and run;
  - ROI model;
  - website sections 06–08;
  - Stage 5 development (C5, G);
  - a separate algorithm-discovery programme.
- **Nov 24:** P1 frozen test reported (H2, H3 supported; H1 not supported). P2 reported (H2 supported; H1, H3, H4 and S1–S3 not supported), with a post-hoc guard diagnosis. Phase II theory merged (TH1 registered held-out simulation). The HF Space site was redeployed with sections 09 (P1, P2) and 10 (theory) on Nov 24 (Space commit acf4f79).
- **Nov 25:** P3 reported (not supported); replay Space redeployed.
- **Nov 26–27 (final round):** TXN1 registered and run (H3 not supported as a family; addenda D-TXN1-1 and D-TXN1-2), P4 (all three tests supported), P5 (all four bases supported; addendum D-P5-1), R7 (exploratory), B1 addendum 1 (large open-weight baselines), the live ZeroGPU Space, Dw_aug models, replay Space redeployed with TXN, P4 and P5 sections. An independent adversarial review of every public surface was run before the freeze; its findings were fixed in the documents (wording, labels and omissions; no result changed).
- **Nov 27, after the final hostile review:** three independent reviews (all FAIL as found). Two post-hoc addenda (TXN1 addendum 3, SEL1 addendum 1) and an exploratory validity analysis (task 1840). Every surface reworded to their scope; a false statement that both SEL1 arms pass B4's VTC was corrected (only CTL does). No registered verdict changed.
- **Nov 27 (superiority sprint, branch `stage2-superiority-sprint`):** competitor forensics; SEL1 registered (05:39 IST) before any training and run (H1, H3, H4 supported; H2 passes; held-out-step transfer seed-fragile); README rebuilt as the landing page; deck; live Space judge mode, labelled recorded fallback and run export; impact projection from TXN rates.
- **Deviations:** `docs/stage2/DECISIONS.md` (this round: D-R1-1, D-P1-1) and the addenda above.

## 35. Appendix pointers

| Topic | Location |
|---|---|
| Registrations | `PREREGISTRATION_*.md` |
| Claims | `claims/claims.yaml`, `docs/final/FINAL_CLAIMS.md` |
| Audit | `docs/upgrade/AUDIT.md` |
| Inventory | `docs/upgrade/INVENTORY.md` |
| ROI | `docs/upgrade/ROI.md` |
| Toolkits | `docs/upgrade/TOOLKITS.md` |
| Traces | `docs/submission/TRACES.md` |
| Submitted paper | `paper/main_v4.tex`, `paper/appendix_v4.tex` (superseded drafts' PDFs in `paper/archive/`) |
