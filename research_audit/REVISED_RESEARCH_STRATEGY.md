# Revised research strategy (addendum §5)

2026-11-25, branch `submission-candidate`, which contains every branch (`BRANCH_AUDIT.md` §5). Deadline Nov 28, 23:59 IST.

**Grounded in:**
- `STUDY_HISTORY.md`: every registered study and its outcome;
- `P1_DECOMPOSITION.md`: the P1/P2 arms re-analysed from episodes;
- `ADAPTIVE_RL_AUDIT.md`: every algorithmic variant;
- `OPTIMIZER_AUDIT.md`: what the update computes;
- `RED_TEAM_VERIFICATION.md`: the hostile reviews, recomputed.

The ranking pasted with the brief is treated as a hypothesis:
1. recovery SFT + GRPO first;
2. then the recovery curriculum;
3. then judges;
4. then reward observability;
5. no FCA;
6. capability retention as validation.

Each answer says where the evidence agrees or disagrees with that ranking. Nothing below is a new claim of support. Exploratory contrasts are labelled.

---

## 1. Should recovery SFT + GRPO be the primary contribution?

**As the primary practical result: yes. As an RL contribution: no.** The evidence credits the demonstrations, not GRPO.

- **Registered and robust.** Against the recovery-start agent A (0.513):
  - recovery SFT (C) 0.772 (+0.259);
  - recovery-SFT warm start + the same GRPO (D) 0.819 (+0.306);
  - Holm 0.0003, all 9 cross-seed pairs positive;
  - cell-clustered 95% CI [0.169, 0.343] and [0.203, 0.396] (`results/p1/p1_summary.json` `test`; `results/audit/redteam_recompute.json` `p1`).
- **The gain is demonstration-driven.**
  - Partial-payment VSC rises from 0.343 (A) to 0.867 (C) and 0.935 (D) (`results/audit/p1_decomposition.json` `by_fault_class.partial`).
  - No arm without the demonstrations exceeds 0.513: W 0.335, M 0.479, B 0.483.
- **GRPO's own increment on top of recovery SFT is unmeasured.**
  - C and D are not nested: C = RSC then SFT; D = SFT data in the warm start, then RSC GRPO.
  - D's warm start `p1_dwarm` was never evaluated.
  - D's GRPO stage had zero reward variance in 0.78–0.82 of its groups (`OPTIMIZER_AUDIT.md` §3).
- The exploratory D − C +0.047 [0.030, 0.065] bundles three things: SFT order, data mix and SFT settings.

**Wording for the submission:**
> "Verified recovery demonstrations teach the repair that reward alone did not; with them in the warm start and the same GRPO recipe afterwards, sampled VSC on 450 fresh tasks rises from 0.513 to 0.819."

Do not write "GRPO improves recovery SFT".

**Versus the pasted ranking:** agree on rank 1 as the practical headline. Disagree that its novelty lies in "SFT + GRPO"; it lies in the demonstrations.

WHAT WE LEARNED → demonstrations are the active ingredient (+0.26 to +0.31; partial repair 0.34 → 0.87–0.94). An inference-time hint fails (−0.030).
WHAT THIS RULES OUT → crediting GRPO for P1's gain; "the agent only lacked the arithmetic".
WHAT REMAINS UNCERTAIN → GRPO's increment over `p1_dwarm`; the data × RL interaction.
NEXT EXPERIMENT → evaluate `p1_dwarm` on dev and the frozen test. Evaluation only, about 1 GPU-hour; **feasible before Nov 28**, after P3 frees a lane. Together with P3 (Dn vs D), it gives GRPO's increment and its normaliser sensitivity in one table (`P1_DECOMPOSITION.md` §5).

## 2. What exactly do the S3 controls establish?

| Comparison | Registered outcome | Robustness check (this audit) | What it establishes |
|---|---|---|---|
| RSC vs A12 R_V (recipe) | +0.206 rec, full non-inferior, 5/5 (S3 H1/H2) | cell CI [0.060, 0.365] full, [0.098, 0.322] rec | the **recipe** (start states + added fault classes) beats the A12 recipe. Per class: duplicate delivery +0.116, partial +0.093 (claim S3-R) |
| RSC vs class-matched control, recovery starts | H3′ +0.107, Holm 0.036, **supported** (5 v 5) | cell CI [0.040, 0.183] | **start states help on recovery-start evaluation**, with fault coverage held fixed |
| RSC vs control, full episodes, partial class | H4′p +0.384 over tasks, Holm 0.111, **not supported** | learners 5/5 vs 2/5, Fisher 0.083; all-class full-episode mean +0.102, cell CI [0.014, 0.200] | the mean difference is real across cells, but **seed-level evidence that start states teach partial repair on full episodes is not there**: a bistable seed lottery |
| RSC vs control, duplicate class | H4′d, Holm 0.337, not supported; control better on duplicate stress with 3 seeds, RSC better with 5 | — | duplicate gains are **fault coverage**, not the curriculum |
| ACK starts (no scripted read) | D1, D2 not supported ("inconclusive") | RSC − ACK not cell-robust (CI includes 0) | the scripted read in recovery starts is **not shown** to matter |
| Rebalanced pool (2B) | receive-then-pay repair 0.005 | — | the workflow binding of repair is **not pool frequency** |
| Fresh P1 tasks, A − M (exploratory) | — | +0.034 [0.021, 0.046], cell [0.012, 0.054], 7/9 seed pairs; pp +0.079, rec +0.085, wf4 −0.033 | a **small** start-state advantage carries to fresh tasks, in the same places |

WHAT WE LEARNED → the curriculum has a registered, cell-robust effect on recovery-start evaluation and a small exploratory effect on fresh tasks. The large S3 headline (+0.21) is mostly the recipe's extra fault coverage.
WHAT THIS RULES OUT → "recovery-start training teaches partial repair on full episodes" (seed level); "the duplicate gain is the curriculum".
WHAT REMAINS UNCERTAIN → the mechanism: start position vs oracle-written history vs handed evidence. SELF-history starts (red team §4.3) were never run.
NEXT EXPERIMENT → SELF starts, 5 seeds plus new rollout code. **Not feasible** before Nov 28 with analysis time.

## 3. Which recovery-start claims survive S4 and the later controls?

**Survive:**
- recovery-start VSC over the control (S3-A2 H3′);
- the recipe gain under Hindi and Hinglish requests and two-fault episodes (S4-EVAL G2, G4). The surviving gain is the same coverage split (red team §3.6);
- a small fresh-task advantage over the control (P1 A − M, exploratory);
- capability retention on IFEval and BFCL (R1 H2).

**Fail:**
- renamed tools (G1 −0.016; SR1 −0.248 on canonical tools);
- the unseen four-step workflow (G3 +0.016, 2/5; P1 A − M on wf4 −0.033);
- the sibling domain (care_admin VSC 0.158 → 0.047; S3-R);
- the partial class at seed level (H4′p);
- sampled-decoding contrasts (E1 P1–P3);
- the ACK decomposition (S3-A4).

WHAT WE LEARNED → recovery-start gains are bound to the trained tool surface and workflow structure. They carry across request language and fault combinations.
WHAT THIS RULES OUT → any general "robust recovery" claim; transfer to other APIs, workflows or domains.
WHAT REMAINS UNCERTAIN → whether demonstrations (P1 C/D) transfer better. P1 wf4 is 0.397/0.486 against A's 0.174, the only unseen-workflow improvement in the repository, but still half the oracle (1.000).
NEXT EXPERIMENT → none before Nov 28. Afterwards: S4 shifts for C/D (inference only, about 2 GPU-hours).

## 4. Does the adaptive research deliver actual gains?

**No new adaptive method delivered a gain.**

Every new estimator failed its pre-set rule or its registered test:
- FCA −0.097 (dev kill);
- C5 −0.32;
- G −0.11;
- D2 failed G1;
- D3 passed its gates, but its mechanism prediction failed;
- AWRB over no-std −0.088 (registered, predicted);
- LAB1's CDVO 0/5 seeds.

**The only confirmed training-side gain from this line is removing the group std:** AD-C1 H_nostd +0.120 [0.088, 0.153], Holm 0.016, 5/5.
- That is Dr. GRPO's advantage, a known method.
- Its value here is that the theory predicted it before any output (TH1 H1–H5 supported; AD-C1 T1–T7 met).

**The adaptive recovery sampler in the pasted ranking was never implemented** (`ADAPTIVE_RL_AUDIT.md` §1, last row). The only related evidence is one development pilot, fixed 25% vs 50%.

WHAT WE LEARNED → the contribution of the algorithm line is explanatory: why std-normalised GRPO cannot weigh stakes under a shared hidden state. Its practical fix is known.
WHAT THIS RULES OUT → claiming a new algorithm; launching a new adaptive sampler without development evidence. There is no lane or analysis time for it either.
WHAT REMAINS UNCERTAIN → whether no-std helps the practical agent (P3, running).
NEXT EXPERIMENT → P3 (already registered and running).

## 5. Which algorithm experiments failed?

Each was dropped by criteria fixed in advance, or not supported at registration:
- **dropped in development:** FCA, C5, G, AD-D2 (G1);
- **mechanism failed:** AD-D3 (gates passed);
- **registered test not supported:**
  - AWRB over no-std (AD-C1 H_AWRB);
  - CDVO (LAB1 L1 H1/H2), and LAB1 H3;
  - SR1 tool-surface randomisation (H1, H2);
  - S3-A4 ACK starts (D1, D2);
- **not carried forward:** the safety-first lexicographic and Lagrangian pilots;
- **analysed, not trained:** potential shaping, which is invariant under a group baseline;
- **withdrawn before training:** addendum 3 STATE.
- **P2:** the ledger at inference (H3 −0.260), the action guard (H4 −0.122, rule R1 blocks the repair), ledger training vs D (H1 −0.038).
- **P1:** the inference hint (H1 −0.030).

Sources: `ADAPTIVE_RL_AUDIT.md` §1; `STUDY_HISTORY.md` §2.

WHAT WE LEARNED → changes to the credit estimator, the reward shape or inference-time aids did not beat the baseline. Every change that helped altered the training data (recovery starts, demonstrations) or removed a known distortion (std).
WHAT THIS RULES OUT → reviving FCA, guard rule R1 or the ledger at inference as contributions.
WHAT REMAINS UNCERTAIN → FCA with forced branches added rather than substituted, weighted by class probability (red team §3.7). Low priority.
NEXT EXPERIMENT → none.

## 6. Is judge reliability strong enough to be a separate contribution?

**Not yet as a standalone contribution. It is a strong supporting result.** The final call depends on the judge study (`JUDGE_RELIABILITY.md`, main session, tasks 1600/1601).

**What exists:**
- **J1, registered; both hypotheses not supported.** With a rubric stating that "ok is not proof", frontier transcript judges abstain rather than accept (≤ 1/120). With the database snapshot they reach 0.886–0.989 accuracy.
- **J1 addendum 1, exploratory.** Removing that one sentence makes three of four judges accept unverifiable claims (120/120, 120/120, 115/120, against 9/120 for DeepSeek).
- **J2, descriptive.** Aya Expanse 32B, the Stage 1 judge family, accepts 0.700 of false completions from the transcript and **0.573 with the database snapshot**, against ≤ 0.013 for the four J1 judges (`results/j2/j2_summary.json` `main_vs_verifier`).
- **Stage 1.** AuditKIT with the ledger accepts 0.715.

**What it lacks:**
- **human labels.** Every judge number is model-vs-verifier;
- **a registered test of prompt vs evidence.** Pending in judge_next;
- **J2 has no ledger claim.**

WHAT WE LEARNED → evidence access is necessary but not sufficient. Judge family and one rubric sentence decide false acceptance.
WHAT THIS RULES OUT → using any LLM judge as ground truth or as a reward in this setting; calling a snapshot-reading judge a "verifier".
WHAT REMAINS UNCERTAIN → human agreement; whether structured evidence fixes small judges (judge_next).
NEXT EXPERIMENT → judge_next (running); add a J2 ledger claim (CPU, minutes).

## 7. Can any result be independently reproduced?

| Level | Status |
|---|---|
| Numbers from stored episodes, by a second pathway | **Yes.** 95,100 episodes recomputed by the reference verifier, final-state agreement 100% (`CLAIM_VERIFICATION.md`). P1/P2 arms and contrasts reproduced exactly from episodes (`P1_DECOMPOSITION.md`). Red-team cell intervals reproduced (`RED_TEAM_VERIFICATION.md` §4) |
| Verifier independence | **Partial.** The reference verifier and golden traces were written by separate LLM agents from the same specification: a consistency check, not ground truth. No human audit of ProcureSim |
| From a public clone | **Partial.** Published: code, splits, A12/S3/ID1/FCA adapters, Stage 1 traces. **Not published:** P1/P2/AD-C1/SR1/ACK adapters, the P1/P2 traces (~760 MB, author decision), S3 rollout logs. P1 and AD-C1 cannot be re-evaluated from a clone |
| Re-training | not attempted. Hardware differs across lanes (A100 vs H200; disclosed in ID1/SR1) |
| Independence of seeds | **Weaker than counts suggest under greedy decoding.** Identical greedy scores across seeds in AD GRPO (5/5), RSC s0 = s1 on `proc_dev`, three pilot arms (`RED_TEAM_VERIFICATION.md` §7.4). The sampled evaluations (P1–P3, E1) are less affected |

WHAT WE LEARNED → the result files are internally consistent and recomputable. External reproduction of the practical headline (P1) needs published adapters.
WHAT THIS RULES OUT → claiming "independently verified" in the human or third-party sense.
WHAT REMAINS UNCERTAIN → retrain variance across hardware.
NEXT EXPERIMENT → publish the P1 C/D and AD-C1 adapters with `scripts/publish_procure.py` (author approval; outward-facing).

## 8. The narrowest defensible narrative

> **In a simulated procurement environment where tool acknowledgements can be wrong, three results hold up.**
>
> **(1)** A reward computed from database state selects read-back that the starting model already samples. It does not discover read-back when the starting model never samples it.
>
> **(2)** Reading is not repairing. Repair of partial and duplicate payments came from verified recovery demonstrations, not from the reward or from inference-time hints. Starting RL from recovery states adds a small, registered effect on recovery-state evaluation.
>
> **(3)** Std-normalised GRPO, run here as single-update group-baselined policy gradient, is provably blind to per-state stakes when a group shares one hidden state. Removing the std, a known fix, raised verified completion as predicted before the test.
>
> **And:** LLM judges are not substitutes for the state verifier. Judge family and a single rubric sentence decide whether false completions are accepted, even with database evidence.

Supporting evidence, by part:

| Part | Studies | Main figures |
|---|---|---|
| (1) | A12 H2, 5/5 (H3 failed); ID1; K1 | ID1 0/164 sampled reads |
| (2) | P1 H2/H3; P1 H1 and P2 H3/H4 failed; S3-A2 H3′ | +0.259 / +0.306 |
| (3) | TH1; AD-C1 | +0.120, 5/5 |
| Judges | J1, J1 addendum 1, J2, A9 | — |

**Scope, stated with the narrative:**
- one base model;
- one simulator family;
- free authoritative reads;
- a scripted teacher;
- no prompted or larger-model baseline on ProcureSim;
- negative transfer to a sibling domain;
- no gain under renamed tools or a new workflow.

**No new optimizer is claimed.**

**Versus the pasted ranking:**

| Pasted item | Verdict |
|---|---|
| Recovery SFT + GRPO highest | yes, as the practical result credited to demonstrations |
| Recovery-state curriculum defensible | narrower: registered on recovery starts only |
| Judge reliability strong complementary | yes as supporting; standalone pending |
| Reward observability narrow | agree; narrow further to "selection from the repertoire" |
| No FCA | agree |
| No new reward function | agree |
| Capability retention as validation | agree (R1 H2) |

**Elevated above the pasted ranking:** the theory + AD-C1 result. It is the only registered, pre-predicted training-method finding.

WHAT WE LEARNED → the defensible story is about what each training signal can and cannot supply under unreliable acknowledgements, not about a new method.
WHAT THIS RULES OUT → titles or abstracts claiming a new RL algorithm, general robustness, or that the reward "teaches verification".
WHAT REMAINS UNCERTAIN → GRPO's increment over recovery SFT (`p1_dwarm`); P3; judge_next.
NEXT EXPERIMENT → in order of value per GPU-hour, all feasible before Nov 28 only if lanes free by the evening of Nov 26:
1. the `p1_dwarm` evaluation (about 1 GPU-hour);
2. nothing else new;
3. P3 and judge_next as already queued.
