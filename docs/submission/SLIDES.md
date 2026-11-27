# ActionProof: final-round deck (source)

**The rendered deck** is a Slides artifact: https://claude.ai/artifact/N5NxS6YHhbaGqGSUMqeYbW (private until the
author shares it). It exports to PDF and PPTX. Its source files are in `docs/submission/deck/`.

**This file mirrors every number and claim on it.** It is scanned by `python scripts/s2_numcheck.py`, so every number
here resolves through `claims/claims.yaml`. The deck shows the same numbers, some rounded for display ($1.49M on a
slide = 1,493,182 here).

**Structure.** Six core slides, then an appendix. The thesis follows `docs/submission/FINAL_STORY_DECISION.md`.

---

## 1. The tool said “ok”. The database disagreed. (hook)

| The tool replied | What the payment system did | If the agent trusts the reply |
|---|---|---|
| ok | nothing (phantom write) | it reports a payment that never happened |
| ok | paid | trusting the reply is right |
| timeout | paid (the acknowledgement was lost) | retrying without checking pays twice |
| timeout | nothing | retrying is right |

- Same request, same tool, same arguments, same reply.
- An agent that trusts the reply falsely completes on 1.000 of ok-but-nothing tasks and on 0.000 of ok-and-paid ones.

*Source: locked TXN benchmark, 440 tasks (`PREREGISTRATION_TXN1.md`). All tools are simulated (ProcureSim).*

---

## 2. Why the usual fixes fail

| Scripted policy, locked TXN test | Verified completion | False completion |
|---|---|---|
| B0: trust the acknowledgement | 0.136 | 0.864 |
| B1: check only the final state | 0.448 | 0.416 |
| B3: bounded retry with idempotency keys | 0.491 | 0.509 |
| B4: reconcile the evidence (scripted reference) | 0.864 | 0.000 |

- Trusting the reply reports payments that never happened.
- Retrying blindly pays twice when the first attempt went through.
- Checking only the end state cannot tell a repair from a duplicate.
- One deterministic episode per task, with the same tools, budget and verifier for every policy. B4 is hand-written,
  not learned.

---

## 3. The agent sees replies. The verifier sees the database.

| | What it is |
|---|---|
| Agent (sees) | Tiny Aya Global (3.35B). It sees the request, the tool schemas and every reply. Recovery demonstrations (SFT), then GRPO through AgentTune. |
| Tools (simulated) | `place_supplier_order`, `pay_supplier`, `record_goods_receipt`, `post_journal_entry`; five reads with `consistent=true`; `get_audit_log`; `escalate_to_human` |
| Database (hidden) | ProcureSim (SQLite). Faults: phantom, partial, duplicate, lost acknowledgement, late commit, conflicting log, concurrent write, visibility lag |

- A deterministic verifier reads the database after the run. It scores verified completion, false completion,
  duplicate mutation and unnecessary retry, and never reads the transcript.
- The benchmark: 440 locked TXN test tasks built around counterfactual quadruples; 8 fault classes, 3 evidence regimes,
  6 OOD groups; 25 of 25 mutants of the scoring code killed by its tests.
- The paired, observation-equivalent design is not new (cf. "Did It Happen?", LIMBO). The composition, the scripted
  ladder B0–B4 and the released episodes are ours.
- The reference verifier shares an author and a specification with the main one, so it is a consistency check, not
  independent ground truth.

---

## 4. Recovery demonstrations carry the capability

**Registered P1** (450 frozen tasks, 3 seeds per arm, sampled):

| Arm | Verified completion |
|---|---|
| A: recovery-start GRPO | 0.513 |
| C: recovery SFT | 0.772 |
| D: recovery SFT, then GRPO | 0.819 |

C and D each beat A (Holm 0.0003, every seed positive). C and D start from different checkpoints, so their gap is not
GRPO's contribution.

**Registered TXN H3** (320 in-distribution tasks):
- Duplicate mutations: D 0.024 against 0.191 for bounded retry with idempotency keys. This test passes.
- Fewer unnecessary retries fails, so the registered family is not supported.

**Registered P4** (warm start, no GRPO, 3 seeds): randomised demonstrations lift
- renamed tools from 0.137 to 0.741;
- the unseen four-step workflow from 0.268 to 0.842;
- with no cost on the P1 test (+0.048).

The RL agent D itself still scores 0.20 on renamed tools.

**Registered P5** (one SFT seed per base), the same recipe on four more Cohere bases:
- Fire 0.018 → 0.686;
- Water 0.011 → 0.639;
- Earth 0.012 → 0.704;
- Aya Expanse 8B 0.076 → 0.772.

---

## 5. It repairs well. It rarely hands off.

- **Escalation when the evidence cannot settle it.** D escalates in 15 of 540 unresolvable cases (three seeds), and
  Dw_aug in 0 of 540. The scripted reference escalates in all of them.
- **Unnecessary retries.** The registered H3 family is not supported: fewer duplicates, but not fewer retries.
- **Transfer of the RL agent.** D scores 0.20 on renamed tools. SR1 did not help. Healthcare transfer is negative.
- **The scripted reference still wins.** On TXN, B4 reaches 0.864 against D's 0.724. No-std GRPO (P3) gives 0.813
  against 0.819: not supported.
- **Registered SEL1 (sealed test): can demonstrations teach selective hand-off?** The target behaviour: escalate when a
  write's outcome is irreducibly unobservable; re-read and continue when the outage is transient. The test runs
  against a matched-volume control from the same parents (3 seeds), with always-escalate and never-escalate
  references beside it. `[result: filled only from results/sel1/sel1_summary.json after the registered analysis]`

---

## 6. Training removes most of the exposure. Checks remove the rest.

Projected cost per 1,000 workflows that meet an ambiguous fault (moderate scenario; `results/impact/impact_txn.json`):

| Agent | Projected cost |
|---|---|
| B0: trust the reply | $1,493,182 |
| B3: bounded retry with idempotency keys | $621,773 |
| D: trained agent (3 seeds; range $128,000 to $162,288) | $145,111 |
| B4: scripted reconciliation checks | $8,182 |

- **Rates vs dollars.** The rates are measured on the locked TXN test. The dollars are projections from stated
  assumptions: 2% fault incidence, a $5,000 payment, a $60 exception, 3× cost for late discovery, 60% duplicate
  recovery. Low and high scenarios are in the appendix. This is a simulator, so there are no observed savings.
- **Links.**
  - Run it live: https://huggingface.co/spaces/atharvgaur18/actionproof-live (a real Cohere model per run; the
    database is revealed after the run).
  - Inspect every episode: https://huggingface.co/datasets/atharvgaur18/actionproof-traces.
  - Re-derive every number without a GPU: `bash scripts/verify_numbers.sh`.
- **Uses.** Accounts payable, refunds, payouts: any write whose "ok" can be wrong.
- **Licence.** The models are CC-BY-NC-4.0, for research use only.

---

## Appendix

- **A1. TXN, all agents (VTC / FCR).**
  - Scripted: B0 0.136 / 0.864; B1 0.448 / 0.416; B3 0.491 / 0.509; B4 0.864 / 0.000.
  - Trained: W 0.445 / 0.475; Dw 0.598 / 0.258; D seed 0 0.724 / 0.121; Dw_aug seeds 0.723 / 0.683 / 0.686 with
    FCR 0.180 / 0.245 / 0.236.
  - Prompted: Aya Expanse 32B 0.161 / 0.610; Qwen3-235B-A22B (supplementary) 0.261 / 0.641.
- **A2. H3 test table.** DMR vs B3: −0.167 [−0.271, −0.070], Holm 0.0012, supported. DMR vs B1: −0.314 [−0.454, −0.176],
  Holm 0.0008, supported. URR vs B3: −0.027 [−0.067, +0.003], Holm 0.123, not supported. URR vs B1: −0.095
  [−0.199, −0.015], Holm 0.064, not supported. D's VTC is non-inferior (+0.205 vs B3, +0.289 vs B1).
- **A3. P4 and P5.**
  - P4 seed differences: renamed tools +0.612, +0.613, +0.587; workflow +0.545, +0.623, +0.553; P1 test +0.027,
    +0.027, +0.091 (Dw 0.666, Dw_aug 0.714).
  - P5: Tiny Aya Water reaches only 0.017 on the unseen workflow.
- **A4. SEL1 design and result** (`PREREGISTRATION_SEL1.md`). Selectivity J = escalation on unresolvable tasks minus
  escalation on degraded-but-resolvable ones. Scripted references: selective (B5) 1, escalate-on-first-error (B4)
  0.5, always-escalate 0, never-escalate 0 (exact by construction; `tests/sel1/test_sel1.py`). `[result: filled from results/sel1/]`
- **A5. Earlier studies and non-claims.**
  - Reward information (A12, ID1): a database reward makes read-back the default in 5 of 5 seeds; it selects
    checking, it does not create it.
  - Judges (J1): the best judge with a database snapshot reaches 0.989, the committee 0.897.
  - The companion theory paper is `paper/main_v3.pdf` (a separate claim).
  - Not claimed: a new optimizer, a novel benchmark design, safety or generalisation beyond the stated splits,
    real-system validation, an independent verifier, a mechanistic circuit.
