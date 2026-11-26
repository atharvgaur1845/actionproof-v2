# ActionProof — final-round deck (source)

Source of truth for the slides and the video. Every number resolves through `claims/claims.yaml`
(`python scripts/s2_numcheck.py` scans this file). Six core slides, then an appendix.

---

## 1. The acknowledgement can lie

**"ok". Did the payment happen?**

Same request, same tool, same arguments, same reply, four hidden realities:

| The tool said | What the payment system actually did |
|---|---|
| ok | nothing (phantom write) |
| ok | paid |
| timeout | paid (the acknowledgement was lost) |
| timeout | nothing |

An agent that trusts the reply (B0) falsely completes on 1.000 of the ok + phantom tasks and on 0.000 of the
ok + paid tasks. It fails exactly where the reply and the state disagree.

*Source: TXN benchmark, locked test, 440 tasks (`PREREGISTRATION_TXN1.md`).*

---

## 2. Why retrying is dangerous

- After "timeout", a blind retry pays twice when the first payment did commit.
- After "ok", trusting the reply reports a payment that never happened.
- Checking only the final state cannot tell a repair from a duplicate.

| Policy on the locked TXN test | Verified completion | False completion |
|---|---|---|
| B0: trust the acknowledgement | 0.136 | 0.864 |
| B1: check the final state only | 0.448 | 0.416 |
| B3: bounded retry with idempotency keys | 0.491 | 0.509 |
| B4: reconcile the evidence (scripted reference) | 0.864 | 0.000 |

---

## 3. ActionProof

**Intent → tool call → evidence → reconciliation → safe disposition → post-action verification.**

- ProcureSim: a SQLite procurement world (order, receive, pay, post) with phantom, partial, duplicate, lost-ack,
  late-commit, conflicting-log, concurrent and visibility-lag faults.
- A deterministic verifier reads the database; the agent can see it only through read tools. Every number is scored
  against the database, never the transcript.
- The agent: Cohere Labs Tiny Aya Global (3.35B), recovery demonstrations (SFT, CuratorKIT-deduplicated), then GRPO
  (AgentTune `TrlAgenticGrpo`).

---

## 4. What the experiments show

**Registered P1** (450 frozen tasks, 3 seeds per arm, sampled):

| Arm | Verified completion |
|---|---|
| A: recovery-start GRPO | 0.513 |
| C: recovery SFT | 0.772 (+0.259) |
| D: recovery SFT, then GRPO | 0.819 (+0.306) |

**Registered TXN H3** (D vs blind retry and final-state checking, 320 in-distribution tasks, Holm over 4):
- duplicate mutations 0.024 vs 0.191 (B3) and 0.338 (B1): **both pass**;
- unnecessary retries: **both fail** (Holm 0.123, 0.064), so the registered family is **not supported**.

**Registered P4** (randomised demonstrations, 3 seeds per arm): renamed tools 0.137 → 0.741, unseen four-step
workflow 0.268 → 0.842, no cost on the P1 test (+0.048).

---

## 5. What did not work

- **Escalation.** When evidence cannot settle the outcome, D escalates in 0.050 of cases (the scripted reference
  in 1.000). Escalation was never demonstrated in training.
- **P3.** Removing the group standard deviation from GRPO (no-std, Dn): 0.813 vs 0.819, not supported.
- **Transfer of the RL agents.** Randomising tool names during GRPO (SR1) did not help; healthcare transfer is
  negative (0.158 → 0.047).
- **LLM judges** are not verifier-grade: the best judge with a database snapshot reaches 0.989, the committee 0.897.
- **Frontier models without recovery training** (Qwen3-235B, gpt-oss-120b) repair no partial payment (0.000).

---

## 6. Released, and why it matters

- Live Space: the trained agent runs on a chosen task and reality, then the database is revealed —
  https://huggingface.co/spaces/atharvgaur18/actionproof-live
- Replay Space, models (D, Dw, A, Dn, Dw_aug), traces dataset, code, paper.
- Every headline number has one file and one command; `bash scripts/verify_numbers.sh` checks them without a GPU.
- Use: accounts payable, refunds, payouts, IT change execution — anywhere an "ok" can be wrong.
- Next: escalation demonstrations; a real service behind a fault-injecting proxy; more base models.

---

## Appendix

- **A1. What GRPO adds after SFT** (R7, exploratory): D − Dw +0.140 [+0.120, +0.161]; 69 tasks improve by at least
  0.5, none worsens. D's top remaining failure: false success on a goods receipt that never happened (266 of 976).
- **A2. TXN per-model table**: D seed 0 0.724, Dn seed 0 0.717, Dw 0.598, W 0.445, Aya Expanse 32B 0.161.
- **A3. H3 test table** with intervals: −0.167 [−0.271, −0.070], −0.314 [−0.454, −0.176], −0.027 [−0.067, +0.003],
  −0.095 [−0.199, −0.015].
- **A4. P4 seeds**: renamed tools +0.612, +0.613, +0.587; workflow +0.545, +0.623, +0.553.
- **A5. Frontier baselines per split** (`results/b1/`), greedy: Qwen3-235B full 0.383, gpt-oss-120b full 0.250.
- **A6. Earlier studies**: reward source (A12), selection not invention (ID1), recovery starts (S3), shifts (S4),
  theory (TH1, AD-C1), judges (J1), capability retention (R1).
- **A7. Negative-results table**: `docs/submission/WRITEUP.md` §22.
