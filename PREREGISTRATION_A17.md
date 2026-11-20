# A17: cross-domain transfer of reward-induced verification, and the cross-domain arm of A15

Drafted 2026-11-21. The registration time is the commit time of this file and `results/a17_hashes.json`, which is before any model is run on the registered OpsSim splits. Development use before registration:
- the scripted references on the registered splits (they read no model output);
- the GPU pilot, task 729: the warm start and R_V seed 0 on 12 pilot tasks per domain, generated from seeds 131 and 132, which no registered split uses.

Pilot outputs are kept in `results/a17/pilot` and are not analysed. Every earlier registration is unchanged.

## 1. Question

ActionProof's problem is not refunds or payments. It is whether an agent that changes something in an external system establishes that the change actually happened, on the intended entity and with authorisation, before claiming completion. A12 showed that, in procurement, a reward that observes the final state (R_V) produces read-back after writes, and an acknowledgement reward (R_A) does not. A17 asks whether that behavioural difference transfers, **without further training**, to two domains with different tools, state and consequences:
- **IT operations:** deployments, access revocation, backup restores;
- **healthcare administration:** lab-result routing, referrals, contact records. Administrative only.

## 2. Environments and data

- **Domains:** `actionproof/integrity/opsim/it_ops.py` and `care_admin.py`, specified in `docs/stage2/OPSIM_SPEC.md`. Each is self-contained and follows ProcureSim's pattern:
  - one write chokepoint applies a per-(tool, attempt) fault plan the agent never sees;
  - ids are drawn before the fault;
  - a verifier reads only the final state.
- **Fault classes:**
  - *Transfer classes*, analogues of ProcureSim's: clean, phantom, partial, lost acknowledgement, duplicate, never commits.
  - *Out-of-taxonomy classes*, with no ProcureSim training analogue: wrong target, stale read, conflicting views, unauthorised request.
- **Splits:** `data/bench/opsim/{it_ops,care_admin}.jsonl`, 120 tasks each (24 templates × 5 instances), and `{it_ops,care_admin}_cf.jsonl` for the forks.
- **Prompts:** identical across fault classes for the same template instance (tested).
- **Tests:** 84 tests across the two domain files, including golden traces with hand-computed scores.
- **System prompts:** follow the ProcureSim neutral style and never mention verification.

**Scripted references** (registered splits, all classes):

| Policy | it_ops VSC | care_admin VSC | Phantom read-back |
|---|---|---|---|
| oracle (reads after every write) | 0.83 | 0.83 | 1.00 |
| never-read | 0.33 | 0.35 | 0.00 |
| fixed retry | 0.57 | 0.23 | 0.00 |

The oracle's VSC is below 1 because the must-escalate classes never count as completed. Its correct-escalation rate on those classes is 1.00.

## 3. Policies and runs

- **Models:** warm start; R_V and R_A seeds 0–4 (frozen step-200 adapters); the six K1 models when trained (secondary, K1 §3).
- **Settings:** greedy, up to 16 turns, 256 tokens per turn, one run per model per split.
- **Driver:** `scripts/eval_opsim.py`, which uses the lockstep loop, parser and tool-result format of `scripts/eval_proc.py`.
- **Executors:** HF Jobs and the Slurm H200 worker; the executor is logged.

## 4. Endpoints

- **Primary (A12 estimand, transferred):** phantom read-back on each new domain. This is the share of writes acknowledged ok but committed nothing or only part, after which the agent reads the authoritative record for that entity before its next write or report (same rule as ProcureSim's `trace_metrics`). Reported per seed, per domain.
  - Contrasts: R_V minus R_A (same seed) and R_V minus warm start.
  - Statistics: template-cluster bootstrap (10,000 resamples; templates resampled, then instances within templates, because instances of a template are not independent).
  - Across seeds: sign consistency and exact sign-flip p (1/32 one-sided, 2/32 two-sided at five seeds).
- **Key secondary:** VSC and false success on the transfer classes; collateral writes; tool-call validity.
- **Secondary:**
  - per-class outcomes;
  - correct escalation on never-commit, conflicting and unauthorised tasks;
  - `resolved` on wrong-target tasks;
  - `verify_after_write` over all writes. This differs slightly between domains: it_ops also counts writes answered from the idempotency store, care_admin does not. It is therefore compared only within a domain.

## 5. A15 cross-domain arm

- **Forks:** forced-evidence forks on each domain's first write (it_ops `deploy_version`; care_admin `submit_referral`). The prefix is scripted. q = 0, 1/4, 1/2, 3/4 or 1 of the change is stored:
  - it_ops: instances updated out of 4;
  - care_admin: required documents attached out of 4.
- **Classification:** the first decisive action is classified by the domain (`first_decisive`, `fork_correct`).
  - Correct for absent: redo.
  - Correct for partial: complete the remaining part, roll back, or escalate.
  - Correct for complete: proceed (the next step, or report completed when the first write was the last step).
- **Endpoints:** accuracy per q, per model; contrasts as in A15 §3, per domain.
- **Baselines:** never-read and fixed retry. In it_ops, repeating a deploy finishes the remaining instances, so fixed retry is correct at every q < 1 and wrong at q = 1. In care_admin, repeating a referral is not a repair. A policy is therefore credited with evidence use in a domain only if its accuracy exceeds the better of the two scripted references in that domain on q < 1 *and* it stays correct at q = 1.
- **Sample size:** the arms are small (12 base tasks per q per domain). They are reported descriptively with task-bootstrap intervals; no category is assigned.

## 6. Prediction

Transfer is uncertain. A positive result: R_V phantom read-back exceeds R_A's in at least 4 of 5 seeds in a domain, with the template-cluster 95% interval of the seed-averaged difference excluding 0. A null result, R_V ≈ R_A ≈ warm start, would show that the A12 behaviour is bound to the training domain, and it is reported with equal weight.

## 7. Protocol

The registered splits are evaluated once per model. Deviations go to `docs/stage2/DECISIONS.md` with their time.
