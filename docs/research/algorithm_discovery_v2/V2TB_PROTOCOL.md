# V2-TB: controlled costed-verification testbed: registration

Registered 2026-11-23 23:05 IST, before any registered testbed output.

**Seen before registration** (logged in SMOKE_EXPERIMENT_LOG.md):
- smoke runs of single configurations, single seed: `scripts/ad2/tab_smoke.py`;
- the plumbing dry run `tab_sweep.py --dry 3`, which runs 3 spread-out runs per family and writes nothing.

**Hash file:** `results/ad2/hashes_V2TB_PROTOCOL.json` (`scripts/ad2/freeze.py V2TB_PROTOCOL`).

## 1. Testbed

The testbed is defined in `actionproof/ad2/tabular/env.py`, with the environments in `envs.py`.

**Workflow.** A workflow is a sequence of K write operations. Each operation has an observable type with
parameters: p (silent-fault rate behind an ok ack), s (repair success), f (false-alarm rate), q_to
(visible-timeout rate) and a (P(applied | timeout)).

**Hidden fault plan.** z ∈ {clean, silent, timeout-applied, timeout-not-applied}^K, drawn in one of two modes:
- `single`: at most one silent fault per workflow;
- `multi`: faults independent per operation.

Repair success and false alarms are per-rollout randomness.

**Decisions.**
- verify or skip after an ok ack;
- repair or proceed after a fault indication;
- re-verify or stop after a repair or a clean read;
- proceed, verify or blind retry after a timeout;
- retry or proceed after a timeout read.

**Return.** R = V·succ − c·reads − L_f·(1 − succ) − D·dups, with V = 1. This is
verified task utility − verification cost − failure loss − duplicate loss: the net utility of the assignment.

**Exact quantities.**
- The optimum J* and any policy's J(π) are computed exactly by expectimax and expectation over the observable
  history tree. The world posterior at every node is exact.
- The per-op VOI is Q(verify) − Q(skip) at the op's ok ack, on the canonical history, with the optimal
  continuation.
- The Rao-Blackwellised return is the exact E_assumed[R | trajectory]: every fault plan consistent with every
  observation, with the rollout's own randomness held fixed.

**Policy.** A softmax head per decision type.
- Mode `tab`: features are an op-type one-hot and "known failed".
- Mode `feat`: features are the observable risk descriptors logit p, log s, log c, logit a, log(1 + D),
  log(V + L_f) and "known failed". The VoI rule is linear in these features.
- The policy never observes z.

**Training (all estimators identical except the estimator).**
- Adam, lr 0.05;
- 4 tasks × G = 8 rollouts per step, 200 steps;
- warm-start verify probability ρ0 = 0.3; repair probability 0.9;
- the loss sums the per-decision log-probability gradients and divides by the number of decisions in the batch
  (TRL 'dapo').

**Group conditions.**
- **A:** all 8 rollouts share the task's fault plan (as in ActionProof).
- **B:** each rollout draws its own plan from the prior.

## 2. Estimators

They are defined in `actionproof/ad2/tabular/train.py`, and each is identified there by its closest prior method.

| Family | Estimators |
|---|---|
| Standard and known fixes | grpo; nostd (Dr. GRPO std removal); rloo; batchstd (TRL "batch" / REINFORCE++); gdpo (TRL normalize_then_sum); ema (first-stage D2); lagr and lagr_grpo (Lagrangian budget under none / group scaling) |
| Learned critics (no privileged information) | critic_v (state-value baseline; REINFORCE with a learned baseline / actor-critic); critic_q (learned Q, all-action gradient; mean actor-critic) |
| Privileged simulator / prior | awrb and awrbn (first stage; the prior may be misspecified); igc (exact −c credit on reads that cannot change the posterior; needs the prior's *support* only) |
| Candidates | ctxema (per-observable-context EMA scale, Direction B); fork_act (forked continuations in the actual world: VinePPO/TreeRL-style, needs a simulator snapshot, no prior); fork_bel (forks in posterior-resampled worlds, POMCP root sampling; needs snapshot and prior); fork_bel_std (fork_bel with a fork-group std: a test of the normaliser-independence principle); cvc (igc where reads are uninformative, belief forks where they are informative) |

Forks are charged as partial episodes in `episodes_used`.

## 3. Families

| Family | Content | Seeds |
|---|---|---|
| TB1 | The required sweep on the core workflow (one risky op, one never-faulted op): p ∈ {0.05, 0.1, 0.2, 0.4, 0.6, 0.8} × c ∈ {0, 0.01, 0.025, 0.05, 0.1, 0.2} × s ∈ {0.3, 0.6, 1.0} × L_f ∈ {0, 1}; 18 estimator/condition arms | 3 |
| TB2 | Rich workflows: 4 op types incl. timeouts with duplicate risk, false alarms, a safe op; 3 workflows trained jointly; mode ∈ {single, multi} × c ∈ {0.02, 0.05, 0.1} × L_f ∈ {0, 1}; D = 0.5. Baselines at 1×, 4× and 8× steps for budget matching; fork estimators at fork_k ∈ {1, 4} | 5 |
| TB3 | Generalisation, feature policy: 24 training tasks on grids p {0.05, 0.2, 0.6}, s {0.5, 1}, c {0.01, 0.05, 0.2}; evaluated on test_iid, test_shift (unseen p {0.1, 0.4, 0.8}, s {0.3, 0.6}, c {0.025, 0.1}) and test_long (unseen workflow length) | 5 |
| TB4 | Prior misspecification on rich multi (c 0.05, L_f 1): awrb, awrbn, fork_bel, cvc, igc × prior ∈ {true, est (empirical-Bayes from revealed reads), mis:0.25, mis:4, flat:0.3}; references nostd and fork_act | 5 |
| TB6 | Mechanism ablations on core (s = 1): grpo vs nostd × ρ0 {0.05, 0.3, 0.6} × G {4, 8, 16} × optimiser {Adam, SGD} × condition {A, B}, at p {0.2, 0.4, 0.6} × c {0.01, 0.05, 0.2} | 3 |

## 4. Registered hypotheses (TB1)

All hypotheses are computed by `scripts/ad2/tab_analyze.py`. "Agreement" means the learned verify decision
(prob > 0.5) has the sign of the exact VOI, over cells with |VOI| > 0.02 and c ≥ 0.01, pooled over seeds.

| ID | Hypothesis |
|---|---|
| H1a | grpo's risky-op verify probability is c-insensitive: in ≥ 80% of (p, s, L_f) cells, its range over c ≥ 0.01 is < 0.25. |
| H1b | grpo's wrong-decision rate is ≥ 0.30; nostd's is ≤ 0.15. |
| H1c | In condition B, grpo's wrong-decision rate is ≤ 0.15: the shared world is necessary for the distortion. |
| H1d | At c = 0, grpo verifies (prob > 0.5) in ≥ 90% of cells with p·s ≥ 0.05. This is the free-read A12 result. |
| H1e | gdpo never verifies (prob < 0.1) in ≥ 90% of cells with c ≥ 0.01. Per-component standardisation turns cost into an O(1) penalty in every group. |
| H1f | batchstd, with 4 groups per batch, has a higher wrong-decision rate than nostd. |
| H1g | Mirror bias. In cells where p < 1/2, p(1 + L_f) > c + 0.02 and VOI < −0.02, awrb verifies in ≥ 50% and awrbn in ≤ 20%. |

## 5. Candidate gates (TB2; revisable only before TB2 output exists)

**Strongest fair baseline (SFB).** For each candidate variant, the SFB is the lowest-mean-regret member of
{nostd, rloo, critic_v, critic_q, ema, ctxema, lagr}. Each member is taken at the smallest step multiplier
(1×, 4×, 8×) whose episodes are ≥ the candidate's. If none is, the 8× run is used, and the gate is marked
`budget_matched=false`.

A candidate passes T1–T3 iff:

| Gate | Requirement |
|---|---|
| T1 | Seed-mean regret improvement over the SFB ≥ 20% of the SFB's regret, and the candidate is better in ≥ 4 of 5 seeds. |
| T2 | Safe-op verify rate ≤ the SFB's + 0.02. |
| T3 | Duplicate loss ≤ the SFB's + 0.01. |
| T4 | On TB3 test_shift, regret ≤ the best non-privileged baseline at matched budget. Read descriptively from the TB3 table. |

Reported per candidate: which privileged information it needs (none / simulator snapshot / prior support /
prior weights).

**Progression.** Only a candidate that passes T1–T3, and whose information requirement is implementable in the
LLM testbed, proceeds to an LLM development registration.

Rejecting a candidate requires a failure diagnosis in FAILURE_ANALYSIS.md.

FILES:
- actionproof/ad2/tabular/env.py
- actionproof/ad2/tabular/policy.py
- actionproof/ad2/tabular/train.py
- actionproof/ad2/tabular/envs.py
- scripts/ad2/tab_sweep.py
- scripts/ad2/tab_analyze.py
- tests/ad2/test_tabular.py

## 6. Deviations

(none yet)
