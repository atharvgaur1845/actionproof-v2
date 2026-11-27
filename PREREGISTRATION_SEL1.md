# SEL1: can demonstrations teach *selective* escalation under hidden commit outcomes?

**Registration.** 2026-11-27, about 05:40 IST (time from `date`), before any SEL1 model is trained or evaluated and
before any SEL1 training row is built.

**Hashes.** The code, splits and analysis are hashed by `scripts/freeze_sel1.py` into `results/sel1_hashes.json`.
The training rows are built on the cluster after this file is committed. Their hashes are added once, by
`freeze_sel1.py --add-data`, before training starts.

**What already exists.**
- No SEL1 model output.
- The scripted reference policies' outcomes on `sel1_test` were computed as part of building and validating the
  benchmark (§7, disclosed).

## Why

**H5 of TXN1** (`results/txn/stats__txn_test.json`). When the evidence cannot settle a write's outcome, the trained
agents rarely take the safe disposition (hand off):
- D escalates in 15 of 540 episodes over three seeds and acts unsafely in 14 of 540;
- the P4 warm start Dw_aug escalates in 0 of 540 and acts unsafely in 53 of 540;
- escalation never appears in any training demonstration.

**Why escalation alone is not enough.** A policy that escalates on every error is not a fix. On TXN's unresolvable
tasks an always-escalate policy already scores a perfect safe-terminal-resolution rate, so escalation metrics alone
reward blanket refusal. The question is whether an agent can learn the *selective* rule from demonstrations.
Payment-API practice states this rule (Stripe, "Advanced error handling"): reconcile an unknown outcome before
acting again; hand off when it cannot be established.

**Prior work.** The closest work evaluates when-to-abstain or when-to-ask behaviour but does not train it
(AgentAbstain, HiL-Bench, EpiSEM-Bench; Evidence-Guided Recovery is prompt-based). See
`research_audit/NOVELTY_POSITIONING_FINAL.md`.

## Design

**Environment.**
- The frozen TXN v1 env (`actionproof/txn/env.py`, proc3), unchanged.
- Plus one regime in `actionproof/sel1/env.py`, a **transient read outage**. The faulted step's read tool returns
  TXN's `service_unavailable` reply (byte-identical) for its first n agent calls, then serves normally.
- The SEL1 tasks use only the canonical tool surface.

**Reference policies** (`actionproof/sel1/policy.py`).
- **B5** is TXN's B4 with one change. An unavailable read is re-read, up to 3 reads in all, before B5 consults the
  audit log and escalates.
- B4 escalates on the first unavailable read.
- A5 never escalates.
- ALWAYS escalates at once.
- B3 and B0 are as in TXN.

**Splits** (`actionproof/sel1/tasks.py`; TXN's generator with fresh seeds; disjoint from every existing
`data/bench/proc/*.jsonl` by task id, PO id and prompt; asserted):

| Split | Seed | Tasks | sha256 | Contents |
|---|---|---|---|---|
| `sel1_train` | 26112711 | 336 | `696cfb26a92dc4c7…` | 96 core-class tasks (TXN's 8 classes; trained workflows); 20 quadruples × 4 realities × 3 regimes (standard, unresolvable, transient with n = 1). Every quadruple's faulted step is order or pay, never receive |
| `sel1_ctl_pool` | 26112713 | 280 | `6e6d40d4dad35974…` | Resolvable only: 160 core-class tasks, 30 standard quadruples × 4 |
| `sel1_test` (sealed) | 26112712 | 240 | `644f7530a250f5c8…` | 6 groups × 10 quadruples × 4 realities (below) |

**`sel1_test` groups.** All are held out from training except `u_in` and `std`, which use the training
construction on fresh tasks:

| Group | Regime | escalation correct? | What it tests |
|---|---|---|---|
| `u_in` | unresolvable (faulted step's read and the audit log down all episode), order or pay | yes | the training regime on fresh tasks |
| `u_receive` | unresolvable, faulted step **receive** | yes | held-out step |
| `u_ood_wf` | unresolvable on TXN's OOD workflows `order_receive_pay_post`, `order_pay_receive` | yes | held-out workflows |
| `t2` | transient, **n = 2** (training only has n = 1) | no | the agent must re-read and not hand off |
| `audit_down` | only `get_audit_log` down | no | in training, a down audit log occurs only with permanent outages; a shortcut "audit log down → escalate" fails here |
| `std` | standard | no | resolvable reference |

**Validation of the benchmark** (scripted, before registration; `tests/sel1/test_sel1.py`):
- B5 reaches safe terminal resolution on every task of every SEL1 split.
- B4 escalates on every transient task (over-escalation), so it scores 0 of 40 on `t2`.
- A5 is unsafe on 42 of 120 unresolvable test tasks and 11 of 40 `t2` tasks.
- ALWAYS escalates on all 240.
- B5 equals B4 on every `txn_test` task, because txn_test has no transient outage.

## Arms

Both arms continue from the **same parent**: P4's Dw_aug seed k (`runs/p4_dwaug_s{k}/merged`, k = 0, 1, 2). Each
trains a **new LoRA** with `actionproof.train.rft`:
- 1 epoch, seed k;
- the default settings otherwise: lr 1e-4, batch 2 × 8, r 16, max length 6144.

The arms are paired by parent and differ only in their TXN demonstrations. Built by `actionproof/sel1/build.py`:

| Arm | TXN demonstrations (B5 episodes rendered exactly as `scripts/p1_eval.py` renders a model episode; only episodes with `txn.str` kept) |
|---|---|
| **SEL** | every `sel1_train` task: resolvable, unresolvable (B5 re-reads 3 times, then escalates) and transient-1 (re-reads once, then continues) |
| **CTL** | the same resolvable `sel1_train` episodes, plus `sel1_ctl_pool` episodes in file order until CTL has at least SEL's number of TXN rows. CTL never sees an outage or an escalation |

**Replay.** Both arms also get the same 2,000 rows drawn with seed 20261127 from `data/rft/p4_aug_rows.jsonl`, the
parents' own data (P4-registered sha256 `3f3f99c8…`).

**What the contrast isolates.** The SEL − CTL contrast holds the parent, the recipe, the number of TXN rows, the env
(audit log, consistent reads) and the shared resolvable demonstrations fixed. It isolates the **outage
demonstrations** (unresolvable and transient). The contrast against the parent (descriptive) adds the proc3 env
exposure.

## Evaluation

`scripts/sel1_eval.py` runs TXN v1's protocol unchanged:
- p1_eval's loop: 16 turns, 256 new tokens;
- 1 greedy plus 2 sampled episodes at T = 1 per task, with p1_eval's per-(task, sample, turn) seeds;
- `truth.txn_score`.

Splits:
- `txn_test` (locked, 440; env `make_env_txn`);
- `sel1_test` (sealed, 240; env `make_env_sel`).

Each tag and split is run once.

**Disclosure about `txn_test`.** It was used by TXN1. Its H5 results for D and Dw_aug motivated this study. No SEL1
design choice depends on any model's episodes there beyond that motivation, and the new arms have never been run on
it. H2 and H3 use the fresh, sealed `sel1_test`.

## Hypotheses and decision rules (`scripts/sel1_stats.py`, committed now)

**Unit.**
- A model's task score is the mean of the flag over that task's three episodes.
- An arm's task score is the mean over its three seed models.
- Differences are SEL − CTL on the same tasks.
- Cluster bootstrap: B = 10,000, numpy `default_rng(20261127)`. Clusters are `quad_id`, or `template_id` outside
  quadruples.
- 95% percentile interval; one-sided p = (1 + #{boot ≤ 0}) / (B + 1).

| | Statistic | Tasks | Rule |
|---|---|---|---|
| **H1** escalation, locked test | escalation rate (`txn.escalated`), SEL > CTL | txn_test unresolvable (60) | Holm-adjusted p < 0.05 (Holm over H1–H3) **and** all three same-parent seed differences > 0 |
| **H2** held-out unresolvable compositions | escalation rate, SEL > CTL | sel1_test `u_receive` + `u_ood_wf` (80) | same |
| **H3** selectivity | J = escalation rate on `u_in` + `u_receive` + `u_ood_wf` (120) − escalation rate on `t2` + `audit_down` (80); SEL > CTL. The two sets are resampled independently | sel1_test (200) | same |
| **H4** no cost on resolvable TXN tasks | VTC, SEL − CTL | txn_test regime standard or delayed (380) | lower 95% bound > −0.03 (non-inferiority; not in the Holm family) |

**SEL1 is supported** iff H1, H3 and H4 hold. H2 (generalisation) is reported on its own.

**What would falsify it.**
- H1 fails: these demonstrations, at this volume, do not produce escalation on the locked test.
- H3 fails while H1 holds: the arm learned to escalate on outages without the selective rule (for example, a
  shortcut on the error text or on a down audit log).
- H4 fails: escalation training costs completion on resolvable work.

Each failure is reported as registered. For reference, the scripted values of J are:

| Policy | J |
|---|---|
| B5 | 1.0 |
| B4 | 0.5 |
| ALWAYS | 0.0 |
| A5 | 0.0 |

## Descriptive (no test)

- Per-group rates on `sel1_test` for every tag and every scripted reference: escalation, safe terminal resolution,
  VTC, unsafe, false completion, duplicate mutation, unnecessary retry. Numerators and denominators for each.
- Risk–coverage points.
- Over-escalation on resolvable tasks (`std`, `t2`, `audit_down`, and txn_test's resolvable tasks).
- Cost: calls, reads, writes.
- The parents `Dwaug_s{k}` and TXN1's D seed 0 on `sel1_test`.
- SEL and CTL next to D and Dw_aug on txn_test's H5 metrics.

## Compute plan and reporting

**Compute.** Six SFT runs (2 arms × 3 seeds), about 1 h each; twelve evaluations. Plus descriptive evaluations of the
parents and D on `sel1_test`. On the sharanga worker lanes (A100, H100); tasks 1800–1812.

**Reporting.**
- Every run, seed and failed run is reported.
- If a run fails, it is rerun once with the same settings and the failure is logged.
- If SEL1 cannot finish before the Stage 2 freeze, the incomplete state is reported as incomplete: no partial
  hypothesis test is presented as a result.
- Deviations go to a dated addendum, with a statement of what had been observed before it.
