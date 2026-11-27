# PREREGISTRATION RG2: does a bounded re-read make the Recovery Gate selective?

Written 2026-11-28 02:44 IST. The registration is the commit that adds `results/rg2_hashes.json` (`scripts/freeze_rg2.py`,
which refuses if any rg2_test output exists). It follows RG1 (43e000c) and RG1 addendum 1 (5c3aa08); no RG1 or
addendum result was seen before this file was written (their runs are still queued).

## 1. Question

The RG1 gate hands off on the first unavailable read of a step that was written. That is correct when the outage is
permanent and premature when the record returns after a short outage. On SEL1's training split, B4 behind the RG1
gate hands off in 80 of 80 transient-outage tasks (scripted, CPU). Gate v2 (`actionproof/recovery_gate/gate_v2.py`)
changes only this:
- When a read of a written step comes back unavailable, the gate reads it again, up to 3 unavailable reads in all
  since the step's last write.
- Only then does the step count as unresolvable.
The budget 3 is SEL1's registered read budget (`READS_BEFORE_ESCALATING`, used by its selective script B5). It was
not chosen from any outcome. Does v2 keep the hand-offs where the record cannot be read, and stop them where it comes
back?

## 2. Design

- **Sealed split.** `rg2_test`: SEL1's test construction (six groups of 10 counterfactual quadruples, 240 tasks) with
  a new seed (26112802). sha256 `a889c8da…`; instance-disjoint from every existing split; never run.
  - **U** (hand-off correct): `u_in`, `u_receive`, `u_ood_wf` (120 tasks).
  - **R** (resolvable outages): `t2` (the record returns after two unavailable reads) and `audit_down` (only the audit
    log is down) (80 tasks).
  - `std` (40 tasks).
- **Confirmatory arms.** V2_D_s{k} (D seed k behind gate v2) and V1_D_s{k} (behind the RG1 gate, unchanged), for
  k = 0, 1, 2.
  - Env: SEL1's env with the outage (`actionproof/recovery_gate/rg2_env.py`).
  - Loop: TXN1's (`scripts/rg2_eval.py`).
  - 1 greedy and 2 sampled episodes per task.
- **Descriptive arms.**
  - Models: D seed 0 with no gate; SEL1's SEL seed 0 behind v2 and with no gate.
  - Scripts (`scripts/rg2_policies_eval.py`): B4, B5, always-escalate; B4 behind v1 and v2; B5 behind v2.

## 3. Hypotheses (scripts/rg2_stats.py, hashed at registration)

Unit, statistic and bootstrap as RG1 (B = 10,000; clusters = quadruples; rng seed 20261129).

| | Metric | Direction | Supported iff |
|---|---|---|---|
| H1 | selectivity J = escalation rate on U minus escalation rate on R; U and R resampled independently | J(V2) − J(V1) > 0 | Holm p < 0.05 (H1, H2) and all three seed differences > 0 |
| H2 | STR on t2 (40 tasks) | V2 − V1 > 0 | same |
| H3 | escalation rate on U | V2 − V1 | lower 95% bound > −0.03 (non-inferiority) |

RG2 is supported overall iff H1, H2 and H3 all hold.

## 4. What was seen before registration

- **Scripted checks on SEL1's training split (CPU).**
  - B4: transient hand-offs 80/80 with no gate and behind v1, 0/80 behind v2; unresolvable hand-offs 80/80 in all
    three.
  - B5: 0/80 transient and 80/80 unresolvable, with or without either gate.
- **Tests.** 4 tests in `tests/recovery_gate/test_gate_v2.py`: transient read-through, permanent hand-off, no
  hidden-state access, no-gate equivalence with SEL1's env.
- **No model was run with gate v2 before registration.** Gate v2 changes only what happens after an unavailable
  read, and its replies are ordinary records.

## 5. Runs, exclusions, deviations

- **Tasks** 1950–1960: the six confirmatory arms, then the descriptive arms, the scripted arms and the analysis
  (task 1960, once, after all).
- **Checks.** Every rg2_test run passes `scripts/freeze_rg2.py --check` first.
- **Exclusions: none.** A crashed run is rerun from scratch.
- **Deviations** go in a dated addendum.
