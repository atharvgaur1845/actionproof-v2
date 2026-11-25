# TXN1: transaction integrity under ambiguous execution (the TXN benchmark v1)

Registered 2026-11-26, about 02:30 IST, before any model sees the locked test split. The benchmark specification
is in `docs/txn/TXN_SPEC.md`. Code: `actionproof/txn/`, `scripts/txn_*.py`. Splits are hashed in
`data/bench/proc/txn_manifest.json` and in `results/txn1_hashes.json`. Governing plan: the approved sprint plan,
§2–6.

## Benchmark

**Environment.** proc3 (`TxnEnv`) keeps every ProcureSim fault and adds:
- an audit log of requests received, as a second evidence source;
- `log_conflict`: the log says failed, the ledger shows the commit;
- `late_commit`: a timed-out request commits after the agent's 2nd later call;
- `concurrent`: a colleague's identical write lands first;
- the evidence regimes `delayed` (`visibility_lag`, with a `consistent` read flag) and `unresolvable` (the step's
  read service is down).

**Locked test** `txn_test`, 440 tasks:

| Group | Tasks |
|---|---|
| core: 8 fault classes × 25 (phantom, partial, lost_ack, dup_delivery, visibility_lag, log_conflict, late_commit, concurrent) | 200 |
| counterfactual quadruples: 15 POs × 4 hidden realities × 2 regimes (resolvable, unresolvable) | 120 |
| OOD: 6 groups × 20 (tool alias, schema, workflow, fault combination, evidence shift, domain vocabulary) | 120 |

The dev and val splits are only for debugging. No model was run on either before this registration. The scripted
policies were run on dev while the benchmark was being built.

## Agents (every one evaluated once on `txn_test`)

**Scripted** (`scripts/txn_policies_eval.py`, deterministic):
- B0: acknowledgement-trusting;
- B1: final-state-only;
- B2: audit-log-only;
- B3: bounded retry with idempotency;
- B4: transaction-aware reconciliation;
- ablations A1–A6 of B4.

**Models** (`scripts/txn_eval.py`: P1's loop, 1 greedy + 2 sampled episodes at T = 1):
- W, A_s0–2, Dw, D_s0–2, Dn_s0–2;
- tag_base (Tiny Aya Global, untrained);
- aya8b;
- aya32b_bf16;
- secondary, if P4 finishes before the GPU gate: Dwaug_s0–2.

**API (supplementary):** `qwen` (Qwen3-235B-A22B-Instruct-2507 through the HF router), greedy, `--max-usd 3`. The
combined API spend across B1 and TXN1 stays within the author's $10. B1's cap is lowered to $7 to keep that true.

## Hypotheses (`scripts/txn_stats.py`, committed now)

- **H1 (descriptive).** B0's false-completion rate per fault class, with exact Clopper–Pearson intervals.
- **H3 (confirmatory).** The trained agent D (per-task mean over D_s0–2) has fewer harmful recovery actions than
  bounded retry (B3) and final-state-only checking (B1).
  - Tests: DMR and URR, both lower-is-better, on the in-distribution groups (core and quad), Holm over 4 tests.
  - A test is supported only if its Holm-adjusted template-clustered bootstrap p < 0.05, **and** the clustered 95% CI
    of D − comparator lies below 0, **and** D's VTC is non-inferior with margin 0.05.
  - H3 is supported only if all 4 tests are.
  - FCompR is reported descriptively.
  - **Change before registration:** the brief had FCompR in the H3 family, but B1 and B3 never refund, so their
    FCompR is 0 by construction and that test could never pass. FCompR was dropped from the family before any
    model output existed.
- **H4 (descriptive).** For every agent, per OOD group: STR, FCR, DMR and harm, against the in-distribution core
  group.
- **H5 (descriptive).** On the unresolvable regime: correct escalation rate, unsafe-action rate, escalation
  precision, and coverage–risk.

**Statistics.** The bootstrap resamples **templates** (group, workflow, fault/reality, regime), B = 10,000, seed
20261126. Rates always show numerator and denominator.

## Gates (no TXN result is reported unless all hold)

- **Tests pass:** `pytest tests/txn`, 142 at registration.
- **Mutation testing:** every one of 25 seeded verifier and environment mutants is killed (`results/txn/mutation_report.json`).
- **Splits unchanged:** `scripts/txn_make_splits.py --check` prints OK.
- **proc2 unchanged:** the 91 proc1 goldens agree 100% with the reference verifier, and the proc2 behaviour checks
  pass.
- **Noted limitation:** `scripts/reference_verifier_txn.py` was written by the same agent as `actionproof/txn/truth.py`.
  It is a second implementation, not an independent one.
