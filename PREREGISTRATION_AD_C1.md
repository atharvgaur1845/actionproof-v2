# Pre-registration AD-C1: estimators for costly verification (confirmatory)

Registered 2026-11-23 16:21 IST.
- **Seen at registration:** AD-D1 seed-0 outputs on cw_dev (`docs/research/algorithm_discovery/SMOKE_TEST_LOG.md`).
- **Not seen:** no seed-1 output, and no D2 output. `cw_conf` has never been evaluated by any model.
- **Hash file:** `results/ad_c1_hashes.json`.

## 1. Arms, by a fixed rule

`grpo` (D1) is always the baseline. A candidate enters C1 iff it passes all of G1–G4 in its development round:
- D1: `awrb`, `awrbn`, `nostd`;
- D2: `nostd_ema`, `awrbn_ema`.

`scripts/ad/d1_gates.py` and `scripts/ad/d2_gates.py` apply the rule mechanically. If no candidate passes, C1 runs
no new training, and the development results are the final word.

## 2. Seeds and training

- **Seeds 0–4 per entering arm and for grpo.** Seeds 0–1 are the frozen development runs; seeds 2, 3 and 4 are new.
- **Training as in the arm's development round,** except that no development evaluation is run for seeds 2–4.
- **Every model is evaluated once on `cw_conf`** (300 tasks, seed 1203, generated at 11:27 and never opened since):
  greedy, plus 4 samples at T = 1, c = 0.05 (`actionproof/ad/evaluate.py`).

## 3. Hypotheses

**H_c, for each entering candidate c.** VSC(c) > VSC(grpo) on cw_conf, greedy.

**H_AWRB.** Tested only if both members of a pair enter:
- awrbn > nostd (D1 scale);
- awrbn_ema > nostd_ema (D2 scale).

**Statistic.** The seed-averaged VSC difference.

**Test.** A one-sided unpaired seed-level permutation test: 5 vs 5, 252 assignments.

**Multiplicity.** Holm over every H tested in C1.

**Decision rule.** H is *supported* iff both hold:
- the Holm-adjusted p < 0.05;
- the seed-averaged paired task-bootstrap 95% interval (runs fixed, the project convention) excludes 0.

Also reported: the two-level interval (seeds, then tasks), per-seed differences, and the number of positive seeds.

## 4. Secondary measurements (descriptive, no tests)

- **On cw_conf:** J_c, reads per episode, verify-after-write by site, faulted-task VSC, sampled VSC, duplicate rate,
  and false completion.
- **Non-regression on the standard free-read split** `proc_dev` (VSC): seed 0 of each arm.
- **Shifts, seed 0 of each arm, through `scripts/s4_eval.py` unchanged:**
  - `s4_api` (renamed tools);
  - `s4_lang_hi` (Hindi);
  - `s4_wf4` (unseen 4-step workflow).

  Measured: VSC and reads.
- **Capability retention:** the R1 harness of the main session (`actionproof/retention/gen.py`,
  `scripts/r1_score.py`) on seed 0 of each entering arm and grpo, against the warm start (IFEval, BFCL).
- **Mechanism:**
  - P(read | ok acknowledgement) split by the actual world. Aliased worlds should not differ; this is a leakage
    check.
  - P(repair | a read revealed a fault).
  - Reads after the journal entry (zero value under cw1).
  - The c = 0.30 intervention (D1 tasks 1209 and 1211).

## 5. What C1 cannot show

- Superiority over the free-read-trained ActionProof baselines (RSC, A12 R_V). These differ in their training pools
  and rewards. They are reported on cw_conf as context only.
- Selectivity: development showed none (FAILURE_ANALYSIS F3).
- Anything about other model families or domains.
