# PREREGISTRATION TH1: held-out analytical testbed for the latent-state theory of group normalisation

Registered 2026-11-24 (IST) on branch `research/latent-state-theory`, **before any held-out output exists**.
Hashes: `results/th1_hashes.json` (`python scripts/theory/freeze_th1.py`). Evidence level of every TH1 result:
**analytical simulation** (CPU, synthetic). It is not a language-model result and not confirmatory evidence about LLMs.

## 1. Purpose

The theory (`docs/theory/PROOF_AND_ASSUMPTIONS.md`) makes exact predictions for the expected update of one binary
verification decision under any exchangeable group law. TH1 tests these predictions with an **independently
implemented trainer**:
- `actionproof/theory/sim.py`: sampled groups and per-step SGD. It shares no code with `actionproof/ad2` or
  `actionproof/lab`.
- The cells are held out: random problems drawn from a fixed RNG seed that no theorem statement or parameter was fitted
  to.

The exact engine was cross-checked against `actionproof/lab/exact_vote.py` before registration (44 tests).

## 2. What was seen before registration

- **The development split only** (`th1_grid.py --split dev`, seed 101, 60 cells + twins), three runs, used to fix
  coverage of the strata. Changes made from dev information:
  - run 1: the exact regimes covered 2/72 cells, so the exact and neutralise strata were added;
  - run 2: H3 had 4 cells, so the exact stratum uses the shared law w.p. 0.7;
  - run 3: confirms coverage.
- **No change was made to the trainer settings, the estimators or the criteria thresholds** after seeing dev output.
  The thresholds below were written into `th1_analyze.py` before dev run 1.
- **Dev run 3 (final generator), recorded before the held-out run:**
  - H1 met: medians ≤ 0.0003, p90 ≤ 0.0074;
  - H2 1.00 (73 cells), H3 1.00 (4 cells), H5 1.00 (64 cells);
  - **H4 0.80 on its first criterion (10 cells): not met**; 1.00 on the second;
  - **H6 0.857: not met**;
  - H7 untestable (2 pairs).

  The thresholds are kept as written. Explanation, recorded in advance: under Cor. 3.3 the interior equilibrium π*
  approaches 1 as G grows and p → 1/2 (scan: π* = 0.969 at p = 0.45, G = 16). A fixed 0.95 cut can therefore fail
  without contradicting the theory. H1 is the quantitative test of that regime. H4 and H6 may fail on the held-out
  split for this reason.
- The analytical scan `results/theory/scan.json` (canonical problems; not TH1 cells).
- The held-out grid file was generated once, locally, to check that it builds (518 cells under the first generator).
  **Its contents were not inspected**, and no output was produced for it.

## 3. Design (fixed)

- **Cells.** `scripts/theory/th1_grid.py --split heldout`: RNG seed 20261124, 600 cells, plus a Dirichlet-multinomial
  twin of every mixture cell with the same ρ.
  - Strata: broad 45%, exact 25%, neutralise 15%, adversarial 15% (definitions in the file).
  - Parameters: 1–3 fault states, total fault mass U(0.05, 0.9), severities 0 / U(0,3) / U(3,30), repair probability
    1 / 0 / U(0.2,0.95), detection 1 / U(0.5,1), false alarm and duplicate penalty, read cost log-U(0.005, 0.6)
    (state-dependent w.p. 0.15), exogenous failure, G ∈ {2, 4, 8, 16} (reduced if the exact enumeration exceeds 150k
    compositions), law ∈ {shared, iid, mix(ρ), dirmult(ρ)} with ρ ~ U(0.05, 0.95), π₀ ∈ {0.1, 0.3, 0.5, 0.7}.
- **Estimators (10):** grpo, nostd (Dr. GRPO's centring), rloo, batchstd, reinforce (EMA baseline), ppo (EMA
  baseline, batch-normalised advantage, 4 clipped epochs), dapo (zero-variance filter with refill), grpo_kl
  (β = 0.02 to π₀), awrb (observation-measurable return), gdpo (per-component normalisation).
- **Trainer:** plain SGD, lr 0.5, 300 steps, B = 16 groups per step, 10 seeds per (cell, estimator), seeds
  1000·k + 7.
- **Predictions** (exact, `actionproof/theory/exact.py`): π_T from the deterministic mean-field SGD map with the same
  lr and steps; the flow endpoint; corner derivatives; for grpo, nostd, rloo, awrb, gdpo.
- **Compute:** Slurm compute partition, own copy `/scratch/pabitra/lab_theory`. The held-out run is executed once.

## 4. Hypotheses and decision rules (`scripts/theory/th1_analyze.py`, frozen)

A cell's simulated value is the mean over its 10 seeds. Rates carry a 95% bootstrap interval over cells.

| ID | Statement | Cells | Criterion |
|---|---|---|---|
| H1 | The mean-field map predicts the trained read probability | all | median \|sim − pred\| ≤ 0.03 **and** 90th percentile ≤ 0.10, for each of grpo, nostd, rloo |
| H2 | It predicts GRPO's learned decision | \|pred − 0.5\| > 0.1 | agreement (sim > 0.5) = (pred > 0.5) ≥ 0.95 |
| H3 | Stake-free vote (Cor. 3.2) | shared law, deterministic within-state returns, min \|β_z\| ≥ 0.005, vote ≠ 0 | GRPO moves from π₀ in the direction sign(vote) in ≥ 0.95 (also reported: the subset where vote and ΔQ disagree) |
| H4 | Endogenous homogeneity (Cor. 3.3) | law ≠ shared, reads neutralise, vote < 0 < ΔQ, π₀ ≥ 0.3 | GRPO π_T < 0.95 in ≥ 0.9 **and** nostd π_T > grpo π_T in ≥ 0.9 |
| H5 | Centred estimators follow ΔQ under every law (Prop. 2.2) | \|ΔQ\| ≥ 0.05 | nostd moves with sign(ΔQ) in ≥ 0.95 |
| H6 | GRPO is not worse when vote and ΔQ agree | H3 cells with sign(vote) = sign(ΔQ) | GRPO regret ≤ nostd regret + 0.01 in ≥ 0.9 |
| H7 | Pairwise correlation is not sufficient (Prop. 3.5) | mix/dirmult twins whose predicted GRPO π_T differ by ≥ 0.05 | the simulated difference has the predicted sign in ≥ 0.9; **testable only if ≥ 10 such pairs** (otherwise reported as untestable) |

A hypothesis is **supported** iff every criterion in its row is met. No multiplicity correction is applied: these
are prediction-accuracy criteria, not significance tests. Every criterion is reported, met or not.

**Descriptive predictions (no decision rule), stated now:**
- dapo follows GRPO's predicted decision (Prop. 4.4);
- ppo, batchstd and reinforce follow sign(ΔQ) like Dr. GRPO;
- gdpo has the highest regret and never verifies under the shared law;
- awrb is right whenever s = d = 1, the cost is state-independent and f = 0, and fails otherwise
  (state-dependent cost: scan `state_dependent_cost_b`).

## 5. Reported regardless of outcome

Per estimator:
- agreement with the Bayes-optimal action;
- mean and median regret, overall and by law;
- false-completion probability;
- gradient mean and sd;
- rollouts used;
- results by stratum.

Seed-level values are kept in the shard files.

## 6. Deviations

(none yet)

FILES:
- PREREGISTRATION_TH1.md
- actionproof/theory/__init__.py
- actionproof/theory/exact.py
- actionproof/theory/problems.py
- actionproof/theory/sim.py
- scripts/theory/th1_grid.py
- scripts/theory/th1_run.py
- scripts/theory/th1_analyze.py
- slurm/theory/th1_array.sh
- tests/theory/test_exact_theory.py
