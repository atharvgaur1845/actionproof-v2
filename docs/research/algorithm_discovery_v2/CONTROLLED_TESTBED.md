# Controlled costed-verification testbed (CPU)

Registration: `V2TB_PROTOCOL.md`, hashes `results/ad2/hashes_V2TB_PROTOCOL.json`, frozen at 22:08 IST on
2026-11-23, before any output.

- **Code:** `actionproof/ad2/tabular/`.
- **Tests:** `tests/ad2/test_tabular.py`, 15 tests.
- **Raw results:** `results/ad2/tabular/<FAM>.jsonl`, one JSON line per training run.
- **Summaries:** `results/ad2/tabular/<FAM>_summary.json`.
- **Figures:** `figures/`.

## 1. Why this testbed

The LLM testbed cannot isolate the mechanism:
- every arm costs about 1.5 GPU-hours;
- greedy cw_dev outcomes saturate (identical VSC across seeds);
- the optimum is known only for scripted rules.

Here the optimum is **exact**: expectimax over the observable history tree, with exact posteriors. Every policy's
net utility is also computed exactly, so regret is J* − J with no evaluation noise. Hidden fault plans, aliasing,
repair, false alarms, timeouts with duplicate risk, safe operations, single or multiple faults, and Condition A
vs B are all controlled parameters.

## 2. What is exact and what is sampled

| Quantity | How |
|---|---|
| J*(cfg) | expectimax over observable histories, with rows = fault plans × repair bits × false-alarm bits (`oracle_value`) |
| J(π) | exact expectation over the same tree (`policy_value`); verified against Monte Carlo (`test_exact_policy_value_matches_monte_carlo`) |
| VOI of op j | Q(verify) − Q(skip) at op j's ok ack on the canonical history, optimal continuation (`root_voi`); equals p·s·(V + L_f) − c for one op (`test_single_op_voi_formula`) |
| E[R \| τ] (AWRB) | exact over every fault plan consistent with every observation, rollout randomness fixed; unbiased and lower-variance (`test_rao_blackwell_unbiased_and_lower_variance`); consistent worlds reproduce the observations (`test_consistent_worlds_reproduce_the_observations`) |
| Precision, recall, uninformative reads, recovery, duplicates, cost per completion | Monte Carlo evaluation (300–2,000 episodes per task, fixed seed per run) |
| Training | sampled groups; 4 tasks × G = 8 per step; Adam, lr 0.05; 200 steps (×4 or ×8 for budget-matched baselines) |

## 3. Reproduction

```
python -m pytest -q tests/ad2                                   # 20 tests
python scripts/ad2/freeze.py V2TB_PROTOCOL --check
for f in TB1 TB6 TB4 TB3 TB2; do python scripts/ad2/tab_sweep.py $f --workers 14; done
for f in TB1 TB2 TB3 TB4 TB6; do python scripts/ad2/tab_analyze.py $f; done
python scripts/ad2/tab_plots.py --dir results/ad2/tabular --out docs/research/algorithm_discovery_v2/figures  # needs matplotlib
python scripts/ad2/derivation_numeric.py --out results/ad2/derivation_numeric.json
```

**Wall-clock:** 16-core laptop, 14 workers. TB1 is 11,664 runs in about 40 min. The rest are listed in
STATISTICAL_ANALYSIS.md §4.

## 4. Results

All results below come from the summary files.

### 4.1 TB1: the required sweep

The sweep covers p × c × s × L_f = 216 cells, 18 arms and 3 seeds: 11,664 runs.

| Estimator | Agreement with VOI sign | Wrong-decision rate | Mean regret (c ≥ 0.01) | Share of cells where verify prob is c-insensitive |
|---|---|---|---|---|
| cvc | 0.988 | 0.012 | **0.0043** | 0.47 |
| igc | 0.966 | 0.034 | 0.0060 | 0.39 |
| fork_bel | 0.968 | 0.032 | 0.0060 | 0.39 |
| fork_act | 0.949 | 0.051 | 0.0072 | 0.39 |
| ema | 0.952 | 0.048 | 0.0110 | 0.39 |
| awrbn | 0.958 | 0.042 | 0.0123 | 0.39 |
| critic_q | 0.947 | 0.053 | 0.0124 | 0.42 |
| ctxema | 0.929 | 0.071 | 0.0127 | 0.44 |
| nostd = rloo | 0.943 | 0.057 | 0.0132 | 0.39 |
| nostd_B | 0.943 | 0.057 | 0.0160 | 0.42 |
| critic_v | 0.941 | 0.059 | 0.0163 | 0.39 |
| awrb | 0.909 | 0.091 | 0.0192 | 0.53 |
| grpo_B | 0.685 | 0.315 | 0.0440 | 0.86 |
| batchstd | 0.634 | 0.366 | 0.0480 | 0.83 |
| lagr_grpo | 0.372 | 0.628 | 0.1235 | 0.75 |
| **grpo** | **0.356** | **0.644** | **0.1388** | **0.97** |
| gdpo | 0.253 | 0.747 | 0.1880 | 1.00 |

The fork arms in TB1 are **not** budget-matched; TB2 matches them.

**Registered hypotheses.**

| ID | Result | Value |
|---|---|---|
| H1a | supported | grpo is c-insensitive in 97% of cells |
| H1b | supported | wrong-decision rate: grpo 0.644 ≥ 0.30; nostd 0.057 ≤ 0.15 |
| **H1c** | **not supported** | grpo_B 0.315 > 0.15 (V-F1) |
| H1d | supported | at c = 0, grpo verifies in 100% of cells with p·s ≥ 0.05 |
| **H1e** | **not supported** (narrowly) | gdpo never verifies in 89% < 90% of cells (V-F2) |
| H1f | supported | batchstd 0.366 > nostd 0.057 |
| H1g | supported | awrb verifies in 18/18 mirror cells; awrbn in 0/18 (DV-3) |

**Slices** (c = 0.05, L_f = 0; learned P(verify) at p = 0.05, 0.1, 0.2, 0.4, 0.6, 0.8).

s = 1, where VOI = 0.00, 0.05, 0.15, 0.35, 0.55, 0.75:

| Estimator | P(verify) by p |
|---|---|
| grpo | 0, 0, 0, 0.01, 0.95, 0.99 (the vote threshold p = 1/2) |
| grpo_B | 0, 0, 0.01, 0.84, 0.98, 0.99 |
| gdpo | 0, 0, 0, 0, 0.01, 0.29 |
| nostd | 0.08, 0.92, 0.97, 0.99, 0.99, 0.99 |
| cvc | 0.02, 0.98, 0.99, 0.99, 0.99, 0.99 |

s = 0.3, where VOI = −0.035, −0.02, 0.01, 0.07, 0.13, 0.19:

| Estimator | P(verify) by p |
|---|---|
| grpo | 0 everywhere: p·s ≤ 0.24 < 1/2, so it never verifies even when VOI = +0.19 |
| awrb | 0.01, **0.99**, 0.99, 0.98, 0.97, 0.04 (mirror bias at p = 0.1, where VOI < 0) |
| awrbn | 0.01, 0.03, 0.24, 0.94, 0.97, 0.98 |
| cvc | 0, 0, 0.51, 0.98, 0.99, 0.99 |

At p = 0.2, s = 1, grpo verifies with probability 0.99 at c = 0 and 0.00 at every c ≥ 0.01. Its net utility is
0.80, against an optimum of 0.99 at c = 0.01.

**Figures.**
- `figures/tb1_verify_vs_p.png`: verification probability vs fault probability;
- `figures/tb1_verify_vs_voi.png`: vs expected VOI;
- `figures/tb1_utility_vs_cost.png`: net utility vs read cost;
- `figures/tb1_regret.png`: regret vs the oracle.

### 4.2 TB2: candidate gates and selective verification

See SELECTIVE_VERIFICATION_RESULTS.md and ALGORITHM_CANDIDATES.md.
- igc passes T1–T3 at 1× and 4× budget, not at 8×.
- Fork-based candidates fail at matched budget.
- Figure: `figures/tb2_safe_vs_risky.png` (safe- vs risky-op verify rates).

### 4.3 TB3, TB4 and TB6

See GENERALISATION_RESULTS.md (TB3, TB4) and ABLATION_RESULTS.md (TB6).
