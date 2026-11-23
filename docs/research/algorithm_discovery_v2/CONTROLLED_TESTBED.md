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

The results are filled in below from the summary files only.
