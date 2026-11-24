# Site and dashboard: visual spec for the theory results (for the lead engineer to wire up)

**Data:** `results/theory/site_theory.json` (`scripts/theory/site_data.py`, Slurm). Every series is computed by the
exact engine or copied from a result file. Do not type numbers into the site. Each chart carries its `labels` entry
as a visible evidence badge.

| # | Chart | Data key | Encoding | Badge | Message it must support, and no more |
|---|---|---|---|---|---|
| 1 | Execution ambiguity | none (static diagram) | the same "ok" acknowledgement over clean / phantom / partial / duplicate database states | — | the acknowledgement does not reveal the state |
| 2 | Decision-boundary phase diagram | `phase` | x = fault probability p, y = read cost c (log); three small multiples: GRPO (shared), GRPO (iid), Dr. GRPO; cell = learned verify (endpoint ≥ 0.5); overlay the Bayes boundary c = p | analytical (exact) | GRPO's boundary is vertical at p ≈ 1/2 (cost-blind) under the shared law. Dr. GRPO's follows c = p. iid moves GRPO partway. Counts: Bayes agreement 323/500, 406/500, 500/500 |
| 3 | Expected update vs value | `update` | x = ΔQ (= 0.3 − c), y = expected update at π = 0.3; two lines | analytical (exact) | Dr. GRPO's update is proportional to ΔQ. GRPO's stays negative at every ΔQ > 0 shown |
| 4 | Read probability vs read cost | `cost` | x = c (log), y = learned read probability, p ∈ {0.3, 0.6} | analytical (exact) | GRPO is flat in c. Dr. GRPO switches at c = p |
| 5 | Group-correlation sensitivity and regret | `rho` | x = ρ, y = endpoint and regret; two families (mix, dirmult) | analytical (exact) | a small shared fraction restores the collapse. Same pairwise correlation, different outcome |
| 6 | Theory vs trained agent | `llm` | read share per 10 steps; one line per run, coloured by train scale (group / none / ema) | language agent, development, descriptive | std-normalised runs lose reading; centred runs gain it. Do **not** label this "confirmatory" |
| 7 | Estimator comparison on held-out problems | `th1` | dot plot: Bayes-action agreement and median regret per estimator | analytical simulation, registered | centred estimators ≈ 0.97 agreement; GRPO, DAPO, GRPO+KL ≈ 0.60; GDPO-style 0.22 |
| 8 | Task-level results | canonical pipeline (P1, AD-C1 when reported) | per-seed dots + interval | confirmatory | only registered outcomes |
| 9 | Failure decomposition | `results/quality/diagnosis_p1_*.json` | stacked bars per arm | descriptive | — |
| 10 | Judge limitations | `results/j2/j2_summary.json` | false acceptance by judge × view | model-vs-verifier, exploratory; **no human labels** | the policy-family judge accepts 70% of failed executions without a snapshot |

**Required page text:** "Analytical results are exact computations on a stylised problem. The language-agent curves
are development runs (≤ 2 seeds per arm, except AD-C1). No judge result is validated against human labels."
