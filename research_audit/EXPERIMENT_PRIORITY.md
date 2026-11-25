# Experiment priority (sprint to Nov 28)

2026-11-25. GPU capacity: the H200 node is draining, which leaves one A100 lane (two leases) and one H100 lane. A Tiny Aya GRPO run (200 steps) with its evaluation takes about 3–5 hours per lane.

## 1. Candidates, ranked

Ranked by:
1. ability to falsify a key claim;
2. contribution;
3. statistical reliability;
4. reproducibility;
5. cost.

| # | Experiment | Question | Hypothesis | Independent variable | Dependent | Controls | Train cost | Eval cost | n | Test | Success / failure | Info gain | Feasible by Nov 28 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **P3** (`PREREGISTRATION_P3.md`) | Does AD-C1's no-std fix improve the best practical agent D? | Dn > D | advantage normaliser | sampled VSC, 450 frozen tasks | same init, data, reward, steps, seeds, evaluator | 3 runs × ~3 h | ~1 h per seed | 450 tasks × 3 seeds | paired task bootstrap, one-sided, plus all seeds positive | registered rule | high: tests the confirmed mechanism in the practical setting | **yes** | **running** (tasks 1610–1613) |
| 2 | **Judge reliability** (`evaluation/judge_next`) | Can a structured, evidence-grounded LLM judge reject false completions? | the state view beats the transcript view; 32B beats 8B; the gate removes false acceptance | view, prompt, model | FAR, FRR, F1, AUROC, ECE, coverage | identical cases, greedy decoding | none | ~3 GPU-hours | 497 cases, 17 categories | paired bootstrap, exact McNemar | descriptive (not registered: development of a judge) | high: the brief's priority workstream | **yes** | **queued** (1600, 1601) |
| 3 | Claim verification | Do headline numbers reproduce from raw episodes by a separate pathway? | — | pathway | per-episode agreement | — | 0 | CPU | 95,100 episodes | — | 100% final-state agreement | high | yes | **done** |
| 4 | Candidate A: adaptive recovery sampling | Does adaptive allocation beat a fixed 50% mix at equal budget? | adaptive > fixed | sampler | VSC on rec/full | equal rollouts | ~15 runs | ~15 h | 5 arms × ≥ 3 seeds | registered | — | moderate; low novelty | **no** | deferred: command in `ALGORITHM_DISCOVERY.md` §4 |
| 5 | Candidate B: constrained GRPO | — | — | — | — | — | — | — | — | — | — | low: already negative in Stage 3 pilots | — | not run again |
| 6 | Candidate C: adaptive group size | — | — | — | — | — | ~9 runs | — | — | — | — | low to moderate | marginal | deferred |
| 7 | ToolSandbox (external, stateful) | Does training transfer to a stateful external benchmark? | descriptive | model | pass rate | overflow stop rule | 0 | adapter work + GPU | — | descriptive | — | moderate | **no** (adapter not built) | deferred |
| 8 | A16 Q0-vs-Q4 direction control | Is the ablated direction "not complete" rather than "partial"? | — | direction | ΔM cross-effects | random directions | 0 | ~1 GPU-hour | 100 tasks | registered addendum | — | low to moderate | possible | deferred (no registration yet) |
| 9 | Dataset proc3 (new faults) | — | — | — | — | — | — | — | — | — | — | moderate | **no**: needs a verifier spec change, golden traces and registration | deferred |

## 2. Rules applied

- **Development data only for selection.** P3's design comes from AD-C1 and the theory. The judge study is development, so nothing in it is called confirmatory.
- **Registration before any confirmatory output.** P3 was committed (7074e8d) before its tasks were queued.
- **Existing registrations are unchanged.** Every freeze check passes (P1, P2, AD-C1, P3).
- **Failed or missing runs are reported:**
  - the judge task's first attempt failed on a context-length setting, and was fixed and requeued;
  - AD-C1's H_D3 was not run (note 3);
  - `s3_rv_mix25_s1` is incomplete (run manifest).
