# Stage 2 status

Updated at each milestone. Times IST. Last update: Fri 2026-11-20 07:15.

| Milestone | Target | State | Notes |
|---|---|---|---|
| M0 bookkeeping, specs, queue | Fri 04:30 | done | Docs in `docs/stage2/`; Slurm workers 460633 (H200 ×2), 460629 and 460648 (A100), 460649 and 460650 (H100) still pending; HF used under the 3–4 h rule. |
| M1 A11 registered, H1 queued | Fri 16:00 | registered; running on HF | A11 registered (`results/a11_hashes.json`). H7 done (`results/a11/h7_matrix.md`). Policy evaluations 500–505 on two HF L40S jobs (relaunched 07:06 after a bundle bug). 506 (32B prompt-only) and 507 (instructed judge) wait for the cluster or a budget decision. |
| M2 ProcureSim + A12 | Sat 14:00 | done | A12 registered; first gate failed (informative groups), Addendum 1 rebalanced the pool, re-gate passed (`results/a12b/gate.json`). H3 predictions committed before step 1. |
| M3 training (seeds 0–2, target 5) | Mon 12:00 | seed 0 running (HF H200) | 614 (R_V) and 615 (R_A) started 06:46. Stop rule fired for R_A seed 0 (0.864 > 0.80 over steps 40–50); continued as logged deviation D-A12-1. R_V passed (0.545). |
| M4 gateway process and correctness | Mon | done | Separate-process gateway, contracts, recovery, ledger, dashboard; A13 registered and run (`results/a13/h5_report.md`). |
| M5 analysis, validation, dashboard, CI | Mon | in progress | CI green on GitHub (ruff, mypy, frozen-file guard, 708 tests, CLI run, numcheck). `claims/claims.yaml` and `scripts/s2_numcheck.py` in place. Pending: `s2_stats.py` on seeds, A11 estimands. |
| M6 freeze, slides, video | Tue | script drafted | `docs/stage2/VIDEO_SCRIPT.md`; freeze Tue 12:00. |
| M7 compliance and submission | Wed 12:00 | not started | Deadline Wed 17:29 |
