# Stage 2 status

Updated at each milestone. Times IST. Last update: Fri 2026-11-20 09:00. Strategy review approved 09:30 plan (narrowed thesis: reward channel vs acquisition; A14 forced-evidence forks for evidence-conditioned control).

| Milestone | Target | State | Notes |
|---|---|---|---|
| M0 bookkeeping, specs, queue | Fri 04:30 | done | Docs in `docs/stage2/`; Slurm workers 460633 (H200 ×2), 460629 and 460648 (A100), 460649 and 460650 (H100) still pending; HF used under the 3–4 h rule. |
| M1 A11 registered, H1 queued | Fri 16:00 | registered; running on HF | A11 registered (`results/a11_hashes.json`). H7 done (`results/a11/h7_matrix.md`). Policy evaluations 500–505 on two HF L40S jobs (relaunched 07:06 after a bundle bug). 506 (32B prompt-only) and 507 (instructed judge) wait for the cluster or a budget decision. |
| M2 ProcureSim + A12 | Sat 14:00 | done | A12 registered; first gate failed (informative groups), Addendum 1 rebalanced the pool, re-gate passed (`results/a12b/gate.json`). H3 predictions committed before step 1. |
| M3 training (seeds 0–2, target 5) | Mon 12:00 | seed 0 done; seeds 1–2 running (HF) | Seed 0 done (R_A stop rule fired, D-A12-1); dev diagnostics: phantom read-back R_V 142/142, R_A 0/96. Addendum 2 registered 08:37; seeds 1–2 launched 08:39. Seed 0 frozen; confirmatory running (800). Reward-signal analysis seed 0 done. |
| M4 gateway process and correctness | Mon | done | Separate-process gateway, contracts, recovery, ledger, dashboard; A13 registered and run (`results/a13/h5_report.md`). |
| M5 analysis, validation, dashboard, CI | Mon | in progress | CI green. Claims ledger: 48 numbers traced across README, video script, product page. A13 reported as finite-set results (D-A13-1). Stage 1 reporting corrected. A14 registered, pilot passed, statistics script ready. |
| M6 freeze, slides, video | Tue | script drafted | `docs/stage2/VIDEO_SCRIPT.md`; freeze Tue 12:00. |
| M7 compliance and submission | Wed 12:00 | not started | Deadline Wed 17:29 |
