# Stage 2 status

Updated at each milestone. Times IST. Last update: Fri 2026-11-20 13:00. Strategy review approved 09:30 plan (narrowed thesis: reward channel vs acquisition; A14 forced-evidence forks for evidence-conditioned control).

| Milestone | Target | State | Notes |
|---|---|---|---|
| M0 bookkeeping, specs, queue | Fri 04:30 | done | Docs in `docs/stage2/`; Slurm workers 460633 (H200 ×2), 460629 and 460648 (A100), 460649 and 460650 (H100) still pending; HF used under the 3–4 h rule. |
| M1 A11 registered, H1 queued | Fri 16:00 | registered; running on HF | A11 registered (`results/a11_hashes.json`). H7 done (`results/a11/h7_matrix.md`). Policy evaluations 500–505 on two HF L40S jobs (relaunched 07:06 after a bundle bug). 506 (32B prompt-only) and 507 (instructed judge) wait for the cluster or a budget decision. |
| M2 ProcureSim + A12 | Sat 14:00 | done | A12 registered; first gate failed (informative groups), Addendum 1 rebalanced the pool, re-gate passed (`results/a12b/gate.json`). H3 predictions committed before step 1. |
| M3 training (seeds 0–2, target 5) | Mon 12:00 | done (5 seeds) | Seeds 0–4 trained and frozen (seed 0 R_A stop rule fired, D-A12-1; seed 2 R_V resumed, D-A12-2). Confirmatory: seeds 0–3 done, seed 4 running. **H2 holds in seeds 0–2** (phantom read-back 1.00 vs 0.00; task level 153/153 vs 0/153); **H3 fails**, so composite C1 is not made. **A14:** every policy acts correctly on present/absent evidence (BCA 1.00), thesis check met, 'seeing is not knowing' abandoned; partial-amount branch R_V 0.35–0.48 vs R_A 0. A11: runtime check passes; registered interaction not supported. |
| M4 gateway process and correctness | Mon | done | Separate-process gateway, contracts, recovery, ledger, dashboard; A13 registered and run (`results/a13/h5_report.md`). |
| M5 analysis, validation, dashboard, CI | Mon | in progress | CI green. Claims ledger: 48 numbers traced across README, video script, product page. A13 reported as finite-set results (D-A13-1). Stage 1 reporting corrected. A14 registered, pilot passed, statistics script ready. |
| M6 freeze, slides, video | Tue | script drafted | `docs/stage2/VIDEO_SCRIPT.md`; freeze Tue 12:00. |
| M7 compliance and submission | Wed 12:00 | not started | Deadline Wed 17:29 |
