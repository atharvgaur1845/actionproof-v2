# HF credit ledger (Stage 2)

Caps: HF Jobs $150 cumulative; live inference endpoint $40; hard stop $195 (of $200). If the cluster stays unavailable, the endpoint reserve moves to jobs (research first) and the live demo uses the scripted agent or a cluster vLLM tunnel; any such move is logged here. Every job is logged before it starts, with its reason, and updated with its real duration.

Starting balance: **not yet confirmed** (the CLI cannot read billing; the user checks the HF billing page).

| When (IST) | Job or endpoint id | Hardware | Reason | Est. $ | Actual $ | Cumulative $ |
|---|---|---|---|---|---|---|
| 2026-11-20 02:30 | job 6aff6374efaf5e14f4b20eed (task 600_a12_warmstart) | h200 | A12 warm start + gate; no Slurm GPU started (workers pending since 01:55) and every seed depends on it | ~5–8 | 16.5 min ≈ 1.4 | 1.4 |
| 2026-11-20 06:39 | job 6aff9ddaefaf5e14f4b254de (601_a12b_regate) | h200 | A12 addendum 1 re-gate (8 samples per task) | ~2 | | |
| 2026-11-20 06:41 | job 6aff9e504fdc5ab3c55a8f47 (500, 501, 502) | l40sx1 | A11 H1 factorial batch A; Slurm pending > 4.5 h (user rule) | ~9 | | |
| 2026-11-20 06:42 | job 6aff9e704fdc5ab3c55a8f73 (503, 504, 505) | l40sx1 | A11 H1 factorial batch B; same reason | ~9 | | |
