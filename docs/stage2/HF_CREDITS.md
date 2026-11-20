# HF credit ledger (Stage 2)

Caps: HF Jobs $150 cumulative; live inference endpoint $40; hard stop $195 (of $200). If the cluster stays unavailable, the endpoint reserve moves to jobs (research first) and the live demo uses the scripted agent or a cluster vLLM tunnel; any such move is logged here. Every job is logged before it starts, with its reason, and updated with its real duration.

Starting balance: **not yet confirmed** (the CLI cannot read billing; the user checks the HF billing page).

| When (IST) | Job or endpoint id | Hardware | Reason | Est. $ | Actual $ | Cumulative $ |
|---|---|---|---|---|---|---|
| 2026-11-20 02:30 | job 6aff6374efaf5e14f4b20eed (task 600_a12_warmstart) | h200 | A12 warm start + gate; no Slurm GPU started (workers pending since 01:55) and every seed depends on it | ~5–8 | 16.5 min ≈ 1.4 | 1.4 |
| 2026-11-20 06:39 | job 6aff9ddaefaf5e14f4b254de (601_a12b_regate) | h200 | A12 addendum 1 re-gate (8 samples per task) | ~2 | ~6 min ≈ 0.5 | 1.9 |
| 2026-11-20 06:46 | job 6aff9f52efaf5e14f4b256f4 (614 R_V seed 0) | h200 | A12 GRPO seed 0; all Slurm workers pending > 4.5 h (user rule) | ~20 | | |
| 2026-11-20 06:46 | job 6aff9f5fefaf5e14f4b256fe (615 R_A seed 0) | h200 | same | ~20 | | |
| 2026-11-20 06:41 | job 6aff9e504fdc5ab3c55a8f47 (500, 501, 502) | l40sx1 | A11 H1 factorial batch A; Slurm pending > 4.5 h (user rule) | ~0.1 (failed in 3 min: bundle tar ownership error, rc 127, no eval run) | | |
| 2026-11-20 06:42 | job 6aff9e704fdc5ab3c55a8f73 (503, 504, 505) | l40sx1 | A11 H1 factorial batch B; same reason | ~0.1 (failed in 2 min: same) | | |
| 2026-11-20 07:06 | job 6affa403efaf5e14f4b25e5e (500, 501, 502) | l40sx1 | A11 batch A relaunch after the runner fix (`tar --no-same-owner`) | ~9 | | |
| 2026-11-20 07:06 | job 6affa418efaf5e14f4b25e9e (503, 504, 505) | l40sx1 | A11 batch B relaunch, same fix | ~9 | | |
| 2026-11-20 08:39 | jobs 6affb9d6efaf5e14f4b28712 (612 R_V s1), 6affb9e24fdc5ab3c55abc04 (613 R_A s1), 6affb9eeefaf5e14f4b28736 (610 R_V s2), 6affb9fa4fdc5ab3c55abc4c (611 R_A s2) | h200 ×4 | A12 seeds 1–2 under addendum 2; no Slurm GPU since Nov 19 | ~40 | | |
| 2026-11-20 08:40 | note | | seed 0 (614, 615) ran ~1 h 50 min each including dev evaluations; A11 batch A (6affa403) ran ~45 min | | 614/615 ≈ 9 each; 6affa403 ≈ 1.5 | |
| 2026-11-20 08:57 | job (800_a12_conf_s0) | l40sx1 | A12 seed 0 confirmatory evaluation (2 models × 3 splits, once) | ~3 | | |
