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
| 2026-11-20 08:57 | job 6affbde9efaf5e14f4b28f54 (800_a12_conf_s0) | l40sx1 | A12 seed 0 confirmatory evaluation (2 models × 3 splits, once) | ~3 | ~11 min ≈ 0.35 | |
| 2026-11-20 10:13 | job 6affd008efaf5e14f4b2aeee (610 resume) | h200 | R_V seed 2 resumed from checkpoint-175 (D-A12-2) | ~2 | ~25 min ≈ 2 | |
| 2026-11-20 10:46 | job 6affd7a6efaf5e14f4b2b9d8 (701_a14_main) | l40sx1 | A14 registered run: warm start + 6 frozen adapters, 2 representations | ~4 | | |
| 2026-11-20 10:46 | jobs 6affd7be4fdc5ab3c55ae9bb (801), 6affd7d44fdc5ab3c55ae9d2 (802) | l40sx1 ×2 | A12 confirmatory, seeds 1 and 2 | ~1 | | |
| 2026-11-20 10:47 | jobs 6affd7f1efaf5e14f4b2ba4a (616), 6affd8004fdc5ab3c55aea07 (617), 6affd80eefaf5e14f4b2ba68 (618), 6affd81cefaf5e14f4b2ba74 (619) | h200 ×4 | A12 seeds 3–4 under addendum 2 (registered time rule met; projected total ≈ $103 of the $150 jobs cap) | ~36 | | |
| 2026-11-20 11:56 | jobs 6affe7fdefaf5e14f4b2c766 (701 relaunch), 6affe8164fdc5ab3c55af50b (801 relaunch), 6affe8294fdc5ab3c55af515 (802 relaunch) | l40sx1 ×3 | first attempts stopped at their guards in ~2 min (bundle lacked A14 files; guard checked absent adapters) | ~5 | first attempts ≈ 0.2 | |
| 2026-11-20 12:11 | job 6affeb804fdc5ab3c55af68e (590_a12_warm_dev) | l40sx1 | warm start on proc_dev, reference for the registered H3 test | ~0.2 | ~4 min ≈ 0.15 | |
| 2026-11-20 12:29 | jobs 6affefefefaf5e14f4b2cb84 (803), 6afff00befaf5e14f4b2cbe4 (804) | l40sx1 ×2 | A12 confirmatory, seeds 3 and 4 | ~1 | | |
| 2026-11-20 18:00 | cap change | | user decision: the unused $40 inference-endpoint reserve moves to the jobs cap (jobs cap $190; hard stop $195 unchanged) for K1 with 3 seeds per condition | | | |
| 2026-11-20 18:08 | jobs 6b003f404fdc5ab3c55b6046 (631), 6b003f494fdc5ab3c55b605d (632), 6b003f534fdc5ab3c55b6081 (634), 6b003f5befaf5e14f4b3405c (635) | h200 ×4 | K1 R_A seeds 1–2 on both pools; seeds 0 (630, 633) run on the Slurm H200 worker 460633 at no HF cost | ~38 | | |
| 2026-11-21 00:45 | first K1 HF attempts (6b003f40…, 6b003f49…, 6b003f53…, 6b003f5b…) | h200 ×4 | failed in minutes (D-K1-1) | | ≈ 1 total | |
| 2026-11-21 00:45 | jobs 6b009c584fdc5ab3c55bcc43 (632), 6b009c64efaf5e14f4b3ba24 (634), 6b009c70efaf5e14f4b3ba30 (635) | h200 ×3 | K1 relaunch with the warm-start input; 631 moved to Slurm | ~29 | | |
| 2026-11-21 00:46 | job 6b009c7f4fdc5ab3c55bcc86 (709) | l40sx1 | A15/A16 pilot (dev split only) | ~0.5 | | |
| 2026-11-21 01:01 | job 6b00a0194fdc5ab3c55bd114 (709 relaunch) | h200 | A15/A16 pilot; the L40S attempt (6b009c7f…) was cancelled while still scheduling | ~2 | | |
| 2026-11-21 01:02 | job 6b00a0654fdc5ab3c55bd161 (729) | h200 | A17 GPU pilot (12 pilot tasks per domain, two models) | ~2 | | |
| 2026-11-21 01:05 | job 6b00a0f4efaf5e14f4b3c059 (720) | h200 | CircuitKIT smoke test (install, import, layout) | ~1 | | |
| 2026-11-21 01:12 | job 6b00a2b14fdc5ab3c55bd5e6 (730) | h200 | A17 registered run part 1 (scripted, warm start, R_V s0–4; both domains, main and forks); part 2 (731, R_A) queued on Slurm | ~13 | | |
| 2026-11-21 01:14 | jobs 6b00a31e4fdc5ab3c55bd6e6 (710), 6b00a345efaf5e14f4b3c5f5 (711) | h200 ×2 | A15 registered run (R_V+warm, R_A) | ~10 | | |
| 2026-11-21 01:17 | jobs 6b00a3d1efaf5e14f4b3c663 (712), 6b00a50e4fdc5ab3c55bd8c6 (713) | h200 ×2 | A16 trajectory (R_V, R_A; 13 checkpoints each) | ~10 | | |
| 2026-11-21 01:24 | job 6b00a58defaf5e14f4b3c82b (714) | h200 | A16 layer choice on dev + causal tests R_V s0–2; 715 (remaining models) queued on Slurm | ~8 | | |
| 2026-11-21 01:25 | job 6b00a5c84fdc5ab3c55bd978 (721) | h200 | A16 exploratory CircuitKIT EAP-IG | ~2 | | |
| 2026-11-21 01:26 | projection | | ≈ $110 before tonight + ≈ $77 tonight (K1 ≈ $30, pilots ≈ $4, A15 ≈ $10, A16 ≈ $20, A17 ≈ $13) ≈ **$187**, under the $195 hard stop. R_A halves of A16 causal (715) and A17 (731) run on Slurm at no HF cost. No further HF launches without a fresh estimate. | | | |
