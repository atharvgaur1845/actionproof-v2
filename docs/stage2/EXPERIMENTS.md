# Stage 2 experiment registry

Every experiment is listed before it runs. Status values: planned, registered, running, done, failed, not run. A result is never reported here before its file exists.

## Registrations

| Id | Covers | Registration file | Hashes | Status |
|---|---|---|---|---|
| A11 | H1 policy × gateway factorial (e-commerce v2), H4 judge–human audit, H7 recovery matrix, ledger-instructed judge | `PREREGISTRATION_A11.md` | `results/a11_hashes.json` | **registered 2026-11-20 02:40** (2c83930); H7 done |
| A12 | H2 reward observability in procurement (seeds ≥ 3, target 5), H3 predicted mechanism, reward-truth table | `PREREGISTRATION_A12.md` | `results/a12_hashes.json` | **registered 2026-11-20 02:28** (commit 0cf91d9) |
| A12 addendum 1 | rebalanced training split `proc_train_b`, gate at group size 8 | `PREREGISTRATION_A12_ADDENDUM1.md` | `results/a12b_hashes.json` | **registered 2026-11-20 06:35** (f1ad004) |
| A13 | H5 contract-coverage generalization (G-rep, G-comb, G-adv) | `PREREGISTRATION_A13.md` | `results/a13_hashes.json` | **registered and run 2026-11-20 06:43**; result `results/a13/h5_report.md` |
| — | H8 to H12 | none (exploratory, labelled as such) | — | planned |

## Runs

| Task | Registration | What | Pool | Status | Result file |
|---|---|---|---|---|---|
| 600_a12_warmstart | A12 | warm start (SFT on proc_demo) + sampled gate | HF h200 job 6aff6374 | done (16.5 min); gate failed 1 of 4 checks | results/a12/gate.json |
| 601_a12b_regate | A12 add. 1 | gate at group size 8 on proc_train_b | HF h200 job 6aff9dda | running (06:39) | results/a12b/gate.json |
| 500–502, 503–505 | A11 | H1 factorial, six policies | HF l40sx1 jobs 6aff9e50, 6aff9e70 | scheduling (06:41) | results/a11/eval/ |
| h7 | A11 | defences × faults (CPU) | local | done | results/a11/h7_matrix.md |
| h5 | A13 | contract coverage (CPU) | local | done | results/a13/h5_report.md |
| 610–615 | A12 | GRPO seeds 2,1,0 × {R_V, R_A} (Slurm order); HF from seed 0 | all pools + HF | queued | runs/a12_* |
| 616–619 | A12 | GRPO seeds 3,4 (registered time rule) | all pools | queued | |

## Worker jobs (our account only)

| Job | Pool | GPUs | Submitted | State |
|---|---|---|---|---|
| 460633 | h200 (`worker2.sh`, two lanes) | 2 | 2026-11-20 01:55 | pending |
| 460629 | a100 | 1 | 2026-11-20 01:53 | pending |
| 460648 | a100 | 1 | 2026-11-20 02:28 | pending |
| 460649 | h100 | 1 | 2026-11-20 02:28 | pending |
| 460650 | h100 | 1 | 2026-11-20 02:28 | pending |
