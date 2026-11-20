# Stage 2 experiment registry

Every experiment is listed before it runs. Status values: planned, registered, running, done, failed, not run. A result is never reported here before its file exists. Registration times are commit times (IST).

## Registrations

| Id | Covers | Registration file | Hashes | Status |
|---|---|---|---|---|
| A11 | H1 policy × gateway factorial (e-commerce v2), H4 judge–human audit, H7 recovery matrix | `PREREGISTRATION_A11.md` | `results/a11_hashes.json` | registered 2026-11-20 02:36 (2c83930); H7 done; H1 running; H4 not started |
| A12 | H2 reward-channel read-back, H3 predicted mechanism, reward-truth table | `PREREGISTRATION_A12.md` | `results/a12_hashes.json` | registered 2026-11-20 02:25 (0cf91d9) |
| A12 addendum 1 | rebalanced training split `proc_train_b`, gate at group size 8 | `PREREGISTRATION_A12_ADDENDUM1.md` | `results/a12b_hashes.json` | registered 2026-11-20 06:38 (f1ad004) |
| A12 addendum 2 | seeds 1–4 protocol (stop rule as diagnostic under saturation), confirmatory machinery, reporting rules, reward-signal analysis | `PREREGISTRATION_A12_ADDENDUM2.md` | `results/a12c_hashes.json` | registered 2026-11-20 08:37 (d31c75c), before seeds 1–2 |
| A13 | H5 contract coverage (G-rep, G-comb, G-adv) | `PREREGISTRATION_A13.md` | `results/a13_hashes.json` | registered and run 2026-11-20 06:43; registered test invalid (D-A13-1); finite-set results `results/a13/h5_finite.json` |
| A14 | forced-evidence forks: evidence-conditioned control after a write | `PREREGISTRATION_A14.md` | `results/a14_hashes.json`; code/split/adapters `results/a14b_hashes.json` (at freeze) | registered 2026-11-20 08:41 (0d8a955); pilot (a) and (b) passed; freeze after seeds 1–2 |
| — | H8 to H12 | none | — | not run (descoped in the strategy review) |

## Runs

| Task | Registration | What | Executor | Status | Result file |
|---|---|---|---|---|---|
| 600_a12_warmstart | A12 | warm start + sampled gate | HF h200 6aff6374 | done; gate failed 1 of 4 checks | results/a12/gate.json |
| 601_a12b_regate | A12 add. 1 | gate at group size 8 | HF h200 6aff9dda | done; passed | results/a12b/gate.json |
| 614 / 615 | A12 | GRPO seed 0, R_V / R_A | HF h200 6aff9f52 / 6aff9f5f | done; R_A stop rule fired (D-A12-1) | runs/a12_proc_r?_s0, results/eval/a12/dev, results/a12/rollouts |
| 612 / 613 | A12 add. 2 | GRPO seed 1, R_V / R_A | HF h200 6affb9d6 / 6affb9e2 | running (08:39) | runs/a12_proc_r?_s1 |
| 610 / 611 | A12 add. 2 | GRPO seed 2, R_V / R_A | HF h200 6affb9ee / 6affb9fa | running (08:39) | runs/a12_proc_r?_s2 |
| 616–619 | A12 | GRPO seeds 3–4 (registered time rule) | — | not started | |
| reward signal s0 | A12 add. 2 §3 | per-class reward signal from rollouts (exploratory) | cluster CPU | done | results/a12/reward_signal_s0.json |
| 800 | A12 add. 2 §2 | confirmatory, seed 0 (2 models × 3 splits, once) | HF l40s 6affbde9 | running (08:56) | results/eval/a12/conf/ |
| 801 / 802 | A12 add. 2 §2 | confirmatory, seeds 1 / 2 | — | after seeds 1–2 are frozen | |
| 500–502 | A11 | H1: warm start, PCSTR-d s0, s1 | HF l40s 6affa403 | done | results/a11/eval/ |
| 503–505 | A11 | H1: ack_outcome s0, s1, tool_trust s0 | HF l40s 6affa418 | running | results/a11/eval/ |
| 509 | A11 | offline Observe (CPU) | — | after 503–505 | |
| 506 / 507 | A11 | 32B prompt-only; instructed judge | — | not run (descoped) | |
| h7 | A11 | defences × faults (CPU) | local | done | results/a11/h7_matrix.md |
| h5 | A13 | contract coverage (CPU) | local | done | results/a13/h5_coverage.json, h5_finite.json |
| a14 pilot (a) | A14 §6 | scripted references, proc_dev | cluster CPU | passed | results/a14/pilot/README.md |
| 700 | A14 §6 | pilot (b), R_V s0 step 50, proc_dev | HF l40s 6affbc0f | passed (validity and timing only) | results/a14/pilot/ |
| 701 | A14 | registered run on proc_cf | — | after the a14b freeze | results/eval/a14/, results/a14/a14_summary.json |

## Worker jobs (our account only)

All pending since 2026-11-20 02:04 (no cluster GPU): 460633 (h200 ×2), 460629 and 460648 (a100), 460649 and 460650 (h100).
