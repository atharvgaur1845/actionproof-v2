# Stage 2 experiment registry

Every experiment is listed before it runs. Status values: planned, registered, running, done, failed, not run. A result is never reported here before its file exists.

## Registrations

| Id | Covers | Registration file | Hashes | Status |
|---|---|---|---|---|
| A11 | H1 policy × gateway factorial (e-commerce v2), H4 judge–human audit, H7 recovery matrix, ledger-instructed judge | `PREREGISTRATION_A11.md` | `results/a11_hashes.json` | planned |
| A12 | H2 reward observability in procurement (seeds ≥ 3, target 5), H3 predicted mechanism, reward-truth table | `PREREGISTRATION_A12.md` | `results/a12_hashes.json` | planned (gate Sat Nov 21 14:00) |
| A13 | H5 contract-coverage generalization (G-rep, G-comb, G-adv) | `PREREGISTRATION_A13.md` | `results/a13_hashes.json` | planned (Mon Nov 23 09:00) |
| — | H8 to H12 | none (exploratory, labelled as such) | — | planned |

## Runs

| Task | Registration | What | Pool | Status | Result file |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

## Worker jobs (our account only)

| Job | Pool | GPUs | Submitted | State |
|---|---|---|---|---|
| 460633 | h200 (`worker2.sh`, two lanes) | 2 | 2026-11-20 01:55 | pending |
| 460629 | a100 | 1 | 2026-11-20 01:53 | pending |
