# HF credit ledger (Stage 2)

Caps: HF Jobs $150 cumulative; live inference endpoint $40; hard stop $195 (of $200). Every job is logged before it starts, with its reason, and updated with its real duration.

Starting balance: **not yet confirmed** (the CLI cannot read billing; the user checks the HF billing page).

| When (IST) | Job or endpoint id | Hardware | Reason | Est. $ | Actual $ | Cumulative $ |
|---|---|---|---|---|---|---|
| 2026-11-20 02:30 | job 6aff6374efaf5e14f4b20eed (task 600_a12_warmstart) | h200 | A12 warm start + gate; no Slurm GPU started (workers pending since 01:55) and every seed depends on it | ~5–8 | | |
