# RG1 addendum 1 (registered descriptive, before any run): the frozen gate under transient outages

Written 2026-11-28 02:41 IST, after RG1's registration (43e000c, 02:37 IST) and before any run below. It changes nothing in
PREREGISTRATION_RG1.md and uses the frozen gate (`scripts/freeze_rg1.py --check` runs first).

## Why

RG1's sealed split has only permanent outages: in its unresolvable regime, the step's record and the audit log are
down for the whole episode. There, an immediate hand-off is correct. The gate hands off on the first unavailable
read after a write, so RG1 cannot show what it does when the record comes back after a short outage. SEL1's sealed
test can show this:
- `t2`: the record returns after two unavailable reads;
- `audit_down`: only the audit log is down.
Competitor 02's main result is exactly this kind of selectivity, so the limit must be measured, not assumed.

## Prediction (stated before the runs)

The gate over-escalates on `t2`: it hands off in most `t2` episodes in which a write was sent before the first
unavailable read. It does not hand off on `audit_down`, because the gate never reads the audit log.

Check made before this addendum, on SEL1's training split (CPU, scripted, not the test): B4 behind the gate hands off
in 60 of 60 transient-outage tasks (n = 1). B4 alone also hands off there.

## Runs (once each, sel1_test, 1 greedy + 2 sampled episodes per task)

- `X1_RG_D_s0`: D seed 0 behind the gate (task 1940).
- `X1_RG_SEL_s0`: SEL1's SEL arm, seed 0, behind the gate (task 1941).
- Compared with the existing no-gate episodes of the same models (`results/sel1/D_s0`, `SEL_s0`) and the scripted
  B4, B5 and always-escalate references.
- Analysis: `scripts/rg1x_stats.py` (task 1942). It is descriptive: per group, the counts of hand-offs, STR and unsafe
  episodes, with denominators. No test.

## What would follow

If the prediction holds, the gate's escalation rule is too eager for transient outages. A bounded re-read before a
hand-off would be the fix. It must be developed on training-side tasks and tested on a new sealed split; it is not
tuned on sel1_test.
