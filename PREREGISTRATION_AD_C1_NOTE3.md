# AD-C1 note 3: D2 gate outcome, evaluation tasks, scheduling edit (no change to the protocol)

Written 2026-11-23 23:29 IST. No cw_conf output exists, and no C1 seed 2–4 run has finished.

## 1. D2 gates, run as registered (22:5x IST, `scripts/ad/d2_gates.py` → `results/ad/d2_summary.json`)

Neither nostd_ema nor awrbn_ema passes G1:

| Arm | VSC difference vs grpo, seeds 0 / 1 |
|---|---|
| nostd_ema | +0.127 / 0.000 |
| awrbn_ema | 0.000 / +0.067 |

Each arm has one seed that collapses to 0 reads, so neither enters C1.

## 2. D3 is not yet run

Tasks 1216 and 1217 are queued but have not started; the main session's tasks currently hold the lanes. If
nostd_ema_irc passes G1–G4, addendum 1 requires its seeds 2–4.

The main session reprioritised GPUs at about 22:0x IST: only tasks already queued may run, and no new GPU tasks may
be added. Under that instruction, D3 seeds 2–4 would **not** be queued. H_D3 and H_D3b would then be reported as
untestable (seeds missing). This is a deviation imposed by compute, and it is decided now, before the D3 outcome
is known.

## 3. Evaluation tasks

`scripts/ad/c1_eval_tasks.py --arms grpo awrb nostd awrbn --after-a 1230_ad_awrbn_s3 --after-b 1231_ad_awrbn_s4
--ids 1240 1241` wrote tasks 1240 and 1241. They were queued in h100 and h200 at 23:28 IST.

**Scheduling edit.** The script gives each task one dependency, but its models train in both chains. Each `# after:`
line was therefore extended to name both chains' relevant ends:
- 1240: `1230_ad_awrbn_s3 1225_ad_awrb_s4`;
- 1241: `1230_ad_awrbn_s3 1231_ad_awrbn_s4`.

This changes only when the tasks may start. It does not change what they evaluate or how.

## 4. Statistics

`scripts/ad/c1_stats.py` reads the d1, d2 and d3 summaries and tests every passing arm. It will run once, after
1240 and 1241 finish.
