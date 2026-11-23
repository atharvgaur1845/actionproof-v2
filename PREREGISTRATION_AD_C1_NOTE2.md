# AD-C1 note 2: completing the arm set by the registered rule (no change to the protocol)

Written 2026-11-23 21:40 IST, before any action it describes. `PREREGISTRATION_AD_C1.md`, its addendum 1 and their
hash files are not edited.

**What was seen:**
- the cw_dev development summaries of every D1 seed-0 and seed-1 run (grpo, nostd, awrb, awrbn);
- the D1 gate output, `scripts/ad/d1_gates.py`, unchanged.

**Not seen:** any cw_conf output, any C1 seed 2–4 output, and any D2 or D3 output.

**Fact.** With all four D1 arms at two seeds on file, `d1_gates.py` reports that all three D1 candidates pass
G1–G4. All numbers are cw_dev greedy, against grpo:

| Arm | VSC difference per seed | J_c difference | Duplicate difference |
|---|---|---|---|
| awrb | +0.160 / +0.160 | +0.029 | 0 |
| awrbn | +0.067 / +0.067 | +0.015 | 0 |
| nostd | +0.127 / +0.127 | +0.040 | 0 |

**Rule.** PREREGISTRATION_AD_C1.md §1: "A candidate enters C1 iff it passes all of G1–G4 in its development round."
§2: seeds 2–4 are trained for every entering arm. §3: H_AWRB (awrbn > nostd) is tested iff both members enter.
`scripts/ad/c1_stats.py` reads the gate summaries and expects seeds 0–4 of every passing arm.

**Gap.** Seeds 2–4 were queued for grpo and awrb only (tasks 1220–1225). Those tasks were queued when awrb's two
seeds were the only candidate seeds on file. nostd s1 and awrbn s1 finished afterwards (tasks 1207 and 1208), and
their C1 seeds were never queued.

**Action.** Follow the registration exactly; there is no protocol decision to make.
- **Training tasks.** `scripts/ad/c1_tasks.py --arms nostd awrbn --after-a 1224_ad_awrb_s3 --after-b 1225_ad_awrb_s4
  --first 1226` writes tasks 1226–1231: nostd s2–s4 and awrbn s2–s4. The script, and the arm script
  `slurm/ad_arm_c1.sh`, are the registered ones and are unchanged.
- **Chain placement.**
  - Each task depends on the end of one existing C1 chain, so the first stage still uses at most two GPUs at once.
  - No existing task (1207–1225) is edited, reordered or delayed.
  - Tasks 1104–1106 are untouched.
- **Later arms.** The same rule applies to D2 (`nostd_ema`, `awrbn_ema`) and D3 (`nostd_ema_irc`). When their gate
  scripts run, every passing arm gets its seeds 2–4 queued in the same way, after the then-current chain ends.
- **Cost.** About 1.5 h per run, so the six runs add about 4.5 h to each chain.
- **Evaluation.** The C1 evaluation tasks (`scripts/ad/c1_eval_tasks.py`) are written only after every entering
  arm's training is queued, with all entering arms. cw_conf stays unopened until then.
