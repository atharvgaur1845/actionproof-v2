# V2 deviations log

Times come from `date` on the machine named.

## DV-1. V2TB_PROTOCOL header time (2026-11-23 22:09 IST, local)

The header of `V2TB_PROTOCOL.md` says "Registered 2026-11-23 23:05 IST". That time was typed, not read from a
clock, and it is wrong. The true registration time is 2026-11-23 22:08 IST, written by `scripts/ad2/freeze.py`
into `results/ad2/hashes_V2TB_PROTOCOL.json`; the commit time agrees.

The document is left unedited so that its hash stays valid. No testbed output existed at 22:08.

## DV-2. V2-E1 (LLM condition B) suspended; task 1300 failed without training (2026-11-23 22:08 IST, sharanga)

**Failure.** Task 1300 (`ad2_grpo_rs_s0`) failed in its first second with `unknown --groups 1248200074`.
`slurm/ad2_arm.sh` assigns to `GROUPS`. That is a bash builtin array (the user's group ids), and assignments to it
are ignored, so the user's numeric group id was passed instead of "resampled". No training step ran and no output
exists.

**Suspension.** At 22:0x IST the main session reprioritised the final days of the hackathon. The A100 and every
free H200 lane are reserved for the main session's practical-agent study (tasks 1400+), and no new v2 GPU tasks
may be queued. In response:
- tasks 1301–1303 were moved from `queue/a100/pending` to `queue/a100/hold` at 22:08:52 IST;
- 1300 stays in `queue/a100/failed`.

**Status.** V2-E1 is suspended, not cancelled. Its registration and hashes are unchanged. If it is ever run:
- the variable must be renamed in `slurm/ad2_arm.sh`. That file is hashed by V2E1_CONDITION_B, so the fix needs
  an E1 amendment and re-hash before any output;
- the condition-B question is answered on the CPU instead by TB1 and TB6 (V2TB_PROTOCOL.md H1c), which were
  registered before any output.

**Side effect.** 1300's failure counts once toward the A100 worker's two-consecutive-failure pause
(`slurm/worker.sh`). Any successful task resets the count.

## DV-3. Falsy-default bug in a frozen analysis line (2026-11-23 22:45 IST, local; found when TB1 was first analysed)

**Bug.** `scripts/ad2/tab_analyze.py` line 74 evaluates `(value or 1) <= 0.2`. A measured value of exactly 0.0 is
falsy, so it is replaced by 1 and the hypothesis is reported false.

**Effect.** TB1 gives awrbn's mirror-cell verify fraction as 0.0 over 18 cells, which satisfies H1g_awrbn (≤ 0.2).
The summary file still records `false`. We report the hypothesis as **supported (0.0 ≤ 0.2)** on the raw value,
which is stored in `TB1_summary.json` → `per_est.awrbn.mirror_cells_verify_frac`.

**Checks.** No other registered test is affected. The other `or 1` defaults (H1b nostd 0.057, H1c 0.315, H1f) and
the `or 0` defaults (TB2 G2/G3: a 0.0 value maps to 0) are checked by hand.

**Script.** The frozen script is not edited, so its hash stays valid.
