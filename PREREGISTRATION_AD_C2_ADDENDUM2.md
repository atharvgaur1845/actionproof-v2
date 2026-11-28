# AD-C2 addendum 2: runner fix (written 2026-11-29 01:17 IST, before any AD-C2 outcome was read)

**What happened.** Task `slurm/tasks/2003_c2_eval_stats.sh` ran on worker 474160 (H200) from 01:07 to 01:15 IST and
exited with code 1 after the first of the five registered evaluations (`ad_grpo_s0` on `cw_c2`) had finished and
written its files. The evaluation itself succeeded. The failure is in the runner: the helper `ev()` ends with
`grep ... | head -1`; `head` closes the pipe after one line, `grep` receives SIGPIPE, and under the task's
`set -o pipefail` the helper returns non-zero, so the loop exits.

**What changes.** Nothing in the registered design, models, split, metrics or analysis. Task 2003 is frozen by
`scripts/ad/freeze_c2.py` and is left byte-identical. A new task, `slurm/tasks/2004_c2_eval_stats_rerun.sh`, is a
copy of 2003 whose only change is `|| true` after that display line (plus its header comment and no `# after:`
line; the training tasks are done).

**Evaluate once.** The completed `ad_grpo_s0` evaluation is not repeated: the task skips any model whose summary
file exists, so each model is still evaluated once on the sealed split. No AD-C2 result file was opened before this
addendum was written; the analysis (`scripts/ad/c2_stats.py`) has not run.
