# Amendment 12, addendum 2: protocol for seeds 1–4, confirmatory machinery and reporting rules

Registered 2026-11-20 (commit time of this file and `results/a12c_hashes.json`), **before any GRPO run of seeds 1–4**. Nothing in `PREREGISTRATION_A12.md` or addendum 1 is edited; this file adds rules for what follows.

## What was known when this was written (disclosure)

- Seed 0 (tasks 614 R_V, 615 R_A) finished on HF Jobs. The registered stop rule (A12 §5: stop if more than 80% of groups have zero reward variance averaged over steps 40–50) **fired for R_A seed 0** (0.864; R_V 0.545). The run continued to step 200 as logged deviation D-A12-1 (`results/s2/stop_rule_seed0.json`, `docs/stage2/DECISIONS.md`).
- Seed 0 development diagnostics were seen (`results/eval/a12/dev/`, `proc_dev` and `proc_val` only): phantom read-back R_V 142/142 (dev) and 52/52 (val) at steps 50–200; R_A 0/96 (dev) and 0/40 (val) at steps 50–200. VSC R_V 0.795–0.845 (dev); R_A 0.495 (dev). No confirmatory split has been evaluated by any model.
- An audit (2026-11-20) found: (a) R_A pays a blind same-key retry (the idempotency store replays a stored success), so after a lost acknowledgement R_A rewards retrying without reading; (b) at step 0, R_A has almost no reward variation on phantom tasks, and most of its informative groups come from lost-ack and never-commit tasks; (c) the registered H3 test assigns opposite predicted signs to classes the agent cannot tell apart before reading (all success acknowledgements look alike; all timeouts look alike), so it may fail by construction. None of these change the registered rewards, data or tests.

## 1. Protocol labels for seeds

- **Seed 0 runs under the original protocol** and is reported as "stop rule fired for R_A; continued to step 200 as deviation D-A12-1". Its R_A results are labelled as a deviation wherever shown.
- **Seeds 1–4 run under this addendum.** The stop rule is kept as a diagnostic (computed and reported for every run) but does not stop an arm whose mean reward over steps 40–50 is ≥ 0.9 (saturation). An arm with mean reward < 0.9 and zero-variance share > 0.80 is still stopped and reported as uninformative.
- Training code, data, hyperparameters and evaluated checkpoint are identical for every seed. Results are reported **per seed with the protocol label**; any across-seed summary lists which seeds are under which protocol and is never presented as identical replications without that label.
- **Evaluated checkpoint:** step 200 for both arms. A run that does not reach step 200 is reported as missing; no other checkpoint substitutes for it.

## 2. Confirmatory machinery (must exist before the first confirmatory run)

- `results/a12/frozen_models.json`: for every seed and arm, the step-200 adapter path and its sha256, committed before any confirmatory run.
- One confirmatory task per frozen model runs `scripts/eval_proc.py --confirmatory` on `proc_conf`, `proc_combo`, `proc_stress` and writes `results/eval/a12/conf/a12_<arm>_s<seed>__neutral__<split>.jsonl` (the paths `scripts/s2_stats.py` reads).
- The access log is keyed by adapter sha256 and split; it is uploaded to the private HF dataset and checked from there before every confirmatory run on any executor. A second run of the same (sha256, split) is refused. A crashed run resumes from its saved per-task output; it is never restarted.

## 3. Reporting rules

- **H2 (read-back) and H3 (mechanism) are separate results.** H3 is reported exactly as registered (predictions `results/a12/h3_predictions.json` and `results/a12b/h3_predictions.json`, criterion A12 §7), including failure. An exploratory H3 analysis pooled by acknowledgement type (success-acknowledged classes; timeout classes) may explain the outcome but never replaces it. The registered composite claim C1 (A12 §9) is reported as registered.
- **Sensitivity analyses of the primary endpoint** (labelled as such): (a) task-level phantom read-back: share of tasks with a phantom write in which the agent read back after it at least once; (b) phantom read-back stratified by step (pay, order, receive), because benign and phantom attempts differ in step mix.
- **Reward-signal analysis** (exploratory, from the training rollout logs): per fault class and training step, the reward distribution under each arm, the share of zero-variance groups, the mean reward difference between samples that read back and samples that did not, and each class's share of informative groups. It is compared with the per-class change in read-back. A further training control (K1) is run only if this analysis leaves a specific competing explanation open, and then under its own registration.
- **Statistics** (`scripts/s2_stats.py`, as frozen here): per-seed paired task bootstrap and exact McNemar; between-seed sample SD; exact sign-flip over seeds reported one-sided and two-sided (smallest attainable: 1/8 and 2/8 at 3 seeds, 1/32 and 2/32 at 5); no seed-level significance is claimed at 3 seeds. The Holm step was corrected (it previously multiplied the wrong p) and the descriptive hierarchical bootstrap now shares task draws across seeds; a metric with no denominator is missing, never 0. These fixes precede any confirmatory output.

Unchanged: rewards, reward-truth table, training split and settings, seeds and their time rule, confirmatory splits, endpoints, claim criteria.
