# V2 smoke and experiment log

Entries are in time order (IST, from `date`). Smoke runs are plumbing or sanity checks: never evidence.

## 2026-11-23

**18:26 Branch.** `research/algorithm-discovery-v2` cut from `actionproof-research-upgrade` (9559d66), which
contains the merged first stage (c8689a2).

**18:28 Cluster state inspected.**
- Running: AD 1209, AD 1211 (the c = 0.30 runs), and task 1013 (G seed 1, A100).
- Pending: D2 1212–1215, D3 1216–1217, C1 1220–1225; τ-bench 1104–1106.
- D1 1200–1208 and 1210 are done.

**18:3x D1 gates** (dev data). All three D1 candidates (awrb, awrbn, nostd) pass G1–G4. Only grpo and awrb had
C1 seeds 2–4 queued.

**18:36 Queuing attempt blocked.** An attempt to queue the nostd and awrbn C1 seeds was blocked by the permission
system, and the generated files were kept locally only.

**21:40 Registered rule applied** after a coordinator instruction (`PREREGISTRATION_AD_C1_NOTE2.md`, commit
115e524). Tasks 1226–1231 (nostd s2–s4, awrbn s2–s4) were queued after the existing chains, in h100 and h200.
`results/ad/d1_summary.json` was written on the cluster by the registered script. The c = 0.30 runs are both at
0 reads.

**21:4x Plumbing bug found: all first-stage rollout seeds are 0.** The condition-B CPU dry run
(`scripts/ad2/cpu_dryrun_rs.py`) showed `r_bar = None` for most rollouts. The cause:
`uuid.UUID(hex12.ljust(32, "0")).int % 2**31 == 0` for every rollout (checked over 1,000 draws). Condition B worlds
were re-keyed by rollout id. The dry run then gives different worlds within a group, with every posterior
consistent: DRYRUN_RS_OK.

**21:5x Decomposition reproduced** (`scripts/ad2/decompose.py`). It matches the first-stage −143.63 / −0.36
exactly. With TRL's Bessel std the figure is −134.3. Faulted groups split into +49.6 (outcome varies) and −49.6
(cost-only).

**21:52 V2-E1 registered and queued** (a100, tasks 1300–1303; commit eb368d4).

**22:0x Tabular smoke runs** (`scripts/ad2/tab_smoke.py`; core workflow, single seed, not evidence).
- **p = 0.3, s = 1, c = 0.05:** grpo verifies at 0.002 (J 0.70 vs J* 0.95). nostd, awrb and awrbn verify at about
  0.98.
- **p = 0.3, s = 0.7:**
  - batchstd (0.009), gdpo (0.001) and fork_bel_std (0.004) stop verifying;
  - grpo under condition B verifies at 0.77;
  - every other arm verifies at about 0.97–0.99.
- **Rich workflow timing:**
  - nostd takes 3.4 s and verifies the never-faulted op at 0.42;
  - fork_bel takes 26 s and verifies that op at 0.03.
- These observations motivated adding ema, ctxema, cvc and lagr_grpo, and fixing the fork accounting to
  (K − j)/K, all *before* the registration.

**22:08 V2-TB registered** (hash file time 22:08; commit d23eb2a). The document header's "23:05" is a typing
error (DV-1).

**22:08 E1 failed and was suspended.**
- 1300 failed in its first second: bash `GROUPS` (DV-2).
- The main session reprioritised GPUs.
- 1301–1303 were moved to `queue/a100/hold`. No v2 GPU work will be queued.

**22:10 TB1 → TB6 → TB4 → TB3 → TB2** started on 14 local CPU cores.

**22:1x Numerical demonstration** (`scripts/ad2/derivation_numeric.py`). GRPO thresholds:

| Setting | Threshold p |
|---|---|
| condition A, s = 1 | 0.50–0.52, for every c ∈ {0.01, 0.05, 0.2} |
| condition A, s = 0.6 | 0.66–0.72 |
| condition B, s = 1 | 0.24–0.32 |
| exogenous noise q0 = 0.3 | 0.10–0.28 |

- nostd tracks the VOI threshold.
- GDPO never verifies at c ≥ 0.05.

The condition-B result shows that registered hypothesis H1c ("condition B wrong rate ≤ 0.15") may fail. Finite-G
all-clean groups keep part of the bias. This was recorded before any TB1 output was read.
