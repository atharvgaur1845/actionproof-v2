# P1 decomposition: where the 0.513 → 0.772 / 0.819 gain comes from (addendum §2A)

2026-11-25. Recomputed on Slurm from the frozen-test episode files (`results/p1/test`, `results/p2/test`, `results/p2/ref`):
- `scripts/audit/p1_decomposition.py` (job 469876) → `results/audit/p1_decomposition.json`;
- the cell-clustered intervals: `scripts/audit/redteam_recompute.py` (job 469877) → `results/audit/redteam_recompute.json` key `p1`.

Every model and arm here reproduces its registered value exactly:
- A 0.513, B 0.483, C 0.772, D 0.819, E 0.782;
- W 0.335, M 0.479, O 0.924;
- P1 H1–H3 and P2 H1–H2 deltas and intervals.

Compare `contrasts` with `results/p1/p1_summary.json` `test` and `results/p2/p2_summary.json` `primary`, `references`.

**Every contrast below not marked registered is EXPLORATORY.** All were computed after all P1/P2 outputs were known, and none is a new claim of support.

## 1. What each arm is (from `PREREGISTRATION_P1.md` §4 and `PREREGISTRATION_P2.md` §3)

| Arm | Initialisation | Recovery demonstrations (oracle, verified, 554 episodes) | GRPO (R_V, 200 steps) | Recovery starts in GRPO pool |
|---|---|---|---|---|
| W | `rft_proc` warm start | no | no | — |
| M | W | no | yes | no (class-matched full episodes; S3 addendum 1) |
| A = RSC | W | no | yes | yes (50%) |
| B | A | no | — | — (adds a reconciliation line at inference) |
| C | A, merged | **yes, after RL** (1 epoch, lr 5e-5, recovery rows only) | before the SFT | yes |
| D | `p1_dwarm` = original + recovery demos, `rft_proc` settings (2 epochs, lr 1e-4) | **yes, in the warm start** | **after** (RSC recipe unchanged) | yes |
| E | D's warm-start rows in ledger format | yes, ledger format | after, with ledger rollouts | yes |
| O | scripted oracle (greedy) | — | — | — |

**C and D are not nested.** C is not D's warm start: C = RSC followed by recovery SFT, while D = recovery SFT data in the warm start followed by RSC GRPO. **Reading 0.772 → 0.819 as "what GRPO adds to recovery SFT" is wrong.** That reading appears in the user's pasted summary. README and write-up list the two numbers side by side, without this note.

## 2. The comparison on identical frozen tasks (450 tasks, sampled VSC, T = 1, 4 samples; arm = mean of 3 seeds)

| Arm | Union | full (120) | pp, unseen ratios (90) | rec (120) | wf4, unseen workflow (60) | combo (60) | Per seed (union) | Greedy union |
|---|---|---|---|---|---|---|---|---|
| W warm start | 0.335 | 0.583 | 0.017 | 0.421 | 0.183 | 0.296 | 0.335 (1 model) | 0.302 |
| M class-matched control | 0.479 | 0.702 | 0.240 | 0.539 | 0.207 | 0.544 | 0.518, 0.479, 0.440 | 0.563 |
| A recovery-start GRPO | 0.513 | 0.706 | 0.319 | 0.624 | 0.174 | 0.535 | 0.530, 0.513, 0.495 | 0.666 |
| B A + reconciliation line | 0.483 | 0.692 | 0.206 | 0.615 | 0.183 | 0.517 | 0.490, 0.483, 0.476 | 0.528 |
| C A + recovery SFT | 0.772 | 0.806 | 0.860 | 0.846 | 0.397 | 0.799 | 0.768, 0.764, 0.784 | 0.856 |
| **D recovery-SFT warm start + GRPO** | **0.819** | 0.810 | 0.931 | 0.909 | **0.486** | 0.824 | 0.823, 0.806, 0.829 | 0.888 |
| E D in ledger format | 0.782 | 0.824 | 0.931 | 0.910 | 0.264 | 0.733 | 0.779, 0.778, 0.788 | 0.781 |
| O scripted oracle (greedy only) | — | 0.842 | 1.000 | 0.933 | 1.000 | 0.883 | — | 0.924 |

Source: `results/audit/p1_decomposition.json` `arms.*.sampled.{union,per_split,per_seed_union}` and `arms.*.greedy.union`.

**Ceiling.** 27 of the 450 tasks are never-commit tasks:
- 19 full-episode tasks;
- 8 recovery starts.

On these, VSC is 0 for every arm, the oracle included (`by_fault_class.never_commit`, `s3:never_commit:*`). The correct outcome there is an honest failure report, which VSC does not credit. The maximum attainable union VSC is therefore 423/450 = 0.94.

## 3. Contrasts (paired task bootstrap, B = 10,000; cell-clustered interval: 116 cells, B = 2,000)

| Contrast | Registered? | Δ | 95% task CI | 95% cell CI | Same-index seed pairs | Positive of 9 cross-seed pairs | Isolates |
|---|---|---|---|---|---|---|---|
| A − W | no | +0.178 | [0.154, 0.203] | — | — | — | RSC GRPO (RL with recovery starts and extra fault coverage) |
| M − W | no | +0.144 | [0.121, 0.167] | — | — | — | full-episode GRPO on the same fault classes |
| **A − M** | no | **+0.034** | [0.021, 0.046] | [0.012, 0.054] | +0.012, +0.034, +0.055 | 7 | **start states, fault coverage held fixed**, on fresh tasks |
| B − A | P1-H1 | −0.030 | [−0.041, −0.019] | — | −0.040, −0.031, −0.019 | 0 | inference-time reconciliation line |
| C − A | P1-H2 | +0.259 | [0.232, 0.287] | [0.169, 0.343] | +0.238, +0.251, +0.289 | 9 | recovery SFT on top of A |
| D − A | P1-H3 | +0.306 | [0.276, 0.338] | [0.203, 0.396] | +0.293, +0.292, +0.334 | 9 | recovery demos in the warm start, then the same GRPO |
| **D − C** | **no (never registered)** | **+0.047** | [0.030, 0.065] | [0.024, 0.071] | +0.056, +0.042, +0.045 | 9 | **bundled**: SFT before vs after RL; data mix (original + recovery vs recovery only); SFT settings; C starts from RSC weights |
| E − A | P2-H2 | +0.269 | [0.237, 0.301] | [0.162, 0.361] | +0.249, +0.265, +0.293 | 9 | ledger-format D |
| E − D | P2-H1 | −0.038 | [−0.054, −0.022] | — | −0.044, −0.027, −0.041 | 0 | ledger format |
| C − M, D − M | no | +0.293, +0.340 | [0.264, 0.322], [0.309, 0.372] | — | — | 9, 9 | — |
| O − D (greedy oracle vs sampled D) | no | +0.036 | [0.021, 0.053] | — | — | — | mixed decoding; greedy D is 0.888, which leaves 0.036 to the oracle as well |

Sources:
- `results/audit/p1_decomposition.json` keys `contrasts`, `seed_pairs`;
- `results/audit/redteam_recompute.json` key `p1`, field `cell_ci95`.

Cell = (fault class, workflow, first faulted tool, start type). With five or more seeds, the cell-clustered intervals are wider than the task intervals but exclude 0 for every positive contrast above.

### Per split, the exploratory D − C (`per_split_contrasts.D-C`)

| full | pp | rec | wf4 | combo |
|---|---|---|---|---|
| +0.005 [−0.015, 0.024] | +0.070 [0.034, 0.106] | +0.063 [0.035, 0.094] | +0.089 [0.007, 0.171] | +0.025 [−0.018, 0.069] |

### Per split, A − M (`per_split_contrasts.A-M`)

| full | pp | rec | wf4 | combo |
|---|---|---|---|---|
| +0.004 [−0.010, 0.019] | +0.079 [0.049, 0.108] | +0.085 [0.054, 0.117] | **−0.033 [−0.056, −0.011]** | −0.010 [−0.035, 0.015] |

This matches the registered S3 pattern:
- the start-state effect appears on recovery starts and partial payments (S3-A2 H3′ supported);
- it does not appear on full episodes or under an unseen workflow (H4′p not supported; S4-EVAL G3 not supported).

### Where the gain sits, by fault class (`by_fault_class`; single-fault, full episodes)

| Class (n) | W | M | A | C | D | E |
|---|---|---|---|---|---|---|
| partial (103) | 0.027 | 0.265 | 0.343 | 0.867 | 0.935 | 0.939 |
| phantom (21) | 0.274 | 0.766 | 0.730 | 0.952 | 0.913 | 0.952 |
| lost_ack (17) | 0.882 | 0.951 | 0.946 | 0.980 | 0.990 | 1.000 |
| precommit_timeout (19) | 0.908 | 0.860 | 0.873 | 0.925 | 0.930 | 0.947 |
| clean (31) | 0.992 | 0.968 | 0.970 | 0.984 | 1.000 | 1.000 |

The recovery-demonstration gain is almost entirely partial-payment repair, plus duplicate delivery on recovery starts:
- `s3:dup_delivery:read`: A 0.686, C 0.776, D 0.981;
- every other class is near its ceiling for A already.

## 4. Attribution: what the evidence supports and what it cannot separate

| Factor | Evidence | Verdict |
|---|---|---|
| **Data: verified oracle recovery demonstrations** | C − A +0.259 and D − A +0.306 (registered). The two arms put the data in different places, and both gain about 0.26–0.31, concentrated in the partial class (0.34 → 0.87–0.94) | **Dominant, and established**: both arms containing the data gain, and no arm without it comes close (M, A, B ≤ 0.513) |
| **Start-state curriculum (RL data distribution)** | A − M +0.034, 7/9 seed pairs, on fresh tasks. Located on pp/rec, negative on wf4 | **Small, positive, exploratory here.** Consistent with registered S3-A2 H3′ |
| **Inference-time hint** | B − A −0.030 (registered H1 not supported) | **Rules out** "the policy only lacks the arithmetic". Visible values do not substitute for demonstrations |
| **Order and initialisation (SFT before RL vs after)** | D − C +0.047, 9/9 cross-seed pairs, cell CI [0.024, 0.071]; not registered | **Suggestive.** Bundled with the data mix and the SFT settings; cannot be attributed to order alone |
| **GRPO's contribution on top of recovery SFT** | **Not measured.** D's warm start `p1_dwarm` was never evaluated, on dev or test (`results/p1/dev` has only A–D). D's GRPO stage had zero reward variance in 0.78–0.82 of groups (`results/audit/redteam_recompute.json` `train_logs.p1_d_s*`) | **Unknown.** It could be anywhere from about 0 to most of D − (D's warm start) |
| **Data × RL interaction** | Needs the 2 × 2 {recovery data in the warm start: no/yes} × {GRPO: no/yes} = {W, A, `p1_dwarm`, D}. Three of four cells are evaluated | **Not testable yet.** One evaluation completes it |

**Answer to "is P1 reproducible and statistically supported?"**
- **Reproducible from the stored episodes.** A second pathway gives identical means and intervals: this script, and `CLAIM_VERIFICATION.md` for the verifier.
- **Statistically supported** under the registered rule:
  - Holm 0.0003, every seed pair positive;
  - all 9 cross-seed pairs positive;
  - the cell-clustered intervals exclude 0.
- **Seed-to-seed spread is small**: C 0.764–0.784, D 0.806–0.829.
- **Limits.** One base model (Tiny Aya Global), one simulator, one demonstration teacher (the scripted oracle, which also sets the 0.924 ceiling).
- **Not independently reproducible from a clone today.** P1 adapters are on the cluster only (`scripts/publish_procure.py` does not list them). The P1/P2 traces (~760 MB) are unpublished (author decision).

## 5. The one experiment that would resolve the open cell

**Evaluate `runs/p1_dwarm/final` (and, optionally, `runs/p2_ewarm/final`) with the P1 evaluation (`slurm/p1_evalset.sh`, as used by the P3 tasks; greedy plus 4 samples at T = 1), on `p1_dev_*` and the five frozen test splits.**

- **Contrasts to fix before output** (descriptive, a dated note in the style of an addendum):
  - D − Dwarm: GRPO's contribution;
  - Dwarm − W: the recovery data alone;
  - (D − Dwarm) − (A − W): the interaction.
- **Report rule:** paired task bootstrap with seed-paired consistency. Dwarm is one model, so seed consistency is D's three seeds against one model.
- **Cost:** evaluation only, about 1 GPU-hour (5 episodes × 450 test + 180 dev tasks). No training.
- **Integrity:**
  - the test split has been used, but no choice depends on this output;
  - P3 evaluates on the same split and is already registered and frozen;
  - this adds a reference arm and changes no registered hypothesis.
- **Fits before Nov 28?** Yes, if a lane frees after P3 (tasks 1610–1613) and before the final freeze. It needs about 2 hours of analysis and write-up.

The main session owns the queue, so this audit does not launch it.

WHAT WE LEARNED → verified recovery demonstrations are the active ingredient of P1 (+0.26 to +0.31, partial repair 0.34 → 0.87–0.94). The start-state curriculum adds a small +0.034 on fresh tasks. Putting the demonstrations in the warm start beats adding them after RL by +0.047 (exploratory, bundled).
WHAT THIS RULES OUT → that an inference-time reconciliation hint is enough (−0.030). That the P1 gain is a seed artefact (all 9 cross-seed pairs, cell-clustered CI).
WHAT REMAINS UNCERTAIN → how much GRPO adds on top of recovery SFT (`p1_dwarm` never evaluated); any data × RL interaction; transfer beyond one simulator and one teacher.
NEXT EXPERIMENT → evaluate `p1_dwarm` on dev and the frozen test (§5), about 1 GPU-hour.
