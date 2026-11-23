# V2-TB7: decomposition of igc's gain (registration)

Registered 2026-11-23, before any TB7 output. The time is recorded by `scripts/ad2/freeze.py` in
`results/ad2/hashes_V2TB7_IGC_ABLATION.json`.

**Seen.** All TB1–TB6 outputs, including TB2's igc and nostd results.

## 1. Why

Code review after TB2 found an implementation property of the tabular igc. Re-verification after a repair (decision
V2) is labelled uninformative, because the replay rows hold the rollout's own repair-success bit fixed. The agent
cannot observe that bit, so such a re-read can carry information about whether the repair held.

In TB2's workflows the true value of that re-read is ≥ −c and close to −c. Re-verifying after a repair can only save
later verification costs, and those savings are small. The mislabel's direction is therefore benign, but its share
of igc's gain is unknown.

In the LLM simulator repair outcomes are deterministic given the world and the calls, so the issue does not arise
there (`actionproof/ad/readinfo.py`).

## 2. Variants

All variants use TB2's 12 settings, budgets ×1 and ×4, and 5 seeds; everything else is as TB2.

| Variant | Gate applied to |
|---|---|
| igc_full | V1, V2 and V2c: the registered igc, re-run |
| igc_v1 | V1 only (the first read after a write, e.g. of the never-faulted op) |
| igc_v2 | V2 and V2c only |

The reference is TB2's nostd at ×1 and ×4.

## 3. Prediction

**P7a.** igc_v1 retains ≥ 80% of igc_full's regret improvement over nostd, at both ×1 and ×4.

If P7a fails, igc's TB2 advantage depends materially on the re-verification gate, which is partly an artefact of the
testbed. The igc claims in SELECTIVE_VERIFICATION_RESULTS and FINAL_ALGORITHM_REPORT must then be scoped to the V1
share.

FILES:
- scripts/ad2/tb7_igc_ablation.py
- scripts/ad2/tb7_analyze.py
- actionproof/ad2/tabular/env.py
- actionproof/ad2/tabular/policy.py
- actionproof/ad2/tabular/train.py
- actionproof/ad2/tabular/envs.py
