# C5 (verified-progress turn credit): dropped by its kill criterion, with a diagnosis

**Verdict.** The registered verdict, from `scripts/s5_dev.py` → `results/s5/s5_dev_summary.json` (criteria fixed before training, `C5G_DEV.md` §4), is **dropped**. Against recovery-start GRPO (RSC, seeds 0–1):

| Criterion | Rule | C5 | Met? |
|---|---|---|---|
| `s3_rec_dev` VSC difference | at least +0.03 | **−0.32** | no: Kill 1 |
| `proc_dev` VSC fall | at most 0.03 | **−0.37** | no: Kill 2 |
| KL at step 100 vs RSC | at most 10× | 1.36× | yes |

**Development behaviour** (`results/s5/dev/*.summary.json`, `results/s3/dev/*`):

| Model | `proc_dev` VSC | False success | Reads per episode |
|---|---|---|---|
| C5 seed 0 | 0.487 | 0.513 | 0.013 |
| C5 seed 1 | 0.493 | 0.507 | 0.020 |
| RSC seed 0 | 0.860 | 0.020 | 3.187 |
| RSC seed 1 | 0.860 | 0.020 | 3.193 |

**C5 un-learned read-back.**

## Diagnosis: hypothesis failure, not an implementation bug (exploratory, post hoc)

**The mechanism.** C5's turn credit is l_k = Σ_{j≥k} α^{j−k} u_j, with u_k = Φ(after turn k) − Φ(before turn k), where Φ is verified progress on the authoritative state.
- A **read changes no state**, so u = 0 and l_read = α·l_{next}: always below the credit of the write that follows it.
- After z-normalising over all turns of the group, read turns therefore sit **below the group mean by construction**, whatever information they carry.
- The ω-weighted term pushes read tokens down, and the same holds for the final report.

**The data** (`scripts/c5_diagnose.py` → `results/s5/c5_diagnosis.json`; C5's own α = 0.5 and normalisation, recomputed on the recorded training rollouts):

| Training steps | Seed | Read turns | Write turns | Report turns | Reads per episode |
|---|---|---|---|---|---|
| 0–49 | s0 | −0.63 | +0.51 | −1.16 | 0.59 |
| 150–199 | s0 | −0.54 | +0.42 | −1.11 | 0.44 |
| 0–49 | s1 | −0.56 | +0.49 | −1.13 | 0.78 |
| 150–199 | s1 | −0.51 | +0.45 | −1.17 | 0.35 |

Read turns receive negative normalised credit in every block of both seeds, and reading in training rollouts falls by roughly half.

**Caveat.** About half the logged episodes could not be aligned turn-to-call (agent-call count ≠ number of progress entries, mostly recovery starts with scripted prefixes) and were excluded. The direction is the same in both seeds.

## What it means

**Potential-based turn credit keyed on state progress penalises information gathering in a partially observable write setting.** Reading back is exactly the action whose value is informational, not state-changing. This follows from the credit's form; it is not specific to ProcureSim.

**A repair must credit an evidence action by the value of the information it provides.** For example:
- the change in the policy's correct-next-action probability;
- the difference in outcome between the matched read and no-read continuations.

It must not credit evidence by state change. The research mandate noted that a matched-world value-of-information estimator equals GRPO in expectation (`research_mandate/REPORT.md`). So a VoI credit needs a design that changes the gradient's variance or bias, not just a reweighting.

**No further C5 variant is run under this registration.**
