# G (sampler-only reconciliation-hint guidance): dropped by its kill criteria

**Verdict.** The registered verdict, from `scripts/s5_dev.py` → `results/s5/s5_dev_summary.json` (criteria fixed before training, `C5G_DEV.md` §4), is **dropped**. Against recovery-start GRPO (RSC, seeds 0–1):

| Criterion | Rule | G | Met? |
|---|---|---|---|
| `s3_rec_dev` VSC difference | at least +0.03 | **−0.11** | no: Kill 1 |
| `proc_dev` VSC fall | at most 0.03 | **−0.033** | no: Kill 2, by 0.003 |
| KL at step 100 vs RSC | at most 10× | 1.38× | yes |

Seed 1 ran on Slurm after its HF job exited without training (bundle bug; `docs/stage2/DECISIONS.md`).

**What can be said** (descriptive, not tested):
- The hint fired in only about 5% of training rollouts (seed 0: 0.05; seed 1: 0.054). It needs a `list_payments` result that shows a partial payment, which the policy rarely produces in training.
- So the guided signal was small. The re-scored hinted episodes are off-policy for the un-hinted context, which is a deliberate bias inherited from PSP.
- This run cannot separate "too little guidance" from "biased guidance hurts". No further G variant is run under this registration.
