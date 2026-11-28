# AD-C2 addendum 1: the scope of P1's theoretical basis (written before any AD-C2 output)

Written 2026-11-28 18:58 IST. At this time no AD-C2 checkpoint, evaluation or analysis output exists (task 2000 at
step 78 of 200; `runs/ad_c2_*/final` and `results/ad/c2/` absent on sharanga). Nothing in
`PREREGISTRATION_AD_C2.md`, the hashed code or the predictions is changed; this addendum only states a limit of the
basis given for P1, found in an audit of the theory paper.

## The limit

P1 ("GRPO at c = 0.01 stays at the no-read corner") cites Corollary 3.2 (the stake-free vote). Corollary 3.2 is exact
only for **deterministic within-state returns** (`docs/theory/PROOF_AND_ASSUMPTIONS.md` §3, conditions (a)–(d)). In the
language agent, repair success after a read is rollout randomness, so returns within a faulted state are not
deterministic. For that case the theory's own exact enumeration (`results/theory/scan.json`, `threshold_exactness`,
G = 8, π = 0.3, s = 0.6) gives a GRPO threshold that **does** depend on the cost: p* = 0.654, 0.667, 0.706 at
c = 0.01, 0.05, 0.2. Lowering c from 0.05 to 0.01 lowers GRPO's read threshold by about 0.013, against a far larger
shift for the centred estimator (p* = c/s: 0.083 → 0.017).

## Consequence for reading AD-C2

- P1 is a prediction of **approximate** cost-blindness: the theory predicts at most a small GRPO response to the cost
  change, in the direction of more reading at lower c. It does not predict exact invariance in this regime.
- If GRPO at c = 0.01 reads (P1 fails), that is consistent with the stochastic-repair extension only if the effect is
  small relative to the std-free response (D2 against D3); a large GRPO response would contradict the mechanism's
  account of the language agent.
- Clean-state groups, which dominate the vote in AD-C1 (60% clean; most clean groups vary only through reads), have
  deterministic within-state returns (β_clean = −c), so the clean-state vote is exactly cost-blind for c ≫ ε/σ̂.
- The verdicts are still computed exactly as registered; this addendum changes how the paper describes P1's basis.
