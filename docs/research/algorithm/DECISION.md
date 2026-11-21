# Stage 4 algorithm decision (2026-11-21)

Inputs:
- `docs/research/NOVELTY_AUDIT.md` (84 newly verified entries);
- the trainer source (TRL 1.7.1, AgentTune);
- Stage 3 results.

## Decision matrix

Scores are 1 (poor) to 5 (strong). For novelty, a 1 means a published method already does it.

| Candidate | Novelty | Mathematical defensibility | Relevance to ActionProof | Feasibility | Expected gain | Generalisation | Cost |
|---|---|---|---|---|---|---|---|
| A: forked counterfactual action credit (FCA) | 1–2: BiPACE, RTMC, GiGPO, CRR, ReflectPO; SURE-Act forces action classes incl. an irreversible-commit class; COMA when weighted by π | 4: all-action / expected-gradient term with an exact importance weight from TRL's sequence-level correction | 4 | 4 (implemented, tested) | uncertain | low (needs per-domain action classes) | ≈ RSC |
| B: recovery-aware constrained GRPO | 1: Constrained GRPO, Safety as a Budget, SafeDojo, MO-GRPO, GDPO; TRL already has `normalize_then_sum` | 4 | 3 | 5 | low (the safety-first and Lagrangian pilots gave nothing) | — | low |
| C: adaptive recovery-frontier scheduling | 1–2: PLR, PAC, LILO, CeSoR, MoPPS, Depth Before Breadth, RAIL, TRACE | 3 | 3 | 3 | uncertain | — | moderate |

**No candidate survives the novelty audit as a new algorithm.** Neither does the Stage 3 recovery-start curriculum: Fission-GRPO (ACL 2026), R3L, failure-prefix conditioning, ReflectPO and Depth Before Breadth all restart GRPO from failure contexts.

## What is selected, and how it is described

- **FCA continues as an empirical test, not a contribution claim.** The question is whether action-class counterfactual credit at forked recovery states adds anything over recovery-start GRPO, when write acknowledgements are silent about commits. It is described as an *adaptation* of the all-action / counterfactual-baseline family (BiPACE, COMA, GiGPO) to simulator forks of stateful writes. The kill criteria in `FCA_GRPO.md` §7 stand. A positive result would be reported as "this known estimator helps in this setting", nothing more.
- **B is not pursued.** It is redundant, and its development analogue (the safety-first and Lagrangian pilots) showed no gain.
- **C is not pursued as a scheduler.** The audit identifies one testable, setting-specific question: does choosing start states or forks *from the hidden fault label* shift the policy's belief about hidden faults (the SAMPLR problem) toward over-verifying or over-escalating? Stage 3 already logs the needed quantities (reads and escalations on clean and complete starts). This is analysed from existing outputs as an exploratory secondary, not as a new method.

## The defensible contribution

The project's defensible contribution is not an optimiser. It is:
1. **The setting.** Silent write commits with byte-identical acknowledgements, partial and duplicate effects, and a verifier with hidden authoritative state. The audit found no training paper with silent write commits plus duplicate risk; Bench2Robust assumes retries cannot duplicate.
2. **The registered measurements in that setting.** Reward observability selects read-back (A12). Evidence use does not follow from reading (A14, A15 by workflow). Representation is present but the action margin changes (A16). Transfer is mostly absent (A17).
3. **A decomposition of a known curriculum's gain** into fault coverage and start state (S3 addendum 1), with its safety trade-off.
4. **An honest test of a known credit estimator** (FCA) in this setting.
