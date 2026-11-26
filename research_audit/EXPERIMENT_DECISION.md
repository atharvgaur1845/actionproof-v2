# Experiment value / time matrix and keep-drop list (Phase B)

2026-11-27 ~07:30 IST.

**Time available.** The conservative freeze is Nov 28, 23:59 IST: about 40 h. The author must record the video,
so slides and demo must be final by about Nov 28, 12:00 IST. An experiment therefore has to finish, *including
statistics and the write-up*, by about Nov 28, 06:00 IST.

**Compute.** Two worker lanes are idle and alive: `ap_w_a100b` (about 19 h of wall time left) and `ap_w_h100b`
(about 43 h).

**Measured costs.**

| Step | Time |
|---|---|
| Dw_aug SFT, 2 epochs, about 6,000 rows | 9,237 s (2.6 h) |
| GRPO, 200 steps | 1.5–3 h |
| TXN evaluation, one model, 440 × 3 episodes | about 10 min |
| One P1-test split | 3–6 min |

## Candidates

| | A: GRPO from Dw_aug | B: selective escalation (SEL1) | C: transaction-aware runtime gateway |
|---|---|---|---|
| Question | Does P4's warm-start generalisation survive RL? | Can demonstrations teach the agent to hand off only when evidence is irreducibly missing? | Can a state-aware guard block unsafe writes without blocking repairs? |
| Gap it closes | "GRPO was not run on top of Dw_aug" (a scope caveat) | **H5**: D escalates in 15/540; Dw_aug in 0/540 and unsafe in 53/540. This is the headline limitation, and Escalation-Safe Aya's core result. | P2's guard blocked the repair |
| Judge-visible value | Medium: a technical follow-up | **High**: it turns slide 5's main failure into a measured result either way | Medium: engineering; scripted only |
| Scientific novelty | Low (a known recipe) | Medium: prior work on hidden-commit escalation is evaluation-only (`NOVELTY_POSITIONING_FINAL.md` §4) | Low: CASEL, Verified Tool Calls |
| Clean control | Yes (D shares the recipe) | Yes: a matched-volume control arm from the same Dw_aug parents | Yes (scripted) |
| Seeds | 3 | 3 per arm, paired by parent | n/a |
| Fresh held-out test | Needs a fresh surface and workflow: new code in the stage-4 surface machinery | Needs a fresh sealed split. TXN already has `unavailable_reads` and `escalate_to_human`; transient outage is a 20-line env subclass | Not applicable |
| GPU wall time | 3 × (2 h GRPO + 1 h eval) on 2 lanes: about 5 h | 6 × (0.8 h SFT + 0.6 h eval) on 2 lanes: **about 4–5 h** | 0 (CPU on the cluster) |
| Implementation risk | Low | Medium: a demo renderer, a policy, a split generator, the eval wiring | Medium |
| Risk of an uninformative null | Medium | Low: either outcome (escalation learned, or a shortcut that over-escalates) is reportable and decision-relevant | Medium |
| Fits the deadline | Yes | Yes, if launched by about 13:00 IST today | Yes |

**Decision: B (SEL1) is the single primary experiment.**

**A is dropped.** It answers a scope question that the README already states honestly. It needs a fresh renamed
surface to be confirmatory. It cannot address H5, because the ProcureSim GRPO pools contain no unresolvable regime.

**C is dropped.** P2 already reports the naive guard. A gateway result would be scripted-only, and CASEL and
Verified Tool Calls cover the design.

**Gate.** The experiment is launched only if all of these hold before 13:00 IST:
1. `PREREGISTRATION_SEL1.md` is committed;
2. the sealed split is hashed;
3. the demonstration data are built and hashed;
4. the leakage and disjointness assertions pass;
5. a CPU smoke test of the policy and env passes.

If the gate is missed, SEL1 is not run, and the H5 gap is reported with an always-escalate reference.

## Keep / drop list for the remaining sprint

| Item | Decision | Why |
|---|---|---|
| SEL1 selective escalation | **keep (primary)** | above |
| A: GRPO from Dw_aug | drop | above |
| C: runtime gateway | drop | above |
| New RL optimizer or credit method | drop | the novelty audit; abandoned claims |
| More external (non-Cohere) baselines | drop | supplementary only by rule; adds no controlled comparison |
| More interpretability (CircuitKIT retry) | drop | speculative; P2 priority in the prompt rubric |
| Rendered six-slide deck | **keep (P0)** | the final round is judged on the video or slides |
| README restructure (judge path; history moved out) | **keep (P0)** | |
| Stale-surface banners | **keep (P0)** | |
| Paper status: companion theory paper | **keep (P0)** | one thesis |
| Live Space judge mode and replay fallback | **keep (P1)** | our structural lead; must be legible |
| Small no-GPU smoke check, plus the full route | **keep (P1)** | matches competitors 02, 06 and 08 |
| `release/ARTIFACT_MANIFEST.json` with Hub revisions | **keep (P1)** | |
| Impact graphic and assumption table | **keep (P1)** | Impact is our lowest column |
| Clean-clone `verify_numbers.sh` on the cluster | **keep (P0, Phase F)** | |
| Three hostile reviews | **keep (Phase F)** | |
| Video recording | author (narrator); Claude prepares the run of show, captions and the deck | |
