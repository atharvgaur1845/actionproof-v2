# Final review: simulated judging panel

2026-11-22 01:34 IST, branch `stage4`.

**Inputs:**
- the new website build (`demo/static/index.html`, not yet deployed);
- the six-slide deck (Slides artifact, private);
- `docs/final/*`;
- `docs/RESEARCH_RED_TEAM_REVIEW.md`;
- `docs/research/NOVELTY_MATRIX.md`.

## The ten judge questions

| # | Question | Can a judge answer it from the deliverables? | Evidence |
|---|---|---|---|
| 1 | ActionProof in one sentence | Yes: "Tool agents trained with RL learn to check what the reward checks, but checking is not yet knowing how to recover." | Hero, THESIS.md |
| 2 | The main failure mode | Yes. The hero is a real recorded phantom payment; section 01 shows ok next to "only part stored" | site 00–01, slides 1–2 |
| 3 | The strongest result from memory | Yes: 1.00 vs 0.00 read-back, five of five seeds | site 02, slide 3 |
| 4 | Exactly what changed in training | Yes. A12: only the reward's information source. Stage 3: where episodes start (plus one added fault type, disclosed) | slides 3 and 5 |
| 5 | Are the baselines fair? | Mostly. A12 R_A is an unmoved arm (disclosed). The Stage 3 recipe comparison adds a fault type, so the fair class-matched control is shown separately | FINAL_CLAIMS C2, C7–C9 |
| 6 | Are the statistics justified? | Mostly. Seed-level tests are reported, and task intervals are labelled as task-level. **Weak point:** the start-state effect has 3 control seeds, and the red-team's template-cluster interval includes 0 (+0.081 [−0.023, 0.209]) | FINAL_CLAIMS C9; red-team review §2 |
| 7 | Can you see what the model did differently? | Yes: recorded traces of five models on the same task (site 02–03, slide 4) | site, slides |
| 8 | Can you reproduce a result? | Yes for the numbers (`s2_numcheck.py`, result files, registrations). Training needs the cluster; the Stage 2–3 adapters are **not publicly released yet** | site 05 |
| 9 | Does the public product show the actual research? | Yes: the deployed Space shows the procurement study (real simulator, recorded replays, registered results, inside-the-model) | public site |
| 10 | The remaining limitation | Yes: surface binding (renamed tools, a new workflow step), negative transfer, three control seeds, and selection rather than invention | slide 5, site 04 |

## What a hostile judge will still say (and our answer)

1. **"The A12 result is close to true by construction."** Under R_A, reading never earns reward, and the arm is saturated. *Answer:* yes. We present it as what the reward's information *selects*, not as a mechanism. ID1, which is running, tests whether read-back appears without demonstrations.
2. **"The start-state effect is three seeds and may be template noise."** *Answer:* yes. Two more control seeds were registered before training (addendum 2): the effect on recovery starts held across five runs per arm, the partial-payment effect did not reach the rule. Addendum 4 (the handed-evidence explanation) is inconclusive (Holm 0.206 and 0.262).
3. **"No new method."** *Answer:* correct, and stated. FCA failed its pre-set test and is reported.
4. **"Surface binding is already known."** *Answer:* yes. S4 is presented as a replication in a new setting (NOVELTY_MATRIX: "Can Agents Generalize to the Open World?"), and SR1 tested the known fix here: randomised tool names did not help on the renamed API (+0.00) and cost −0.25 on the canonical tools.
5. **"Toolkit use is thin in Stage 2."** *Answer:* AgentTune trains every model. AuditKIT and CuratorKIT were used in Stage 1. CircuitKIT did not install. The deck says exactly this.

## Before submission (owner actions)

- **Site deployed** (approved 2026-11-22): https://atharvgaur18-actionproof.static.hf.space/index.html
- **Adapters, splits and traces publishing** (approved 2026-11-22): `atharvgaur18/actionproof-proc-*` and the collection "ActionProof: procurement study".
- Deadline: Nov 24, midnight IST (confirmed by the author).
- **Share the deck** from its Share menu if anyone else should see it.
- **Record the video** with `docs/final/DEMO_RUN_OF_SHOW.md`.

## Labels for the final report

| Item | Status |
|---|---|
| Website procurement chapter, deck, claims ledger | Completed and verified locally (tests, number check, browser at 1440 px and 375 px) |
| S3 addendum 2, addendum 4, ID1 | Completed and verified (addendum 4 inconclusive) |
| SR1 | Completed and verified (not supported) |
| E1/E2 | Completed and verified (sampled contrasts not supported; reconciliation line +0.48) |
| Stage 5 development (C5, G) | Running; kill criteria fixed before training |
| Addendum 3 | Withdrawn before training |
| FCA | Failed |
| Mechanistic causal test, own-history recovery starts, second model family | Proposed future work |
