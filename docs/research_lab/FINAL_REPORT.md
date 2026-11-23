# Research lab: final report (branch `research/lab-final`, 2026-11-24)

Evidence levels are kept separate:
- **[math]** derived;
- **[testbed]** exact synthetic testbed, registered (development level for algorithms);
- **[LLM-dev]** trained Tiny Aya Global, development seeds;
- **[conf]** confirmatory;
- **[expl]** exploratory.

Every number is traceable through `claims/claims.yaml` (LAB-* entries). `scripts/s2_numcheck.py` checks 406 numbers
across 7 documents with 0 failures.

## 1. The strongest defensible GRPO finding

**[math]** Theorem 1 (`docs/research_lab/GRPO_MECHANISM_FORMAL.md`) holds under these conditions:
- the rollouts of a group share a hidden state;
- payoffs are deterministic given that state.

Then GRPO's expected update on a binary decision is (G−1)·m_G(ρ)·Σ_z π(z)·sign(β_z): a stake-free vote across
hidden states. Dr. GRPO's is exactly (G−1)·ρ(1−ρ)·VOI, and RLOO's is G·ρ(1−ρ)·VOI. GRPO therefore verifies iff
p·s > 1/2, whatever the cost. The edge cases where the effect weakens or vanishes are stated: exogenous noise, s < 1,
condition B, c = 0, ρ = 0 and ε.

It is an extension of Bay and Yearick's group-std identity and Bereket and Leskovec's 1/σ weighting. It is not a new
mechanism in kind.

**[testbed]**
- An independent recount reproduces the v2 wrong-decision rates: GRPO 319/495 = 0.644, Dr. GRPO 28/495 = 0.057.
- The exact sign at the warm start predicts GRPO's learned decision in 494/495 cells.
- Causal intervention (LAB1 H4, supported): a scale from independent same-state rollouts reproduces the failure
  (0.667); a scale from fresh-state rollouts does not (0.061).
- In the multi-decision workflow, GRPO's exact expected gradient is wrong-signed on every risky verify logit (cosine
  −0.175).

**[LLM-dev]**
- The decomposition (−143.6 / +49.6 / −49.6; −0.36 when centred) reproduces byte-identically from raw rollouts.
  The ±49.6 symmetry is a property of those groups, not an identity.
- GRPO's sampled read share falls 0.19 / 0.20 → 0.02 / 0.04 with 66–81% zero-variance groups. Dr. GRPO's rises
  0.23 / 0.28 → 0.48 / 0.50 (same reward, pool and seeds; two development seeds each).
- The development greedy seeds are behaviourally identical, so they are not independent evidence.

**[conf]** The LLM test of the prediction, Dr. GRPO > GRPO (AD-C1, tasks 1220–1241), is **not run**.

## 2. Status of AWRB and IGC

**AWRB** (v2 audit, unchanged): it is not a better estimator.
- Its group-std variant's bias towards verifying helps exactly when verifying is optimal anyway. In LAB1 E1 it has
  regret 0.0105 against Dr. GRPO's 0.0296. It verifies in 18/18 mirror cells where VOI < 0.
- It needs the simulator and the prior weights, and it fails under prior misspecification.
- Confirmatory H_AWRB (awrbn > nostd) is pending in AD-C1. The prediction recorded beforehand is "not supported".

**IGC (igc_v1)** [testbed]:
- It has the best regret of any arm at E4 (0.0062) and cuts never-faulted verification 10× at E1.
- All of its gradient advantage is on uninformative decisions (SNR 23.5 vs 0.34); on risky logits it equals Dr.
  GRPO.
- It buys speed, not a different end point (v2 TB2 ×8).
- It needs the prior's support.
- Its LLM version (D3, tasks 1216–1217) has not run.

## 3. LLM judge calibration

See `docs/research_lab/JUDGE_RESULTS.md` (J2).
- **Human calibration is blocked on people.** There are no human labels yet, and a second annotator is needed for
  double annotation. The UI and adjudication tooling are ready (`docs/research_lab/HUMAN_CALIBRATION_PROTOCOL.md`).

## 4. The best trained agent

- **[conf]** The recovery-start GRPO agent (RSC): VSC +0.21 over the state-reward agent, 5/5 seeds; start-state
  effect vs class-matched control, Holm 0.036.
- **[LLM-dev]** Evaluated on the costed pool (one run), it has J_c 0.812, above every costed-pool arm.
- The P1 practical-agent study is the lead engineer's. Its frozen test is still running, so `results/p1/p1_summary.json`
  does not exist yet. Development showed arm C (recovery SFT) highest.

## 5. Is CDVO novel?

**No.** It is TRPO's vine estimator with common random numbers, used as COMA's counterfactual advantage.
- Forked on-policy counterfactual credit at selected agent decisions is CRR (arXiv 2609.33875, NeurIPS 2026).
- Adaptive branching is RSPI (2008), EPIG-Tree (arXiv 2609.20004) and ARPO.
- Details: `docs/research_lab/RELATED_WORK_AND_NOVELTY.md` §4 and `CDVO_THEORY.md` §7.

## 6. Does CDVO improve net utility or generalisation?

**No** ([testbed], `docs/research_lab/LAB1_RESULTS.md`).

| Hypothesis | Result | Verdict |
|---|---|---|
| H1: adaptive forks vs the strongest simple baseline at matched episodes | E4 0.0170 vs 0.0080; E1 0.0595 vs PPO 0.0215; 0/5 seeds at both budgets; Holm p 1.0 | not supported |
| H2: forks + igc vs igc | 0.0097 vs 0.0062; 0/5 seeds | not supported |
| H3: common random numbers reduce variance | variance ratios 0.94–1.12 | not supported |
| Generalisation | PPO best; fork arms 2–5× worse | fail |

Other findings:
- Adaptive allocation is no better than random (0.0170 vs 0.0143).
- Fewer forks are monotonically better.
- A stored-fork-mean critic collapses verification (regret 0.085).
- Why it fails: forks reduce variance only on uninformative reads, where −c is free. On informative reads the
  dominant variance is across hidden states. Forks cost 1.9–5.1× the episodes per update.

**The go/no-go rule rejects CDVO, and no GPU run was registered.**

## 7. Statistical evidence

- Registered protocols: `PREREGISTRATION_LAB1.md` and `PREREGISTRATION_J2.md`, frozen with
  `results/lab1_hashes.json` and `results/j2_hashes.json`.
- The disclosed pre-registration dry-run and smoke test used seeds 99 and 98–99 only.
- Analysis was run once: `results/lab/lab1_summary.json` (Holm over 4 tests; sign-flip over 60 units).
- Per-seed tables are in `paper/tables_v2/`.

## 8. Paper

`paper/main_v2.tex`, `paper/appendix_v2.tex` and `paper/refs_v2.bib`.
- The main text is 5 pages and builds cleanly; the current `main.tex` is untouched.
- Thesis: "group normalisation turns the value of checking into a vote".
- `\pend{}` marks AD-C1, P1, J2 and the human labels. These must be filled from result files or removed before
  submission.
- Bibliography entries for Sakib, Ding et al. and "others" need full author lists checked.

## 9. Public links and what to push

- Nothing new is public. Push `quality-upgrade` (or `actionproof-research-upgrade`), then `research/lab-final`, to
  origin `atharvgaur1845/actionproof-v2`.
- The cluster login node cannot reach GitHub, so pushing needs the author's laptop.
- Model, dataset and demo links are unchanged from the Stage-2 README.

## 10. Reproducibility and claims audit

- 48 lab tests and 20 v2 tests pass.
- The lab trainer reproduces v2's trainer bit-exactly.
- The run manifest now covers 2,188 CPU runs (`cpu_runs`) and the J2 judge files (`judge_runs`), each with commit,
  config, seed, steps, updates, episodes, forks, Slurm job, node, raw-file sha256 and status. Rebuild with
  `python scripts/run_manifest.py --merge-cpu-judge`.
- The partial local L1 attempt is preserved, unread, under `results/lab/moved_local/`.

## Decisions only the author can make

1. **AD-C1 priority.** H_nostd (Dr. GRPO > GRPO, cw_conf) is the only confirmatory LLM test of the paper's thesis. It
   waits behind P1/P2. Decide whether to keep it behind P2's remaining test tasks.
2. **V2-E1 (tasks 1301–1303).** The failure cause is verified (bash's built-in `GROUPS`). The fix is to rename
   `GROUPS` to `NGROUPS` in `slurm/ad2_arm.sh`, lines 4 and 11, as a documented deviation; that file is in the
   V2E1-frozen set. Re-queuing costs about 22 A100-hours and tests only the weakened prediction. Recommendation: do
   not run it before the deadline.
3. **Which paper.** `main_v2.tex` restructures around one thesis and moves the RSC/acquisition results to supporting
   roles. The lead engineer integrates; the author approves the title and thesis.
4. **Human labels.** Label the 76 cases (about 2–3 h). Recruit a second annotator, or report "single annotator".
5. **Licences.** Gemma 4's Hub metadata says apache-2.0, but J1 says "Gemma licence". Aya Expanse is CC-BY-NC-4.0,
   which constrains any commercial use of judge outputs.
6. **Push** the local branches (§9).
7. **Team contributions** paragraph (`\pend` in appendix_v2).
