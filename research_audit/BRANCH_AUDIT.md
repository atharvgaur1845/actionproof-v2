# Branch audit

Run on 2026-11-25 at 07:40 IST against `origin` after `git fetch`. Commands: `git rev-list --count A..B` and `git diff --name-only origin/main...B`. Every hash below was read from the repository; none comes from a document.

## 1. Comparison

| Branch | Tip | Date | Commits ahead of main | Files changed vs main | Commits not in `submission-candidate` |
|---|---|---|---|---|---|
| main | d19c299 | 11-21 | 0 | 0 | 0 |
| stage4 | 6147ede | 11-22 | 37 | 685 | 0 |
| research/e1e2 | b49a581 | 11-23 | 41 | 843 | 0 |
| research/algorithm-discovery | c8689a2 | 11-23 | 59 | 1,054 | 0 |
| research/algorithm-discovery-v2 | 9738554 | 11-23 | 105 | 1,508 | 0 |
| actionproof-research-upgrade | 295a6a8 | 11-23 | 115 | 1,605 | 0 |
| research/lab-final | b2b2de3 | 11-24 | 128 | 1,698 | 0 |
| quality-upgrade | 4f26d37 | 11-24 | 149 | 2,143 | 0 |
| research/latent-state-theory (local) | 39a09c4 | 11-24 | — | — | 0 |
| **submission-candidate** | **1c3e283** | **11-24** | **188** | **2,239** | — |

**Finding 1. `main` is the least complete state.** It stops at the Stage 3 results (Nov 21). Every remote branch is an ancestor of, or fully merged into, `submission-candidate`. None of them holds a commit that the candidate lacks.

**Finding 2. The branches form one lineage, not parallel forks.** Each was cut from the previous one and later merged forward. No branch carries an alternative version of a result.

**Finding 3. Five early agent worktree commits (Nov 20) are not ancestors of the candidate**, but every file they touched is present there:

| Commit | Content | Files identical in the candidate | Different (later version in the candidate) |
|---|---|---|---|
| 7670411 | Independent ProcureSim golden traces (91 cases) | 1 of 3 | 2 (traces extended later) |
| e8e0bf4 | Independent reference verifier (proc1) | 0 of 2 | 2 (`scripts/reference_verifier_proc.py` was extended for proc2 faults) |
| 48ecfab | Adversarial end-state cases | 4 of 4 | 0 |
| be80c5a, b306f6c, 2e14b97 | A11 gateway, untouched splits, **draft** A11 registration | 10 of 23 | 13 (the draft registration was replaced by the frozen `PREREGISTRATION_A11.md`) |
| 5f1bb58 | Gateway dashboard | 15 of 16 | 1 |

No result or registration exists only on these commits.

## 2. What each branch added (distinctive commits)

| Branch | Experiments and code | Documentation-only | Changed previously reported results? |
|---|---|---|---|
| stage4 | S4-EVAL shifts, FCA (failed), SR1, ID1, addenda 2 and 4, product | stage-4 docs | No. The Stage 3 values stand, and addendum 2 added seeds by registration |
| research/e1e2 | E1/E2 (sampled decoding, decision-point probes), Stage 5 development (C5, G) | ledger E1E2 | No |
| research/algorithm-discovery | AD tabular testbeds, AWRB, D1 gates, first LLM mechanism measurement | FINAL_ALGORITHM | No |
| research/algorithm-discovery-v2 | Vote rule, igc, D2/TB2/TB7, AD-C1 notes 2–3 | algorithm report | **Corrected its own claim**: the igc exactness claim was narrowed (a9eff9f) |
| actionproof-research-upgrade | R1, R1-T, J1, ROI, site sections 06–08 | write-up, final review | No |
| research/lab-final | Exact vote theorem, LAB1 (CDVO rejected), J2 registration, paper v2 | paper v2 | No |
| quality-upgrade | P1, P2, guard diagnosis, reporting module, canonical results, number check over the write-up | quality reports | **Relabelled one diagnosis row**: the "VSC" row was the goal-met rate. Versioned in commit b359ff5 and later; no registered statistic changed |
| research/latent-state-theory | Phase II theory engine, TH1 (registered), mechanism diagnostics, J2 analysis, paper v3 | theory docs | **Narrowed earlier claims**: "494 of 495" counts runs, and the p·s > 1/2 rule holds only with state-level repairability |

## 3. Abandoned or superseded

- **Abandoned methods:** FCA, C5, G, CDVO and D2. Each was dropped by criteria fixed in advance, and each is kept and reported.
- **Superseded papers:** paper v1 (`main.tex`) and v2 (`main_v2.tex`, which carries author identity) are superseded by v3, which is anonymised. The v1 and v2 files are kept.
- **Superseded drafts:** `docs/ARTICLE_DRAFT.md` (marked superseded) and `docs/final/FINAL_CLAIMS.md`, whose status is superseded by the canonical file.
- **Held tasks:** the v2 LLM tasks 1301–1303 stay in `queue/a100/hold` (a GROUPS bug; no new algorithm GPU work).

## 4. Recommendation

**`submission-candidate` is the canonical research branch.** It contains every remote branch, so no merge is needed to recover any experiment.

`main` should be fast-forwarded to it only with the author's approval. Updating the default branch changes what the public repository shows.

## 5. Containment re-check, every local and remote ref (2026-11-25, addendum audit)

Method: `git merge-base --is-ancestor <ref> submission-candidate` for every ref in `refs/heads` and `refs/remotes/origin`; for refs that fail, `git cherry submission-candidate <ref>` and a blob comparison of every file the ref touches.

**Contained as ancestors (no commit missing):** `main`, `stage4`, `quality-upgrade`, `actionproof-research-upgrade`, `research/ad-c1-theory`, `research/algorithm-discovery`, `research/algorithm-discovery-v2`, `research/e1e2`, `research/lab-final`, `research/latent-state-theory`, all nine `origin/*` branches, and the worktree branches `worktree-agent-a8db2b87`, `-ab40f210`, `-ad6ea450`, `-afa8d08a`.

**Not ancestors, but their content is present.** Five worktree branches from 2026-11-20 hold eight commits that were cherry-picked rather than merged:

| Ref | Commits not ancestral | Content | Present in `submission-candidate`? |
|---|---|---|---|
| `worktree-agent-a0dcc706` | 7670411 | 91 ProcureSim golden traces, structural checker | yes, patch-equivalent (`git cherry` "−"); `proc_golden.jsonl` later revised |
| `worktree-agent-a3e141bd` | be80c5a, b306f6c, 2e14b97 | A11 untouched splits and fault labels; e-commerce verification gateway; A11 draft registration and tasks 500–509 | be80c5a and 2e14b97 patch-equivalent; b306f6c re-committed as de0d40c (same message); every file present, some later revised |
| `worktree-agent-a6aa0cd4` | 5f1bb58 | gateway dashboard, API tests, screenshots | patch-equivalent; `api.py` later revised |
| `worktree-agent-a894375d` | 48ecfab | 194 adversarial procurement end-state cases | patch-equivalent |
| `worktree-agent-aa813db5` | e8e0bf4 | independent reference verifier (proc1 spec) and tests | patch-equivalent; `reference_verifier_proc.py` later revised |

**No file touched by any branch is absent from `submission-candidate`.** No branch holds an experiment, result or registration that the canonical branch lacks. The statement in §4 stands.
