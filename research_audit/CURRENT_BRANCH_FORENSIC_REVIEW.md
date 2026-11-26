# Current branch forensic review

**Audited.** 2026-11-27, 05:06–07:00 IST.

## 0. Repository state at audit start

| Item | Value |
|---|---|
| Working copy | `/home/atharv/Desktop/projects/Lexsi_Hackathon/actionproof` |
| Branch at start | `submission-candidate`, up to date with `origin/submission-candidate` |
| HEAD | `7ae863f50dbbd4725c3c97e72489f4c36f6efeba` (2026-11-27 03:32 IST, "Write-up §30: no part predates Nov 1, 2026 (author)") |
| Remotes | `origin` = `git@github.com:atharvgaur1845/actionproof-v2.git`; `old-origin` = `…/actionproof.git` |
| Uncommitted | `slurm/tasks/1600_judge_next.sh` (edited by someone else: the minutes and the 32B line; left untouched); untracked build products (paper `*.aux/*.log/*.pdf`, `build/`, `data/rft/p4_aug_rows.jsonl`, `data/bench/proc/c1_pp.jsonl`, rollouts) |
| Other branches | `main` (`d19c299`), `stage4` (`6147ede`), `quality-upgrade`, `research/*`, `actionproof-research-upgrade`, and 12 `worktree-agent-*` branches. **Every remote branch is already an ancestor of `submission-candidate`** (0 commits ahead), so no relevant work exists only elsewhere. |
| Sprint branch | `stage2-superiority-sprint`, created from `7ae863f`. Local commits only; no push. |
| Commits since `e958f37` | 55 (matches the prompt) |
| Size | 602 commits; 52 `PREREGISTRATION*.md`; 68 test files, 426 test functions; `.github/workflows/ci.yml` present |
| Local check | `pytest tests/txn`: 144 passed in 6.3 s |

## 1. What the 55 commits actually implement

Each claim below was checked against the tree and the result files.

| Claimed addition | Paths / commits | Status |
|---|---|---|
| TXN1 benchmark, spec, registration, addenda | `docs/txn/TXN_SPEC.md`; `PREREGISTRATION_TXN1.md`, `_ADDENDUM1.md`, `_ADDENDUM2.md`; `actionproof/txn/{env,truth,invariants,policies,tasks}.py`; `scripts/txn_*.py`; commits `d6cc756` … `fa61204` | **Implemented.** The env adds the audit log, `consistent` reads, four new faults, and delayed/unresolvable regimes. Mutation testing: 25 mutants, all killed (`33979cc`). There are goldens for escalation correctness. Registered (`36e5a66`, 00:45 Nov 26) before any model episode (first TXN model log 03:29). |
| P4 randomised demonstrations, 3 seeds | `PREREGISTRATION_P4.md` (`b5998ae`, 00:07 Nov 26); `actionproof/p4/build.py`; `scripts/p4_eval.py`, `p4_stats.py`; `results/p4/p4_summary.json` | **Implemented and consistent.** H1 +0.604 [0.540, 0.666], seed deltas +0.612/+0.613/+0.587; H2 +0.574 [0.536, 0.610]; H3 +0.048 [0.026, 0.072]. All supported (Holm 0.0002). |
| P5 recovery SFT on four bases | `PREREGISTRATION_P5.md` + addendum 1; `scripts/p5_eval.py`, `p5_stats.py`; `results/p5/p5_summary.json` | **Implemented and consistent.** Fire +0.668, Water +0.628, Earth +0.693, Expanse 8B +0.697; one SFT seed each. Per-split data exist (Water wf4 = 0.017, stated in the README). |
| B1 prompted baselines (incl. Qwen3-235B, gpt-oss-120b) | `PREREGISTRATION_B1.md` + addendum 1; `results/b1/*` (tag_base, aya8b, aya32b_bf16, qwen, gptoss × 5 splits); `results/b1/api_cost.jsonl` | **Implemented.** Greedy, through the HF router; cost logged. |
| R7 GRPO increment | `scripts/audit/r7_*`; `results/audit/r7_grpo_increment.json` (status "EXPLORATORY (R7; after the 2x2 was known)") | **Implemented; labelled exploratory.** |
| TXN raw episodes and statistics | `results/txn/*__txn_test.jsonl` (A1–A6, A/D/Dn × 3, B0–B4, W, Dw, Dw_aug × 3, prompted); `stats__txn_test.{json,md}`; `invariants_recheck__txn_test.json` | **Present.** |
| Live ZeroGPU Space | `space_live/{app,core}.py`, `space_live/tests/test_app_logic.py`; Hub `atharvgaur18/actionproof-live` (gradio, RUNNING, zero-a10g, last modified 2026-11-25 22:38 UTC) | **Implemented, live.** Greedy, 256 new tokens; four realities; a session cap. No judge mode, no export, no in-Space replay fallback (only a link). |
| Replay Space update | `demo/static/index.html` (built), `scripts/build_site.py`; Hub `atharvgaur18/actionproof` (static, RUNNING, 2026-11-26 21:54 UTC) | **Present.** |
| Model cards, dataset card, trace dataset refresh | `release/model_cards/*.md` (6), `release/dataset_card.md`, `release/build_cards.py`; Hub repos last modified 2026-11-26 21:41–21:55 UTC | **Present and public.** Revisions: agent-d `5418df5`, d-adapters `b3d97d6`, warmstart-dw `0c4b613`, warmstart-dwaug `6c9c156`, agent-a `5ea80eb`, agent-dn `1b79da4`, traces `8e91415`. |
| Six-slide source, video script, captions | `docs/submission/SLIDES.md`, `VIDEO_SCRIPT.md`, `VIDEO_CAPTIONS.srt` | **Markdown source only.** No rendered deck (PDF/PPTX) and no recorded video. |
| Expanded `verify_numbers.sh` and numcheck | `scripts/verify_numbers.sh`, `scripts/s2_numcheck.py`, `claims/claims.yaml` (791 file references) | **Present.** See §4 for scope. |
| Adversarial review corrections | `48f0e6c` | **Present.** 35 findings; fixed in wording. |

## 2. Headline numbers re-read from source files

Every value below was read from the JSON files, not from the README.

| Claim | Source | Value | README agrees? |
|---|---|---|---|
| P1 A / C / D sampled VSC | `results/p1/p1_summary.json` `test` | A 0.5128; C 0.7719 (Δ +0.259 [0.231, 0.287], Holm 0.0003); D 0.8193 (Δ +0.306 [0.276, 0.337], Holm 0.0003); B 0.4830 (Δ −0.030, not supported) | yes |
| TXN H3 family | `results/txn/stats__txn_test.json` `H3` | DMR vs B3 −0.167 [−0.271, −0.070], Holm 0.0012 ✓; URR vs B3 −0.027 [−0.067, +0.003], Holm 0.123 ✗; DMR vs B1 −0.314 ✓; URR vs B1 ✗. n = 320, 48 clusters | yes |
| TXN H5 | `results/txn/stats__txn_test.json` `H5.per_model` | CER: D_s0 9/180, D_s1 1/180, D_s2 5/180 (15/540); unsafe 8 + 3 + 3 = 14/540. Dw CER 4/180, unsafe 8/180. B4 CER 60/60. **B1 CER 0/60 and unsafe 0/60:** "report failed" is also harmless here. A5 (never escalates) unsafe 15/60, all on ok_phantom. | yes. Unstated nuance: D's *unsafe* rate on unresolvable evidence is low (selective risk 0.017–0.029). Its failure is the missing **escalation disposition**, while Dw_aug's failure is both missing escalation and unsafe action (53/540). |
| P4 | `results/p4/p4_summary.json` | as above | yes |
| P5 | `results/p5/p5_summary.json` | Fire 0.0178 → 0.6861; Water 0.0111 → 0.6394; Earth 0.0122 → 0.7039; Expanse 0.0756 → 0.7728 | yes |
| R7 | `results/audit/r7_grpo_increment.json` | exploratory; +0.140 | yes |

## 3. Evidence status of each line of work

| Line | Status | Scope that must accompany it |
|---|---|---|
| P1 C > A, D > A | registered, supported | 450 frozen tasks, 3 training seeds per arm, sampled; C and D are not nested |
| P1 B (inference line) | registered, not supported | — |
| TXN H3 | registered family, **not supported** (2 of 4 pass) | in-distribution groups, n = 320 |
| TXN H1, H5 | descriptive | H5: 60 tasks × 3 episodes × 3 seeds |
| P4 H1–H3 | registered, supported | warm starts only; one held-out surface, one workflow; Dw_aug has +21% rows |
| P5 | registered, supported | one SFT seed per base; task-level test with each model fixed |
| R7 (D − Dw) | exploratory | seed and task variability not separated |
| B1 API baselines | descriptive, supplementary | greedy single runs; not Cohere; not budget-matched |
| P3 (no-std) | registered, not supported | −0.007 |
| TH1, AD-C1 | registered (theory testbed; language-agent comparison) | ProcureSim reads are free, so the theory's regime is not the practical agent's |
| J1 judges | registered | no human gold labels |
| A15/A16 interpretability | exploratory | causal patching failed its registered rule; CircuitKIT did not run |

## 4. Verification tooling: what it covers and what it does not

**`scripts/s2_numcheck.py`.** It scans the 13 files under `claims.yaml: documents`:
- `docs/stage2/VIDEO_SCRIPT.md`, `docs/stage2/PRODUCT.md`;
- `docs/submission/{WRITEUP,SLIDES,VIDEO_SCRIPT}.md`, `VIDEO_CAPTIONS.srt`;
- `README.md`;
- `paper/main.tex`, `paper/appendix.tex`, `paper/main_v2.tex`, `paper/appendix_v2.tex`, `paper/main_v3.tex`,
  `paper/appendix_v3.tex`.

**Not scanned:**
- `release/model_cards/*.md`, `release/dataset_card.md`, `release/README_SECTION.md`;
- `demo/static/index.html` (generated from result files by `scripts/build_site.py`, which is safer but unchecked);
- `space_live/README.md`, `space_live/core.py` copy;
- `docs/final/*`, `research_audit/*`, `docs/ARTICLE_DRAFT.md`, `docs/ABSTRACT.md`, `docs/HF_ARTICLE.md`.

It checks numbers, not sentences. The adversarial review on Nov 27 found wording overclaims that it missed.

**`scripts/verify_numbers.sh`.**
1. Downloads every `*.jsonl` under `results/{p1,p2,p3,p4,p5,b1,txn}` from `atharvgaur18/actionproof-traces`
   (about 2 GB).
2. Reruns `p1_stats`, `p2_stats`, `p3_stats`, `audit/dwarm_2x2`, `p4_stats`, `p5_stats`, `txn_stats` (the
   25 registered tags), `txn_invariants_recheck` and `txn_compact`.
3. Requires `git diff --exit-code results/`, except that the TXN JSON is compared within 1e-9.
4. Runs numcheck.

Gaps:
- It **does not** re-run the Stage 1 studies (A4–A12, ID1, S3/S4, E1/E2, TH1, AD-C1, J1, R1). Their summaries are
  committed and only number-checked.
- It needs `git`, about 2 GB and roughly 10 minutes; there is no smaller smoke route.
- The README's reproduce §2 still describes a Nov 15 clone check ("no CI is configured"), but
  `.github/workflows/ci.yml` exists.
- It does not reproduce training or inference. It verifies stored results only.

**Fresh-clone status.** A fresh-clone run of `verify_numbers.sh` is not recorded in this repository for the current
HEAD. Commit `1021a92` says it was "checked against the cluster rerun", which is not the same as a fresh clone.
Phase F of this sprint must run it from a clean clone, on the cluster: CPU on the worker queue, never the frozen main
copy.

## 5. Coherence problems found (prompt §2.4)

1. **Three theses.**
   - README: "trains and evaluates tool-using agents to reconcile transaction evidence and recover safely".
   - `WRITEUP.md`: title "what reinforcement learning teaches tool agents when 'ok' does not mean it happened". Its
     abstract leads with the reward information source (Stage 2's A12/ID1); TXN, P4 and P5 are appended as §1a.
   - `paper/main_v3.tex`: "When Group Rollouts Share a Hidden State: Group Normalisation Cannot See the Stakes of
     Verification". A theory paper. Its abstract ends on P3, and TXN appears only in the context paragraph.

   These are three different first sentences for one submission (decision: `docs/submission/FINAL_STORY_DECISION.md`).
2. **The README is a historical record, not a submission page.** After a good judges box come Stage 1–4 history,
   an outdated deliverables table (`docs/ARTICLE_DRAFT.md` as the write-up; "29 LoRA adapters"; `demo/` described
   as "Gradio Space (live ZeroGPU runs…)", which is the old demo) and an outdated reproduce section.
3. **No rendered deck or video,** although the final round is judged on a video or slides.
4. **Many stale surfaces** in `research_audit/`, `docs/final/`, `demo/README.md` and `release/README_SECTION.md`
   (table: `STALE_SURFACES_AUDIT.md`).
5. **The reference verifier is not independent.** Same author and specification. Mutation tests exist for TXN
   (25 of 25 killed). The README should call it a consistency check.
6. **H5 framing.** "Rarely escalates" is accurate. A precise framing also says that D seldom acts unsafely there
   (14/540) but almost never hands off (15/540), while Dw_aug acts unsafely in 53/540.

## 6. What this review changes in the plan

- Story decision first (Phase B). Then the deck, the README restructure and the stale-surface cleanup (P0).
- The open scientific gap with the highest judge value is escalation (H5). It is addressable with demonstrations
  and needs only TXN's existing `escalate_to_human` and `unavailable_reads` machinery, plus a transient-outage
  regime to test selectivity (Phase B experiment matrix).
- Reproducibility: add a small smoke route and an artefact manifest with verified Hub revisions. Run
  `verify_numbers.sh` from a clean clone on the cluster.
