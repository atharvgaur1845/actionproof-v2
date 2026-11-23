# Scientific audit and gap report (deliverable A)

2026-11-23, branch `actionproof-research-upgrade`.

The machine-readable index is `results/inventory.json`, rendered as `docs/upgrade/INVENTORY.md` by `scripts/build_inventory.py`. The claim-level source of truth is still `claims/claims.yaml`, checked by `scripts/s2_numcheck.py`: 290 numbers re-read from their files, 0 failures. The per-claim wording rules are in `docs/final/FINAL_CLAIMS.md`.

## 1. What the record contains

| Item | Count | Source |
|---|---|---|
| Registrations | 33 (30 earlier, plus R1, R1-T and J1 today) | `PREREGISTRATION_*.md` |
| Ledger claims | 39: supported 33, not supported 5, refuted 1, pending 0 | `claims/claims.yaml` |
| Claims citing a file that does not exist | **0** | inventory |
| Training runs with a final adapter on the cluster | 93 | `runs/*/final` |
| Recorded evaluation episodes used for judging | 14,400 (`s3_conf_full`, `s4_lang_hi`, `s4_lang_hinglish`) | `evaluation/llm_judge_committee/build_cases.py` |

## 2. Findings by evidential status

| Status | Findings (ledger row in `docs/final/FINAL_CLAIMS.md`) |
|---|---|
| **Confirmed:** registered, rule met | State reward selects read-back, 1.00 vs 0.00 in 5/5 seeds (2). +0.27 to +0.32 VSC per seed, R_V vs R_A (3). Simple forced evidence is acted on (4). Recovery-start recipe +0.21 over the state-reward agent, 5/5 (7). Start-state effect on recovery starts vs a class-matched control, five runs per arm, Holm 0.036 (9a). Hindi, Hinglish and two-fault shifts hold (13). Reward selects and does not create (ID1, 2b). Stage 1 H1: PCSTR-d 0.8425 vs 0.8275 |
| **Negative:** registered, rule not met, or refuted | Renamed tools and an unseen workflow fail (14). SR1 tool randomisation: +0.00 on renamed tools and −0.248 on canonical tools (14b). FCA dropped (16). Partial-specific start-state effect not supported (9a). Addendum 4 inconclusive (9b). E1 sampled contrasts not supported (10a). A11 interaction not made (C3, synced today). Negative transfer to healthcare administration (15). After-receipt partial repair unsolved, and rebalancing does not fix it (12, 12b) |
| **Exploratory:** descriptive or unregistered | Per-class decomposition (8). Learners per run (10). E2 reconciliation line +0.48 (10b; registered as a probe, but one scripted turn). A16 representation ladder. Stage 1 PCSTR-d vs tool-trusting reward, +20.25 pp (different reward forms, not registered) |
| **Untested until today** | General-capability retention (R1, R1-T running). Judge validity beyond one family (J1 running). Human calibration (J1 §5). ROI with a loss model |
| **Future hypotheses** | A second model family. An own-history recovery-start arm. A mechanistic causal test (CircuitKIT did not install). A new learning algorithm: the separate research agent is on it, and C5 and G are in development |

## 3. The strongest defensible headline

**Same base model, same warm start, the reward's information source changed and nothing else (A12).** The state-grounded reward gives **+0.27 to +0.32 verified completion per seed over the acknowledgement-grounded reward, in 5 of 5 paired seeds**, with phantom read-back at 1.00 vs 0.00.

It needs two qualifiers:
1. **The R_A arm barely moves from the warm start.** So the contrast measures what the state reward selects; it does not show that R_V creates the behaviour, as ID1 confirms.
2. **The seed-level sign-flip test over two primary metrics has Holm p = 0.0625.** That is the minimum attainable with five seeds, so "5 of 5" is a consistency statement.

The Stage 1 +1.5 pp best-baseline margin must stay separate from the +20.25 pp tool-trusting contrast. It also has a weakness of its own: both PCSTR-d seeds give identical outcomes on all 400 tasks, so it is effectively one run.

## 4. Weaknesses that a judge can raise, and their status

| # | Weakness | Status |
|---|---|---|
| 1 | Task-level intervals hold the trained models fixed, and seed-level power is low (5 runs) | Disclosed in every row; seed tests reported separately |
| 2 | The S3 recipe comparison adds a fault type (duplicate delivery) | Class-matched control, addenda 1–2 |
| 3 | The template-cluster interval for the start-state effect included 0 (three seeds) | Superseded by five runs per arm (9a), not re-done at template level |
| 4 | Greedy decoding makes run outcomes look bimodal | E1 (sampled): contrasts not supported. Greedy and sampled agree per run (ρ 0.92) |
| 5 | One model family, Tiny Aya Global 3.35B | Not addressed; stated as a limitation |
| 6 | The earlier LLM-judge evidence used Aya, the policy's own family | J1: four other families, human calibration |
| 7 | Training could damage general ability (the user's main concern) | R1 (IFEval, BFCL) and R1-T (τ-bench), registered before output |
| 8 | Stage 1 registration texts are frozen only by git, while later studies hash them | A4–A7 have one commit each, never edited. The main registration carries three in-file amendments, all logged |
| 9 | Ledger drift: C3 was marked pending after its analysis had run | Fixed today; the inventory now flags this class of gap |
| 10 | Ten registrations have no ledger claim (A5–A8 addenda, S3 addendum 3 withdrawn, plus today's three) | Stage 1 ablations are summarised in the README; R1, R1-T and J1 claims are added when their results land |

## 5. Leakage and contamination checks on file

- Split hashes and frozen generators: `results/*_hashes.json`, `scripts/ci_freeze_check.py`.
- The ProcureSim acknowledgement is byte-identical across outcomes, by test: `tests/integrity/test_procsim.py`.
- Confirmatory splits are run once (rule in `docs/stage2/PLAN.md`); E1/E2 used fresh hashed splits.
- R1 benchmarks share no prompts or schemas with ProcureSim training. Tiny Aya's own pre-training exposure is unknown and affects every arm alike (`PREREGISTRATION_R1.md` §3).
- J1 judges receive no hidden-state fields; they live only in `truth.jsonl` (`build_cases.py`).
