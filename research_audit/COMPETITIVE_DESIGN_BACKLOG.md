# Competitive design backlog

Written 2026-11-28 02:15 IST from `FINAL_20_COMPETITOR_INTELLIGENCE.md` and `FINAL_COMPETITIVE_SCORECARD.md`.

Every item is an original ActionProof improvement, suggested by a pattern seen in a competitor. None copies
competitor code, text or assets.

**How items are ranked.** By expected judge impact per hour, for a cutoff of 2026-11-28 23:59 IST.

**Fields.**
- **Impact:** H, M or L, on the scorecard dimension shown.
- **Time:** coordinator hours. GPU hours are listed separately.
- **Risk:** the scientific or integrity risk.
- **Gate:** the condition that must hold to ship the item (go), or to drop it (no-go).

**Owner rule.** This file only proposes. The coordinator decides, and owns the claims ledger.

## P0: compliance and presentation, which cost little and remove judge-visible gaps

| # | Item | Pattern seen in | Impact | Time | Risk | Dependency | Go / no-go gate |
|---|---|---|---|---|---|---|---|
| 1 | **Cite the arXiv paper of every Lexsi kit we use, and of our Cohere models.** AgentTune 2610.13335, AuditKIT 2610.22511 and CuratorKIT 2606.21631 (used). CircuitKIT 2607.19317 (attempted; cite only where WRITEUP §10 names it). Tiny Aya 2603.11510 (base of every trained agent). Aya Expanse 2412.04261 (P5 and the prompted baselines). Cite each at first mention and in the kit table (WRITEUP §10), in `paper/refs_v3.bib` and in the README kit rows. Export the BibTeX from `https://arxiv.org/bibtex/<id>` | 03, 04, 05, 12 (Nov 27, "organiser request / review") | **H** (compliance, P) | 0.5 h | None | None | **Go now.** Every ID's title was checked on arXiv on 2026-11-28. The author should also look in email and Discord for the organisers' review message itself, and apply anything else it asks |
| 2 | **Name the released model in the README headline** (`atharvgaur18/actionproof-agent-d`, from Tiny Aya Global 3.35B), and add a "Lexsi kits used" line at the top | 02 (kits line), 05 ("README headline names the post-trained models", organiser check) | M (S, compliance) | 0.2 h | None | None | Go. The headline must stay one sentence |
| 3 | **A "For the kits" box** (write-up and paper) that states what this project found that the kit maintainers can use: | 05 (paper box; one drafted issue per kit, not posted) | M (O, P; organisers maintain the kits) | 0.5 h | Low | None | Go if every number already sits in `claims/claims.yaml`. Post nothing to the kit repos without the author's approval |
| | – AuditKIT's `TaskCompletion` judge, with the ledger shown, still accepted 71% (213 / 298) of false completions | | | | | | |
| | – AgentTune: what we used (`TrlAgenticGrpo` for every GRPO policy; its parser and prompt in rollouts), and any patch or workaround we needed (to be checked in `actionproof/train/grpo_proc.py` and `actionproof/rollout/`; not verified here) | | | | | | |
| | – CircuitKIT failed to run (D-A16-1) | | | | | | |
| 4 | **Ethics statement and LLM-usage statement inside the paper**, not only in the write-up (WRITEUP §33 already has the share) | 05 (Ethics Statement), 12 (LLM-usage statement, author contributions) | M (compliance) | 0.3 h | None | #6 if a new short paper is made | Go |
| 5 | **The recorded video, the exported deck PDF, and captions matched to the cut** | 04 (2:48 video, slides PDF, .srt), 01 (87 s video), 03 (tour video), 06 (1:44 video) | **H** (P) | Author: 1–2 h | None | The author's go-ahead (the user asked to be told first) | Go once the author says so. Every on-screen label must match the run (`LIVE MODEL RUN`, `SCRIPTED`, `RECORDED MODEL RUN`) |
| 6 | **A 4–6-page short paper of the main result** (TXN benchmark, recovery agent, and RG1 if it passes) in the EMNLP template we already hold, with an Ethics Statement and the kit citations. Label the theory paper `main_v3` a companion | 02, 03, 04, 05 and 12 each ship a Stage 2 paper PDF of their main result | **H** (P, S) | 3–4 h | Medium: every number must come from the claims ledger, or `s2_numcheck` will not cover it | #1, #4; RG1 results if included | Go if 3–4 coordinator hours remain after RG1 analysis. Otherwise no-go: the write-up stays the primary document and the paper stays a labelled companion |

## P1: research and system items that raise the weakest scores (O = 2, hand-off)

| # | Item | Pattern seen in | Impact | Time | Risk | Dependency | Go / no-go gate |
|---|---|---|---|---|---|---|---|
| 7 | **Measured hybrid: the learned agent plus an evidence-only gate (RG1), against B4 on the same tasks and budget, with cost per episode.** It answers the review's "checks plus agent never measured" | 06 (a deterministic guard under the model, with its blind-spot rate), 08 (evidence before acting), 02 (safe stopping as the headline) | **H** (O, C, D) | In progress (coordinator) | Medium: the gate must read only agent-visible observations. A gate that equals B4 shows only that rules work, so state that | A sealed split registered before any test output | Go as registered. Report every result, including "the gate does the work and the model adds only intent parsing", if that is what the data show |
| 8 | **Gate rule ablations** (no read-before-retry, no stale-evidence handling, no partial delta, no escalation gate, no postcondition check) | 12 (circuit edits against random-layer edits: a negative control that names what does *not* carry the effect) | M (C) | GPU: about 1 h (five runs at about 10 min each) | Low | #7 | Go with #7, on the same sealed split |
| 9 | **Selective-escalation view with always-escalate and never-escalate references, on a risk–coverage plot, for D, D + gate and B4** | 11 (refuse-everything baseline), 02 (safe stopping) | M (A, O) | 1 h | Low | #7 results | Go if #7 yields the runs. Coverage and selective risk already exist in `truth.summarise` |
| 10 | **Cost per 1,000 transactions**, measured: tool calls, tokens and GPU seconds per episode for D, D + gate and B4, beside the illustrative dollar projection | 04 ($463 against $1,735 per million steps), 12 (cost per 1,000 replies) | M (O) | 1 h | Low: the GPU price is an assumption, so state it | #7 logs | Go. Separate measured counts from assumed prices |
| 11 | **Hypothesis ledger in the brief:** of N registered hypotheses, H held, P held in part, F failed, U untestable | 05 (25 hypotheses: 11 / 2 / 11 / 1) | M (V, S) | 0.5 h | Low: counting rules must be stated | None | Go if the count can be scripted from `claims/claims.yaml` statuses. No-go if it needs hand judgement on more than five items |

## P2: demo and judge-experience polish

| # | Item | Pattern seen in | Impact | Time | Risk | Dependency | Go / no-go gate |
|---|---|---|---|---|---|---|---|
| 12 | **Outcome-filtered explorer** over the registered TXN test episodes ("fixed by training / still failing / regressions"), in the replay Space. It uses the recorded episodes only, labelled as recordings | 03 (the Lab's explorer over 486 conversations) | M (D, P) | 2 h | Low | None | Go if the site build stays byte-reproducible (`scripts/site_upgrade.py`) and the page stays under about 5 MB |
| 13 | **A gate option in the live Space** ("D + Recovery Gate"): the gate's block decisions appear inline as a separate event type, never as model output | 06 (guard path on CPU), 04 (live demo states) | M (D) | 1.5 h | Medium: the labels must stay honest. A redeploy needs the staging test | #7 frozen | Go only after RG1's code is frozen and its tests pass. Test on staging first, as before |
| 14 | **Hero image of one contrast:** the same "ok" over two database states | 10, 01 (one memorable visual) | M (P) | 0.5 h | None | None | Go; reuse deck slide 1 |
| 15 | **"At a glance" block with a "Where it falls short" bullet** at the top of the README | 04 | L–M (S) | 0.2 h | None | None | Go. The README already has "In one paragraph"; this only adds the falls-short line if it is missing |
| 16 | **Scoring-rule sensitivity table:** the same episodes under VTC, under net-money re-scoring, and under "duplicates counted only if agent-caused" | 01 (three scoring rules) | M (A, V) | 1 h | Low | `results/audit/txn_validity.json` | Go if every cell comes from an existing result file |

## Rejected (do not do)

| Item | Seen in | Why not |
|---|---|---|
| Multilingual breadth as a new axis | 01, 07, 12 | Our problem is execution state, not language. S4 already shows Hindi and Hinglish |
| A frontier model as an in-loop advisor or judge of record | 04, 15 | Not a Cohere core role. External models stay supplementary |
| A 70B-scale model row | 01 | No bearing on the thesis; GPU time is better spent on RG1 |
| Quantization or GGUF results | 02, 05, 09 | Off-thesis |
| Dropping human labels quietly | 02 (`e38fff6`) | Our J1 76-case labels stay "pending, one annotator" until the author labels them; never removed silently |
| Copying any competitor figure, layout or text | all | Licences and originality rules |

## Suggested order for the coordinator

1. #1, #2 and #4: about 1 h, no compute. They close the compliance gap that three of the top five closed on Nov 27.
2. #7 and #8 (RG1), already running; then #9 and #10 from the same runs.
3. #3 and #11.
4. #5: the video, when the author says.
5. #6, if time remains.
6. #13, then #12, #14, #15 and #16, as time allows.
