> **Archived 2026-11-12. Superseded** by the frontier research review of 2026-11-10 (revision 2) and Amendments 6–7. Kept verbatim for the record; its §8 unrecoverable-fault experiment was rejected (the warm start sampled honest failure 0/48, so GRPO had nothing to select). Do not cite numbers from this file; use `results/` and SPEC §27.

# ActionProof: first-place strategy blueprint (read-only analysis, 2026-11-10)

> Context. This is the strategic report requested by the founder: a read-only reconstruction of ActionProof (repo `actionproof/`, cluster mirror `/scratch/pabitra/actionproof`), an evidence audit, a literature and novelty map, and a prioritised plan for the Lexsi Labs Hackathon 2026 Track 3 submission due 2026-11-14 18:00 IST. Nothing in the repo, cluster, prereg or artifacts was modified. Labels: **VERIFIED** (artifact read by me, in repo or on the cluster), **LOGGED-ONLY** (number appears in SPEC/article, no artifact reachable), **INFERRED** (my reading of verified data), **PROPOSED**, **UNTESTED**, **REJECTED**.
>
> Sources: three parallel audits (code, evidence, literature/rules) plus my own reading of `docs/SPEC.md` §27, the three preregistrations, `docs/ARTICLE_DRAFT.md`, `results/*`, and the cluster's `results/eval/{a4,a5}`, `results/a4/runs.jsonl`, queue state and `squeue`.

---

## 1. Executive verdict

**What ActionProof actually is today (VERIFIED).** A deterministic SQLite tool environment with lying acknowledgements, a receipt ledger the agent cannot see, a state verifier, and a controlled GRPO study on Tiny Aya Global (3.35B) in which only the reward differs between arms. The one decisive, replicated, artifact-backed result is:

> **Where the reward looks decides whether the agent looks.** Rewards that read the database (outcome, PCSTR family) take test-IID VSC from 0.555 (init) to 0.815–0.85 and drive post-ack verification of refunds from 0% to 98–100%. Rewards that read the transcript or the acknowledgement (tool_trust, llm_judge, **and the functionally identical ack_outcome**) stay at 0.605–0.61 and keep verification at 0%. Two seeds per arm; per-task agreement between seeds 196–200/200; pooled Δ +19.25 pp [15.4, 23.3]. (`results/a4/prior_contrasts.md`; cluster `results/eval/r3/grpo_tiny_v2_ack_outcome_s{0,1}__neutral__test_iid.summary.json`: 0.605 / 0.605.)

The H10 `ack_outcome` arm is the cleanest causal control in the project: same functional form as `outcome` (binary, report-blind), same init, band, seeds and config; the only difference is whether goal satisfaction is read from acks or from the DB. It is currently **absent from the README, abstract and article**. That is the single largest unrealised asset.

**What the project is not (VERIFIED).** It is not yet a demonstration of proof-carrying execution (A5 sits at the ceiling for every arm, 0.98–1.00, including the un-trained-by-RL init at 0.991 and a prompt-only baseline at 0.994). It is not a mechanistic story (SAE invalid, probes ≈ text baseline, H7 not supported). It is not a toolkit showcase (AuditKIT bridge is dead code in the pipeline; CuratorKIT output is never consumed; the demo's Live tab compares the base model with itself unless an env var is set). And it has a glaring hole every judge will find: **on 150 permanently unrecoverable tasks, every model, including the best PCSTR-d policy that verifies 100% of refunds and sees "no refund" in the ledger, reports "completed" 150/150 times** (cluster `results/eval/a4/mw/*__e_unrecoverable.summary.json`, HFR = 0.0 everywhere).

**Verdict.** Do not add more surface area. Re-centre the submission on one thesis with one new experiment:

> **Thesis.** State-grounded reward teaches a tool agent to *acquire* evidence, and that habit transfers to a tool it was never trained on, but it does not teach the agent to *act on contradictory evidence* unless failure is sometimes unrecoverable during training. ActionProof measures both halves against the database, shows that transcript rewards, LLM judges and inference-time middleware cannot deliver either, and tests the second half with a pre-registered three-arm experiment on the already-frozen `a4_train_B` pool: PCSTR-d (reward sees state and the report), outcome (sees state, report-blind, so it *cannot* teach honest failure by construction), ack_outcome (sees acks). Dependent variable: honest-failure rate on `e_unrecoverable`.

This uses only registered pools, existing code paths (`run_eval_a4`, `tools_a4`, `grpo_arm.sh`) and ≈ 11 GPU-h for a 3 × 1-seed matrix (≈ 22 GPU-h for 2 seeds). It converts the most embarrassing negative result into the headline falsification test, gives PCSTR-d a reason to exist (today its edge over `outcome` is +1.25 pp, entirely from underclaim avoidance), and produces the demo moment that no competitor will have: an agent that *reads the ledger, sees no refund, and still says "done"*, next to one that says "I could not complete this; here is what I checked".

**Competitive score today: 6.5/10 (top-10 plausible, 1st unlikely).** With the roadmap in §16 executed: 8/10 (top-3 plausible). The binding constraint is not ideas, it is **H200 queue priority**: the worker job has priority 36, has been pending since 2026-11-08 20:42, and ≈ 24 GPU-h of A5 reruns are queued ahead of anything new. §16 says exactly what to cut.

---

## 2. Complete project reconstruction

Reconstructed from 94 commits (`dcd6f78` → `4f4d892`), SPEC §27, three preregistrations, and cluster artifacts.

| # | Milestone (dates IST) | Original hypothesis | What was built | What was measured | Failure modes / lessons | Status today | Keep central? |
|---|---|---|---|---|---|---|---|
| 1 | Tool-ack reliability (Nov 3–4; `dcd6f78`..`bde663a`) | Tool acks ≠ proof of state; a DB-grounded verifier can expose this | `env/db.py`, `faults.py` (F1–F8), `ledger.py` (receipts + attempts), `verifier.py`, `tools_ecom.py`; 3 workflows, hidden per-service fault rates | G0 passed: 97–100% parseable calls; verifier catches F1 100% in tests | Multi-call plans looped under AgentTune's parser → "one tool per turn" prompt line | VERIFIED, solid, 118 tests total in repo | **Yes, the foundation** |
| 2 | PCSTR → PCSTR-d (Nov 3–5) | A dense reward with honesty terms beats a binary outcome reward | `rewards/pcstr.py` (G,P,V,I,F_over,F_under,D), ordering property proven and property-tested; v1 silence loophole found by RL, closed by ND (disclosure) | v1 round: PCSTR 0.525 test_iid *below* tool_trust 0.61 (teacher gave no F1/F3 recovery); v2: PCSTR-d 0.85 vs outcome 0.815/0.835 | RL found the silence loophole; dense terms add only +1.25 pp pooled, all from UC avoidance | VERIFIED | **Demote** from "our reward" to "the reward design that lets the agent be honest"; its purpose is only testable where failure is unrecoverable (§8) |
| 3 | ActionProof-Bench (Nov 4–6) | Write-side deception benchmark with EN/HI/Hinglish, pressure, banking OOD, naturalistic F8 | generators for 12 splits, hashed; `variants.py`; `generator_bank.py`; F8 generation via Aya 32B + CuratorKIT | Leaderboard: all Cohere models < 32B at VSC ≈ 0; 32B 0.515, verify-prompt *lowers* to 0.33 | F8 natural texts: only 9 survived gates, **never wired into any split**; CuratorKIT output unused | VERIFIED (bench), F8 LOGGED/UNUSED | Yes as infrastructure; stop advertising F8-natural and CuratorKIT |
| 4 | AgentTune integration (Nov 3–5) | Use the track toolkit end to end | `TrlAgenticGrpo` runs training; AgentTune Cohere fallback prompt + parser shared by teacher/train/eval; own batched lockstep vLLM engine (41 s / 32 rollouts) | G0 step time; 200 steps ≈ 2.3 h | AgentTune's rollout loop bypassed in batched mode; distill trainer rejected (format mismatch); RFT uses raw TRL | VERIFIED, real but partial | Yes; describe honestly (trainer yes, rollout loop no) |
| 5 | RFT / DPO / GRPO ladder (Nov 4–6) | Verifier as curation signal; shared-init GRPO isolates reward | RFT v1/v2 (teacher + 50% lean-expert demos), DPO, curriculum band (214 tasks), 9 GRPO arms × ≤2 seeds | Table §4.1 below | Sampling-seed bug (Amendment 3) invalidated early diversity claims; teacher never re-reads after a success ack (0/3,200, LOGGED-ONLY) | VERIFIED | **Yes, the core** |
| 6 | Reward-source experiments (Nov 5–9; A4 H10/H11) | Information *source* of the reward, not its form, decides verification | `rewards/ack_outcome.py`; tasks 340–342 | ack_outcome s0/s1: test_iid 0.605/0.605, val 0.65/0.64, ood 0.68/0.685 (VERIFIED on cluster); sampled re-check rates (user-quoted 10.1% / 15.3% / 24.6% / 71.3%, computable from `h11/*.jsonl` attempts log; not in any summary file: LOGGED-ONLY) | I-term leaks state via receipts in 0/1,200 dev episodes (D-A5-1) | VERIFIED result, **unpublished** | **Yes, headline** |
| 7 | Middleware comparison (Nov 7–9; A4 H15) | Training beats inference-time read-after-write / verify-before-retry | `eval/a4_runtime.py` RAW/VBR; 72 eval cells run (`runs.jsonl` X7_*) | init+RAW 0.61 (= tool_trust level), init+VBR 0.57; pcstr_d none 0.85 → RAW 0.71, CMR 0 → 0.11, UC 0 → 15, PWV_refund 0.98 → 0.59 (VERIFIED on cluster) | Middleware shifts the observation distribution the RL policy was trained on; harness reads also *cause* collateral writes in the init (e_heldout CMR 0.485) | VERIFIED, unpublished | Yes, as the "middleware is not a substitute" section |
| 8 | Amendment 4 (Nov 7) | Honest-failure (UR) regime, held-out tool, distributed-fault regime B, exploit suite | `generator_a4.py` pools A/B (400 worlds, 15% UR), 5 eval splits, `tools_a4.apply_store_credit`, `exploits.py` (25 attacks), CMR/RHR | S2: every attack below oracle under pcstr_d; CMR tool_trust 0.44/0.30, llm_judge 0.63, state arms 0.00 (VERIFIED). **X5/X6 (training on A/B) never run.** | — | Infra VERIFIED; H12–H14 UNTESTED | **Yes: the X5/X6-style run is the recommended new experiment** |
| 9 | Proof-carrying recovery (Nov 7–10; A5 draft) | RL can teach evidence-aware recovery + verifiable certificates | `tools_a5.py` (op registry, lifecycle, lagging ledger, external billing bot, `wait`, `uncertain`), `verifier_a5.py` (deterministic certificate check), `a5_rewards.py` (T_src + 0.5K), 11 families, obs-equivalent pair | Init 0.991, ack s0 1.00, ack s1 0.982, state s1 0.991, untrained 0.00, **prompt-only evidence-contract 0.994** (VERIFIED on cluster) | Ceiling: demos on 50% of families suffice; GRPO comparison uninformative; `settle()` skips two external-actor rules; ack arm penalises correct keyed retry as "duplicate" | VERIFIED-at-ceiling | **Demote to "environment + verifier contribution"**; do not spend GPU on a5lo before §8's experiment |
| 10 | Amendment 5 (Nov 7) | Register A5 before GPU | PREREGISTRATION_A5.md, hashes | — | D-A5-3 code change after registration (not re-hashed); D-A5-4 post-hoc 10% rerun | Registered | Keep as disclosed |
| 11 | Evidence provenance (A5) | Agent can attribute effects to actors | actor field in registry; `attribution_correct` metric | 1.0 for every trained model, 0.0 untrained | Attribution is read directly from `list_operations` text, so correctness is copying, not inference | VERIFIED-trivial | Supporting only |
| 12 | Certificate verification (A5) | Certificates checkable without trusting the LM | `verifier_a5._supports_committed`: cited evidence must be a model-issued evidence-tool call naming that op as committed; acks never count; adversarial tests | cert_valid 1.0 for arms; RAW middleware drops cert_valid to 0.448 and unsupported_completion to 0.539 | Harness-injected reads are not "model-issued", so certificates citing them fail → a design interaction, not a model failure | VERIFIED | Supporting; useful for the demo's verifier panel |
| 13 | Mechanistic interpretability (Nov 4–10) | Transcript rewards train "knowing liars" decodable by probes/SAE | `interp/extract, probe, text_baseline, h7_compare, guard, knowing_liar`, new `rs_*` plan (M2–M4) | Probe ≈ TF-IDF (all CIs include 0); SAE FVU 3.91 (invalid); pos-A probe failed F1 control (0.62); guard H5 failed (+11% calls, VSC flat); 1,587 prefixes extracted, task 360 pending | Everything decodable was in the text; the guard showed **response, not detection, is the bottleneck** | Mostly negative, VERIFIED | **Demote** to one honest paragraph plus M4 "decodable but ignored" on `e_unrecoverable` prefixes if task 360 runs |

### 2.1 Dependency map (which components support which claims)

```
env/db + faults + ledger + verifier ──► VSC, FCR, DSR, IVR ──► every headline number
        │                                   │
        ├── tools_ecom (hash-frozen) ───────┼──► leaderboard, judge audit (D1), GRPO arms
        ├── tools_a4 (+store credit) ───────┼──► e_heldout transfer, e_unrecoverable HFR, middleware
        └── tools_a5 (+op registry) ────────┴──► proof_valid, certificates (ceiling)
rewards/{baselines,ack_outcome,pcstr} + train/grpo + rollout/batched_vllm ──► reward-source result (H10 + v2 table)
eval/a4_runtime ──► H15 middleware;  eval/exploits ──► H16, CMR/RHR
interp/* ──► negative results only (H7, H5); rs_* ──► nothing yet
demo/app.py ──► replays of v1/v2 only; Live tab misconfigured by default; no A4/A5 content
```

### 2.2 Component table

| Component | Implementation status | Evidence | Research value | Strategic decision |
|---|---|---|---|---|
| Fault environment + ledger + verifier (ecom) | Complete, tested, hash-frozen | 118 tests; sanity matrix; exploit suite | High (ground truth) | Keep; it is the credibility anchor |
| Reward arms (tool_trust, llm_judge, outcome, ack_outcome, pcstr/d/c) | Complete | 9 arms × ≤2 seeds | **Highest** | Make H10/ack_outcome the headline control |
| A4 pools A/B + 5 eval splits + held-out tool | Complete, registered, hashed | Oracle 100%; exclusion audit | High, **unused for training** | Run the §8 experiment on `a4_train_B` |
| Middleware RAW/VBR | Complete | 72 cells on cluster | Medium-high | Publish as "middleware is not a substitute" |
| A5 lifecycle env + certificate verifier | Complete, draft-labelled | At ceiling | Medium (environment), low (RL result) | Demote; use its verifier in the demo |
| Interp pipeline | Complete; SAE invalid | Negative | Low-medium | One honest paragraph; optional M4 |
| AuditKIT bridge | Complete, **never run in pipeline**, no outputs | — | Compliance | Run once on final eval files; otherwise stop claiming "AgentTune → AuditKIT chain" |
| CuratorKIT dedup | Used once; output unused | 9 texts | None | Drop from claims or wire F8-natural in (not worth GPU) |
| Demo Space | Live tab defaults to base-vs-base; assets dir missing; replays v1-mislabelled, cherry-picked | — | Presentation | Rebuild around §14 storyboard |
| Confirmatory pipeline (task 400) | Ready, held until Nov 11 | Dry run H1 +0.0125 [0.004, 0.024] | Rigor | Must run |
| ROI | Measured throughput; API prices unverified | LOGGED-ONLY | Compliance | Verify prices; keep one table |

**Overengineered:** A5 (11 families, external actor, logical clock, certificates) relative to what it has shown. **Underutilised:** A4 pools and `e_unrecoverable`; the ack_outcome result; the middleware cells. **Dead ends:** SAE features; probe guard; knowing-liar via probes; F8-natural; banking transfer (negative, but keep as a limitation). **Unexplored infrastructure:** UR-regime training; regime B; `pcstr_d_noreport` ablation. **Strongest assets:** deterministic verifier + receipts; shared-init design; the H10 control; the 150/150 honest-failure zero. **Surprisingly differentiated:** judge-fooling audit including the toolkit's own judge (AuditKIT TaskCompletion accepts 95% of false completions).

---

## 3. Implementation audit (what is real)

Summary of the code audit (full detail in agent report; file:line cited).

**Toolkit integration, honestly stated**
- AgentTune: `TrlAgenticGrpo` is the real GRPO trainer (`train/grpo.py:65,95-133`); `create_agentic_trainer` for DPO (`train/dpo.py:35`); Cohere fallback prompt + `extract_tool_calls_from_text` parser shared across teacher/train/eval (`rollout/episode.py:23-25`). The batched rollout path (used by every reported run) **replaces** AgentTune's tool loop with `rollout/batched_vllm.py`. RFT uses plain `trl.SFTTrainer` (`train/rft.py:34`). VERIFIED.
- AuditKIT: `eval/auditkit_bridge.py` builds `AgentCase` + `FinalStateAssertion` from stored episodes, but **has zero importers, zero slurm references, and no `results/auditkit/`** in the repo (a `results/auditkit` directory exists on the cluster; contents not inspected). The judge audit borrows `TaskCompletion`'s prompt/parser via private methods (`eval/judge_audit.py:81-82`). VERIFIED.
- CuratorKIT: `ExactDeduplicator`/`MinHashDeduplicator` in `bench/naturalistic_acks.py:45-46` only; the 9 surviving texts are never loaded by any split. VERIFIED.

**Correctness flags worth fixing or disclosing (none invalidates the headline)**
1. `tools_a5.settle()` commits pending ops without the external-actor stand-down check and never starts not-yet-started external jobs (`env/tools_a5.py:258-263` vs `:96-107`). Hidden truth for `external_during` depends on report timing. Disclose.
2. A5 ack arm counts two success acks with distinct op_ids as a duplicate (`a5_rewards.py:51-53`), so correct keyed retry after a phantom is penalised. Intended misalignment, but say so.
3. `ack_outcome` I-term from receipts leaks state (documented; 0/1,200).
4. `eval/confirmatory.py:131` hard-codes H7 as not supported; `:127` IndexErrors if the rft_v2 run is missing.
5. `verifier_a4.py` is a copy of `verify`, guarded by one equality test.
6. Demo: `ADAPTERS` default empty → Live tab is base-vs-base (`demo/app.py:36-37`); `demo/assets/` missing; replays cherry-pick disagreements first (`scripts/build_replays.py:5,44`) and label tool_trust **v1**.
7. `llm_judge.py:23-24` vs `:51`: base prompt asks for justification, suffix demands one word.
8. Untracked: `results/eval/r1/`, `results/a4/CODE_COMMIT`. Commit or ignore deliberately.

**Dead or unused modules:** `eval/auditkit_bridge.py`, `eval/footprint_report.py`, `interp/h7_compare.py` (CLI-only, results logged), `bench/data/f8_natural.json`.

---

## 4. Experimental evidence audit

### 4.1 Consolidated results (VERIFIED unless marked)

**Round v2, shared init `rft_tiny_v2`, greedy, neutral prompt, ≤12 turns** (`results/eval/r3`, `results/a4/prior_contrasts.md`; ack_outcome from cluster `r3`):

| Arm (reward sees) | val | test_iid | test_ood_fault | post-ack refund check (test_iid) | CMR (dev, n=400) |
|---|---|---|---|---|---|
| init RFT v2 | 0.59 | 0.555 | 0.685 | 0.00 | 0.018 |
| tool_trust s0/s1 (acks + report) | 0.65/0.64 | 0.61/0.605 | 0.665/0.675 | 0.00 | 0.438/0.295 |
| llm_judge s0 (transcript) | 0.64 | 0.61 | 0.68 | 0.00 | 0.625 |
| **ack_outcome s0/s1 (acks, report-blind)** | 0.65/0.64 | **0.605/0.605** | 0.68/0.685 | (sampled 10.1%, user-quoted, LOGGED-ONLY) | — |
| outcome s0/s1 (DB, report-blind) | 0.86/0.86 | 0.815/0.835 | 0.815/0.81 | 0.99 | 0.000 |
| pcstr v1 s0 | 0.87 | 0.85 | 0.81 | 0.98 | 0.000 |
| pcstr_d s0/s1 (DB + receipts + report) | 0.87/0.87 | 0.85/0.85 | 0.815/0.81 | 0.98/0.99 | 0.000 |
| pcstr_c s0 | 0.87 | 0.85 | 0.805 | 0.60 | 0.000 |

Pooled contrasts (n=400, hierarchical bootstrap): pcstr_d − tool_trust **+19.25 pp [15.4, 23.3]**; outcome − tool_trust +18.0 [13.9, 22.3]; pcstr_d − outcome +1.25 [0.4, 2.4] (all from 7+3 underclaims in outcome); llm_judge − tool_trust n.s. The "+24.2 pp" in the abstract is test_iid only and is already flagged for withdrawal by D-A4-2.

**Sampled (T=1 × 8, val, n=800) H11** (cluster `results/eval/a4/h11/*.summary.json`): VSC init 0.649; ack_outcome 0.651/0.649; tool_trust 0.665; outcome 0.776; pcstr_d 0.78. FCR init 0.872 → ack_outcome 0.963, tool_trust 0.958 (transcript rewards *raise* false completion in sampling), outcome 0.882, pcstr_d 0.876.

**Middleware H15** (cluster `results/eval/a4/mw`, X7, one seed):

| Model | none | RAW | VBR | RAW+VBR |
|---|---|---|---|---|
| init, test_iid | 0.555 | 0.61 | 0.57 | 0.615 |
| init, e_heldout | 0.385 (CMR 0) | 0.475 (**CMR 0.485**, DSR 0.035) | 0.40 | 0.48 |
| pcstr_d s0, test_iid | **0.85** (CMR 0) | **0.71** (CMR 0.11, UC 15, PWV_refund 0.59) | 0.85 | 0.71 |
| pcstr_d s0, test_ood_fault | 0.815 | 0.605 (SI 8) | 0.815 | 0.605 |
| outcome s0, test_iid | 0.815 | 0.76 (CMR 0.12) | 0.815 | 0.76 |
| tool_trust s0, e_heldout | 0.23 (**CMR 0.84, DSR 0.15**) | 0.33 | — | — |

**Held-out tool `apply_store_credit` (e_heldout, RP n=150 + OK n=50):** pcstr_d 0.555 (RP 0.413; **PWV on the never-trained credit tool 0.73**); outcome 0.56; init 0.385 (PWV_credit 0.00); tool_trust 0.23. INFERRED: state-reward verification *partially* transfers to a structurally similar unseen write (near transfer, same read tools); the transcript policy becomes actively dangerous OOD.

**Unrecoverable tasks (e_unrecoverable, n=150, persistent phantom):** every model, every middleware: VSC 0.0, FS 150/150, HFR 0.0. pcstr_d verifies the refund 100% of the time, sees no refund, and reports "completed". VERIFIED. This is the project's most important unpublished number.

**A5** (cluster `results/eval/a5`, n=330): untrained 0.00; init (50% demos) 0.991; ack s0 1.00; ack s1 0.982; state s1 0.991; state s0 crashed; **init + evidence-contract prompt 0.994**; init + RAW 0.40 (cert_valid 0.448); init + VBR 0.903. Per-scenario evidence-call counts are identical across arms and init (e.g. 3.0 on never_commit, 5.0 on phantom): INFERRED the GRPO arms did not change the policy.

**Leaderboard, judge audit, transfer, interp, ROI:** as in SPEC §27 and the article; judge audit and leaderboard VERIFIED in repo; transfer (HI 0.775 / Hinglish 0.86 / pressure 0.85 / banking ≤ 0.06 with IVR 0.88), probe-vs-text, footprint USER, ROI are LOGGED-ONLY (files on cluster `results/eval/transfer`, `results/interp`, `results/roi`; not inspected individually).

### 4.2 The four questions asked

**H10, does reward source change learned verification?** Yes, as far as this design can show. ack_outcome and outcome have identical form (binary, report-blind, same I-term), identical init/band/seeds/config (`runs.jsonl` X3, commit f2cdea3), and differ by 21–23 pp in VSC and by ~0 vs 0.99 in post-ack checks. Remaining alternative explanations: (a) **verification has zero marginal value and positive turn cost under any ack reward**, so extinction is the rational optimum rather than a side effect: this *is* the source effect, but the write-up should say "the ack-reading reward makes verification worthless, so RL prunes it" rather than "RL unlearns verification"; (b) group-variance asymmetry: ack_outcome pays 1 on most F1/F3 rollouts so many groups are zero-variance; report `frac_reward_zero_std` per arm (available in GRPO logs; not inspected); (c) I-term leak, 0/1,200, negligible. The user-quoted sampled re-check rates (init 24.6%, ack 10.1%, tool_trust 15.3%, outcome 71.3%) are consistent with the h11 summaries but I could not find them in any summary file; they must be recomputed from `h11/*.jsonl` `attempts` before publication.

**H15, what do the middleware results establish?** (i) Read-after-write on the init reaches exactly the transcript-reward level (0.61): extra evidence in context does not change conclusions, consistent with the guard result. (ii) Middleware on top of an RL policy **hurts** (0.85 → 0.71) because harness text is out of the training distribution: PWV_refund falls to 0.59 (the agent stops its own check when it sees an automatic one, then mis-handles the result), 15 underclaims appear, and CMR rises to 0.11. (iii) Harness reads *cause* collateral writes in the untrained-by-RL init (e_heldout CMR 0.485). (iv) It does **not** establish that middleware is useless in general: one seed, two middleware variants, RAW's effect predicate and mapping are the authors' own; VBR is harmless and could be combined with training. Collateral traces were not inspected here; `a4_metrics` can list the offending (op, entity) receipts per episode.

**A5 ceiling, which interpretation?** Demonstration sufficiency dominates: the 50%-demo init already does the evidence-tool pattern (`get_operation` → report), and the evidence-contract prompt alone also reaches 0.994, so the task is prompt- and demo-solvable; GRPO has no gradient. Task simplicity contributes (per-family evidence-call counts are fixed integers, suggesting scripted-like behaviour). Benchmark leakage: no evidence (train/eval families overlap by design; held-out families are hidden-truth variants with identical observations). Distributional diversity: low (3 prompt wordings). The a5lo rerun (10% demos) may restore headroom but then primarily measures RFT→GRPO improvement, not the reward source, unless the ack arm fails to learn: a replication of H10 in a richer environment. Worth running only after §8.

**Interp, what can be claimed?** Only negatives and one design lesson: (1) failure information decodable from activations ≈ information in the visible text (H7 not supported); (2) the web-text SAE does not reconstruct agent transcripts (FVU 3.9); (3) the probe guard raises checks tenfold with no VSC change, so *response* is the bottleneck; (4) the pos-A probe failed its F1 negative control, so earlier "the model noticed" claims are withdrawn. Nothing about circuits, directions or causal roles can be claimed. The frozen INTERP_PLAN M2–M4 is methodologically sound (shuffled labels, random-direction controls, monotonicity rule) but unrun; task 360 is queued.

### 4.3 Claim–evidence matrix (abbreviated; full in agent report)

| Claim as currently written | Status | Action |
|---|---|---|
| Transcript rewards 0.61 vs state 0.815–0.85 | VERIFIED | Keep; add ack_outcome row |
| +24.2 pp [18.5, 30.2] | LOGGED-ONLY, split-selected | Replace with pooled +19.25 [15.4, 23.3] (D-A4-2) |
| PCSTR-d − outcome +2.5 pp | LOGGED-ONLY; pooled +1.25 | Say "+1.25 pp, all underclaim avoidance" |
| tool_trust "removes" the init's ~30% checks | UNSUPPORTED until H11 numbers are recomputed | Recompute from h11 jsonl or drop |
| Seed 1 within 1.5 pp | CONTRADICTED (outcome 2.0 pp) | Fix |
| 32B never re-reads after success (0/3,200) | LOGGED-ONLY, confounded by seed bug | Qualify or drop |
| Judges accept 76–95%, 100% on F8 | VERIFIED | Keep |
| HI/Hinglish/pressure transfer | LOGGED-ONLY | Fetch artifacts into repo |
| Banking negative | LOGGED-ONLY | Fetch; keep as limitation |
| Probe ≈ text | LOGGED-ONLY | Fetch `results/interp/h7_*.json` |
| AuditKIT chain | Bridge never run | Run bridge once on final files or reword |
| H1 confirmatory | PENDING (task 400 held to Nov 11) | Must run |

### 4.4 Preregistered ledger (status)
H1 PENDING (dry run would pass, +1.25 pp, 2% discordant pairs); H2 PENDING/trending null (init FCR at ceiling); H3 PENDING; H4 banking FAILED, ood-fault pending; H5 FAILED; H6 supported (logged); H7 NOT SUPPORTED; H8 trivially passed, memorised prior; H9 null; D1 VERIFIED; H10 VERIFIED on cluster (positive); H11 artifacts exist, numbers to recompute; H12/H13 UNTESTED; H14 NOT RUN; H15 VERIFIED on cluster (training > middleware, one seed); H16 VERIFIED; H5-1 UNINFORMATIVE (ceiling); H5-2..7 effectively uninformative.

### 4.5 Gaps
No artifact in the repo for H10, H11, H15, A5, transfer, interp, ROI (all on cluster). Confirmatory not run. A5 code changed post-registration without re-hash (disclosed). `results/eval/r1/` untracked. Article contains two numbers contradicted by artifacts (seed spread; init column mixing val and test).

