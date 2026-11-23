# ActionProof research upgrade: audit summary and execution plan

Branch `actionproof-research-upgrade`, cut from `research/e1e2` at b49a581. `stage4` is an ancestor of this branch, so nothing is lost.
Written 2026-11-23 10:55 IST. **Deadline: Nov 27, 2026** (moved from Nov 24 by the author on 2026-11-23; the exact hour is still to be confirmed).

## 1. What already exists (audit, first pass)

| Mandate item | State on 2026-11-23 | Gap |
|---|---|---|
| Claims ledger | `claims/claims.yaml` (652 lines, machine-checked by `scripts/s2_numcheck.py`); `docs/final/FINAL_CLAIMS.md` (17 rows plus a "never say" list) | No single machine-readable *experiment* inventory joining registration, seeds, splits, result file and verdict |
| Preregistrations | 30 `PREREGISTRATION_*.md` files, hashed in `results/*_hashes.json` | None |
| Literature and novelty | `docs/research/NOVELTY_AUDIT.md` (84 verified entries), `NOVELTY_MATRIX.md`, `docs/stage3/LITERATURE.md` | Missing coverage: cost-of-observation RL, POMDP information gathering, risk-sensitive and distributionally robust RL |
| Algorithm candidates | FCA failed its kill criterion; B (constrained GRPO) and C (frontier scheduling) rejected on novelty; C5 and G registered and **running** (`docs/research/algorithm/C5G_DEV.md`) | A third candidate built on observation cost |
| LLM judges | Aya 8B and Aya 32B on the Stage 1 pool (`results/judge/`); H4 waits for 60 human labels | No multi-family committee; no human calibration set; no abstention |
| Capability retention | **None** | Whole workstream |
| ROI | `results/roi/roi.md`: inference cost only (Stage 1, measured throughput) | Loss model, scenarios, sensitivity, the comparison with database checks |
| Paper | `paper/` (EMNLP template, main.pdf), `docs/paper/DRAFT.md` | Lexsi checklist, retention, judges, ROI, traces |
| Website | Static Space (deployed 2026-11-22) | Retention, judges and ROI sections; separation of projection from measurement |
| Toolkits | AgentTune (all training), AuditKIT and CuratorKIT (Stage 1), CircuitKIT failed to install (D-A16-1) | AlignTune and SafeTune not yet checked |

The headline that holds up best, from the existing ledger, is the paired R_V vs R_A contrast: **+0.27 to +0.32 VSC per seed, 5 of 5 seeds**, with read-back 1.00 vs 0.00. ID1 shows the reward *selects* read-back rather than creating it. The best-baseline Stage 1 margin (+1.5 pp) is small and must not be conflated with the +20.25 pp contrast against the tool-trusting reward.

## 2. Capability retention: benchmark selection

The question is whether ProcureSim training damages what Tiny Aya could already do. A benchmark earns a place only if it answers that question for *this* model.

Facts that constrain the choice:
- Tiny Aya Global (`Cohere2ForCausalLM`, 3.35B) has an 8,192-token context.
- Its chat template has no tool support (one `default` template, no `tools` block). ActionProof passes its tools as plain text, so every external tool benchmark must run in prompting mode.
- τ-bench retail's system prompt (policy wiki plus 16 tool schemas) is about 4.8K tokens (character estimate; checked against the tokenizer at the pilot). That leaves about 3.3K tokens for the conversation; one product record is about 600 tokens.

| Benchmark | Measures | Fit to this question | Decision |
|---|---|---|---|
| **BFCL** (Berkeley Function Calling Leaderboard; Apache-2.0) | Tool selection, argument correctness, relevance detection (abstaining when no tool fits); multi-turn stateful categories | **Strongest.** Deterministic (AST and state checks, no LLM judge). About 1,200 non-live single-turn cases give roughly a 1-point paired standard error. Hundreds of unseen function schemas directly test the surface binding seen in S4 (renamed tools). | **Primary.** Non-live AST (simple, multiple, parallel, parallel-multiple) and irrelevance. Multi-turn base only if prompts fit in 8K at the pilot. |
| **τ-bench** (Sierra; MIT; retail 115 test tasks, airline 50) | Policy-following agents that change a database, graded by final database state | **Most similar in construct:** its grader is the same idea as our verifier. **Weak in power:** at 115 tasks and a likely floor-level score for a 3B model, only large changes are detectable. It also needs a user-simulator LLM and risks context overflow. | **Secondary, retail only.** Fixed open-weight user simulator for every arm (not the policy, not a judge model); 4 trials per task, reporting pass^1 and pass^k. A context overflow counts as a failure for every arm, and the overflow rate is reported per arm. If the base model overflows in more than half of pilot episodes, τ-bench is reported as infeasible for this model rather than as a capability number. |
| **IFEval** (541 prompts; Apache-2.0) | Verifiable instruction following | Standard check for capability lost during post-training. Deterministic and cheap. Our training never asks for formatting constraints, so it is independent of our data. | **Primary: general-capability canary.** Prompt-level strict accuracy as the endpoint; instruction-level and loose accuracy reported too. |
| **AgentBench** (8 environments; Apache-2.0) | Broad LLM-as-agent ability (OS, database, knowledge graph, games, web) | **Poor.** Each environment needs heavy Docker infrastructure. Small open models score near zero overall, so a floor hides any regression. Most environments have nothing to do with stateful-write verification. | **Excluded,** with this reason stated in the paper. |

Models evaluated: base (no adapter), the SFT warm start `rft_proc`, A12 R_V and R_A (seeds 0–4), RSC and the class-matched control (seeds 0–4), SR1 (0–2), and every Stage 5 candidate. τ-bench runs only on a pre-chosen subset: base, warm start, R_V s0–2, RSC s0–2.

**Hypothesis registered before any output:** most of any loss comes from the SFT warm start rather than GRPO (β 0.02, LoRA r 16). The test is warm start vs base, and GRPO vs warm start, each as a paired task bootstrap plus a seed-level test. Mitigations (LoRA scaling, mixed-data warm start) are tested only if a regression is found, on development data.

## 3. Workstreams and order

| # | Workstream | First step | Needs from the author |
|---|---|---|---|
| W1 | Machine-readable experiment inventory (`results/inventory.jsonl`) and gap report | Script that joins registrations, hash files, summaries and ledger rows | — |
| W2 | Stage 5: C5 and G kill decisions (results expected this afternoon); a third candidate built on observation cost, under development registration | Apply `scripts/s5_dev.py` exactly as registered | — |
| W3 | Capability retention (section 2) | Register `PREREGISTRATION_R1.md`; feasibility pilot on the base model | — |
| W4 | Judge committee: Gemma, Qwen, plus a third family via HF inference; blinded, randomised order, structured JSON, abstention | Probe provider availability, prices and licences | Approve judge inference spend; label the calibration set (about 90 cases, about 2 h) |
| W5 | ROI: loss model, three scenarios, sensitivity, "why not a few database checks" | Measured frequencies from the result files; cited external costs | — |
| W6 | Paper (Lexsi checklist), website, README, claims sync, CI | After W2–W5 land | Final review |

## 4. Schedule (IST)

- **Mon 23:** W1 inventory; R1 registration and pilot; judge provider probe; C5/G kill decisions as results land.
- **Tue 24:** R1 full runs (IFEval and BFCL on all adapters, τ-bench subset); judge committee on the calibration pool; development runs of the third candidate.
- **Wed 25:** confirmatory runs (5 seeds) for any candidate that survives; human labels; ROI model.
- **Thu 26:** paper, website, claims sync, number check.
- **Fri 27:** freeze, final checks, submit.

## 5. Rules carried over

- Never run a confirmatory split twice.
- Frozen files and registered thresholds stay untouched.
- Results are reported as they land, whatever the sign.
- Database truth outranks every judge.
- Every timestamp comes from `date` in the same command that writes it.
