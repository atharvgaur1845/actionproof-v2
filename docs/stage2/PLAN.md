# ActionProof Stage 2: discovery and plan (2026-11-20)

## Context

ActionProof (Stage 1, submitted Nov 15) showed in one simulator that a GRPO reward which reads database state makes a 3.35B agent read back its writes, while a reward of identical form that reads tool acknowledgements removes that behaviour. It also showed that the learned behaviour is a fragile script, and that LLM judges given the evidence still accept most false completions. The team is shortlisted. Stage 2 asks for a larger, more important contribution: an execution-integrity layer for autonomous economic agents, backed by stronger evidence, and shown in a video judged by organisers and industry experts on **Nov 25 17:29 IST**.

**User decisions (2026-11-19/20):**
- **Research first.** The GRPO replication in a second environment and the core experiments come first. The product is the experimental apparatus and the demo: an SDK, a CLI and a minimal dashboard.
- **Human labels:** one annotator (you), 60 episodes.
- **Video:** you narrate and record; Claude writes the script and slides and drives the demo commands.
- **Models:** Aya only.
- **HF credits:** up to $200, tracked.

**Revision 2 (2026-11-20 01:30)** applies the external review of revision 1 (rated 7.8/10; feasibility 6.5, identification 7.0):
- The work is reorganised around one question with three layers (policy, contract, runtime) and three claims.
- **A12 runs ≥ 3 paired seeds,** analysed with per-seed results and a hierarchical bootstrap.
- **H1 uses a frozen three-way fault classification and four gateway conditions,** which separate enforcement from information given to the agent.
- **The verifier is validated** against independent golden traces and mutation tests.
- **Guarantees are stated precisely:**
  - freshness uses commit versions, not timestamps;
  - no exactly-once claim;
  - compensation is a separate operation that can fail;
  - the hash chain gives tamper detection relative to external anchors;
  - a lying authoritative source yields AMBIGUOUS and escalation.
- **Added:** a lifecycle model check, concurrency scenarios, fault combinations, three product modes (Observe, Enforce, Enforce+Inform) compared on the same run, and a compliance checklist that declares pre-Stage-2 work.

**Revision 3 (2026-11-20 02:00)** applies the second external review (rated 8.4/10) and its five mandatory changes:
1. **Reward identifiability.**
   - A **reward-truth table** is a registration artefact, computed before training.
   - The primary endpoint is **phantom-success read-back** (Case A).
   - Error-acknowledgement recovery (Case B) is a separate diagnostic.
   - Scoring rules for multiple attempts, missing acknowledgements and duplicates are fixed in advance.
2. **Protected confirmatory splits.**
   - Step 200 is the default model.
   - Intermediate checkpoints are evaluated on dev/val only.
   - Checkpoint hashes are frozen, then each frozen model is run exactly once per confirmatory split.
3. **Multi-axis fault taxonomy.**
   - Four axes: state observability, contract coverage, recoverability, agent observability.
   - H1 becomes three estimands (Δ_policy, Δ_runtime, Δ_information) with no directional prediction.
4. **Seeds.**
   - 3 is the minimum and 5 the target.
   - Hierarchical intervals are descriptive only.
   - Per-seed diagnostics are added.
   - Seeds 0–2 run on the cluster; HF is used only after a full-path smoke test.
5. **Contract coverage generalization (A13).**
   - Uses independent adversarial faults, clause interactions and held-out combinations.
   - H14 is demoted to a consistency check.

**Also in revision 3:**
- **Framing.** The research paper (C1, reward-information observability) is framed separately from the product (an execution verification and recovery layer, not ACID). The three-layer question is no longer the headline, because of LIMBO.
- **Process boundary.** The gateway runs as a separate process.
- **Compensation failure** is shown as unresolved.
- **H4** is labelled between subjects.
- **Naming.** "Enforce+Inform" is used everywhere.
- **Overhead.** Live overhead is benchmarked.
- **ProcureSim** is described as "separately implemented", not "independent".
- **Effort split:** 45/30/15/10.
- **Traceability.** A `claims/claims.yaml` claim-to-evidence ledger and a benchmark card are added.

> Mode A (discovery and planning). Nothing implemented. Sources: three read-only discovery agents (repo audit, Stage-2 brief and resources, literature) and two read-only planning agents on Opus 5.5 (Fable 5.1 was unavailable: "requires usage credits"; the user chose Opus planners, logged as a deviation from the "Fable plans" rule). User decisions on 2026-11-19: deadline Nov 25; no brief, infer the rubric; Aya-only core; HF credits up to $200, tracked.

## A. Stage-2 challenge analysis

**Official facts** (hackathon.lexsi.ai, single page, read 2026-11-19):
- Final submission **Nov 24 23:59 AoE = Nov 25 17:29 IST**. The page also says Nov 22 in other places; the user confirmed the 25th.
- Shortlisted teams are **judged on a video presentation or slides** by a panel of organisers, partners and industry experts. **No rubric is published.**
- Track 3 checklist: **task success, tool-call accuracy, average latency**, and **how often any LLM judge agrees with human labels**. AgentTune to post-train for reliable tool use; AuditKIT to evaluate tool-call accuracy and reliability.
- Core submission uses the listed Cohere models (Tiny Aya 3.35B variants, Aya Expanse 8B/32B). Other models only as teacher, validator or judge.
- $200 HF credits for shortlisted teams "to improve and polish their submission and artifacts, or to run any extra experiments".

**What this implies for the plan.**
1. The judged artefact is a 5 to 8 minute video plus slides. Everything we build must be visible and understandable in that format. The live system must be shown working on cases it has not seen.
2. The judges are industry experts, not only researchers. The product story ("why should anyone deploy this") weighs as much as the statistics.
3. The Track 3 checklist metrics (task success, tool-call accuracy, latency, judge–human agreement) must appear explicitly, measured through AuditKIT where possible. We have never measured judge–human agreement: this needs human labels (the user labels a sample) and is cheap.
4. Five days. Every item must be judged by how much it changes the video.

**Write-up requirements reported by the external review.** These are not yet confirmed by our own fetch; M0 checks them against the live page. The review says the write-up guidance asks for:
- training configurations, seeds and ablations;
- confidence intervals;
- negative results;
- compute costs;
- datasets;
- reproduction instructions;
- a declaration of work done outside the hackathon.

All of these are in the M7 compliance checklist.

**Unknowns** (user to resolve if possible): video length limit; whether repo and HF artefacts may be updated after Nov 15 (page implies yes); exact HF credit balance.

## B. Repository audit (HEAD 2795c06)

**Reusable core.**
- One write chokepoint with deterministic fault injection: `env/tools_ecom.py:142-189`.
- A split between attempted and committed writes: `env/ledger.py` (Receipt, Attempt).
- A verifier that reads only database state: `env/verifier.py`.
- The A5 environment (`env/tools_a5.py`):
  - an operations registry (op_id, actor, pending/committed/failed, commit_at);
  - a logical clock;
  - late and lost commits;
  - external actors;
  - `get_operation`/`list_operations`/`wait`;
  - an `uncertain` report status;
  - certificates.
- Outcome classes: `env/verifier_a5.py`.
- Read-after-write and verify-before-retry middleware: `eval/a4_runtime.py:53-87`.
- Scripted policies: `rewards/sanity_agents*.py`.
- The GRPO arms and the AgentTune rollout path.
- An independent reference verifier: `scripts/reference_verifier.py`.

**Absent or partial for a product:**
- no adapter to real tools;
- the ledger is an in-memory list with no hash chain, signature or persistence;
- policy is recorded but not enforced (a violating write still commits);
- inconsistent idempotency semantics (v1 and A5 use a raw key, v2 a scoped key; D-ENV-1);
- no recovery or compensation engine;
- a fixed 7-predicate contract language with no schema validation (pydantic is declared but unused);
- three copies of the verifier and of `_write` (the freeze forbids editing originals);
- no SDK, REST API or CLI entry points, no logging, no CI, lint or type check;
- `/scratch/pabitra` paths in 17 files;
- 454 MiB of tracked result JSONL plus 166 MB untracked;
- the dashboard is a static research site.

**Claims check.** All 8 headline numbers trace to files. Two precision notes to fix in the docs:
- The abstract's judge "72% vs 95%" compares different denominators. The paired figure on the same 298 episodes is 0.960 → 0.715.
- "State-trained agents" in the v2 stress test: PCSTR-d 0.176 is one seed; outcome 0.118.

## C. Research landscape (summary; full matrix saved to `docs/research/stage2_related_work.md` in M0)

**Already solved (do not claim):**
- duplicate suppression with provider idempotency keys (Stripe, ACP, LIMBO guard);
- intent and authorization proofs (AP2 mandates, Shared Payment Tokens, Mandato);
- policy gating before a call (AgentSpec, Invariant, NeMo, Cordon);
- staging and compensation of irreversible effects (Atomix, Cordon, Janus);
- tamper-evident signed logs (Notarized Agents, Janus, AIREP, Tool Receipts);
- state-graded benchmarks (AppWorld, tau2).

**Partly solved:**
- postconditions checked against authoritative state, with hand-written predicates (Verified Tool Calls 2608.02645, Stale 2609.29522);
- reconciliation of ambiguous outcomes (Idempotency Is Not Recovery 2611.23353; EGR 2610.41453, which never reads state);
- outcome taxonomies (EGR, AIREP, 2611.23353).

**Toy scale only:**
- EGR (37 assignments);
- Janus (1 model);
- Verified Tool Calls (2 workflows);
- 2611.23353 (frozen responses);
- Notarized Agents (microbenchmarks).

**Fails under realistic faults:**
- late commits (LIMBO Proposition 1: no verification-only policy is exactly-once);
- compensation leaking irreversible effects (Atomix);
- gates that trust declared facts pay wrong amounts (Janus);
- blind retry and idempotency keys both give 0% recovery on mid-write failures (UndoBench 2610.05622);
- silent failures are detected far less often than loud ones (58.8% vs 91.3%, 2610.10062).

**Open gaps that ActionProof can own:**
1. **Training the actor against an integrity layer** under faults whose acknowledgements look exactly like success. Every system above is runtime only. LIMBO names learned, cost-aware verification as future work.
2. **Phantom success acknowledgements.** These are absent from the LIMBO, Atomix and UndoBench taxonomies.
3. **Deterministic state verifier vs an LLM judge given the evidence**, on economic writes. We already have 0.960 → 0.715.
4. **Verification budgeted by money at risk.**
5. **Contract completeness**: postconditions are hand-written, and a missing clause leaks a fault family.

**Scoop risks:**

| Paper | Threat | Our response |
|---|---|---|
| LIMBO 2609.29095 | The contract explains 81% of variance and the model 8% | Position learning only where the model matters: resolvable and observation-equivalent faults |
| Idempotency Is Not Recovery 2611.23353 | Its full configuration is close to our gateway plus ledger plus reconciler | It has no live or trained policy and no lying acknowledgements |
| EGR | Its contracts and recovery actions are close to ours | It never reads authoritative state and works at tiny scale |
| UndoBench | Recovery benchmark | Report its CRSR metric so results are comparable |
| Cordon, Atomix | These set the systems-venue bar for a gateway | We do not claim a gateway as novel |

**Consequence:** the gateway, ledger, idempotency and recovery taxonomy are **engineering we must build well but cannot claim as novel**. The novelty must come from (1), (3), (4) and (5), and from joining them to our existing reward-channel result.

## D. Research framing (revision 3)

**The research and the product are framed separately.**

**Research:** *Reward-Information Observability and Evidence Acquisition in Tool-Using RL.*
- The novelty is a controlled, training-time manipulation of what the reward can see, under phantom writes that look identical to real ones.
- It is replicated in a second, separately implemented procurement environment with five independent simulated services.
- The mechanism is predicted from a reward-truth table before training.
- We also measure where learned evidence acquisition fails to turn into recovery.

**Product:** *ActionProof: Verifiable Execution Infrastructure for Agentic Workflows.*
- An **execution verification and recovery layer**, not a distributed transaction manager. It gives no atomicity across services.
- Components: contracts, authoritative-state verification, bounded recovery, and evidence provenance.

**Why the three-layer question is not the headline.** LIMBO ("Where Does Exactly-Once Live?") already varies model, harness and tool contract over 25,930 episodes. 2611.23353 and Cordon already cover the runtime. The policy / contract / runtime split stays only as the frame for evaluating the product.

**Claims** (each is reported however it comes out):
- **C1, novelty (P0): reward observability.**
  - On writes that return a success acknowledgement without committing, a verdict-grounded reward selects authoritative read-back, and an acknowledgement-grounded reward of identical form does not.
  - The sign is the same in every paired seed (minimum 3, target 5).
  - The per-class mechanism is predicted before training.
  - Measured too: where acquisition fails to integrate into recovery ("read, then ignore").
- **C2, measurement (P1): contract coverage.** Which violations a contract catches; which it misses when one clause is removed; which it misses through clause interactions; and which it cannot express at all. Tested on faults generated independently of the contract.
- **C3, measurement (P0): policy–runtime contributions.** We *estimate*, without presupposing a direction, three quantities per gateway mode and fault cell:
  - Δ_policy, the learned policy's contribution;
  - Δ_runtime, the deterministic runtime's contribution;
  - Δ_information, the extra value of exposing evidence to the agent.

| Layer | Question | Primary (P0/P1) | Supporting |
|---|---|---|---|
| Policy | Does a verdict-grounded reward select evidence acquisition where an acknowledgement-grounded reward cannot? | **A12** (H2; seeds ≥ 3, target 5), **H3** | H8 recovery choice; H16 (P3) |
| Contract | What can a contract establish, miss, or not express? | **H5** coverage generalization (A13) | H6 (P3); H9 (P2); H14 (consistency check only) |
| Runtime | What does enforcement add, and what does exposing evidence add? | **H1** estimands (A11); **H7** | H10; H12 (P2, deferred if P0 slips); H11 |
| Cross-cutting | Does execution evidence improve evaluator decisions? | **H4** audit (Track 3 item) | — |

**Hypotheses** (revision 3 changes: H2's primary endpoint, H3 with a reward-truth table, H1 as estimands, H5 as coverage generalization, H4 between-subjects, H14 demoted):

| # | Hypothesis | Primary metric | Falsified if | Role |
|---|---|---|---|---|
| H2 | **Case A (primary): writes whose first acknowledgement is success but which did not commit.** On untouched `proc_conf`, **phantom read-back** is ≥ 15 pp higher under R_V than R_A, with the same sign in every paired seed. Secondary: verified completion gap ≥ 10 pp. Check: within each arm, read-back after benign success acknowledgements equals read-back after phantom ones (observation equivalence). **Case B (separate diagnostic, error acknowledgements):** lost-ack read-back, precommit recovery and false success, reported per arm with no directional claim | phantom read-back | Hierarchical CI includes 0, or the sign differs in any seed | **P0** |
| H3 | Before step 1 we commit (i) the **reward-truth table** (§G3) and (ii) per-class predictions for the sign of verification's relative advantage under each reward, from step-0 gate samples. Predictions: under R_A, negative in lost-ack and never-commit classes, slightly negative (turn cost only) on phantom; under R_V, positive on phantom and timeout before commit. Training logs (per-class group advantages, informative-group counts) and per-class read-back changes are then compared with the predictions | sign agreement per class | Signs disagree in > 20% of classes, or read-back changes where no signal was predicted | **P0** |
| H1 | Estimation over 4 modes × 7 policies × fault cells on a frozen multi-axis taxonomy (§G1). Δ_policy(m,c) = VSC(π_trained,m,c) − VSC(π_warm,m,c); Δ_runtime(π,c) = VSC(π,Enforce,c) − VSC(π,Observe,c); Δ_information(π,c) = VSC(π,Enforce+Inform,c) − VSC(π,Enforce,c). Registered two-sided test: does the runtime change the policy's contribution? That is, Δ_policy(Enforce,c) − Δ_policy(Passthrough,c) ≠ 0, pooled over full-observability cells. Engineering check: Δ_runtime > 0 on full-observability, safe-recovery cells | the three estimands, with CIs, per cell | No directional prediction. If the engineering check fails, the runtime does not work as designed and is reported as such | **P0** |
| H5 | **Coverage generalization**, on three fault groups: G-rep (what the contract represents), G-comb (combinations of known primitives), and G-adv (adversarial faults written by an independent agent from transaction semantics alone, without seeing the contract). Measured: clause necessity (single removal); redundancy; pairwise interactions (pair leak > sum of single leaks); composition generalization (G-comb); specification gap (G-adv violations no clause can express). Prediction: full-contract false acceptance on G-adv > on G-rep (one-sided) | false acceptance per group; interaction table | False acceptance on G-adv ≤ on G-rep | **P1** |
| H7 | Idempotency keys alone recover 0% of phantom and partial writes. No verification-only policy is exactly-once under unbounded late commits (LIMBO Prop. 1). Keys plus reconciliation leave at most one irreversible effect under stated provider assumptions. Same fault traces for every defence | recovery; duplicates; residual exposure | Keys recover ≥ 30% of phantoms | **P1** |
| H4 | Single-annotator **audit**, between-subjects to avoid carryover. 60 stratified episodes (20 invisible failures, 20 visible failures, 20 true completions), randomly split 10/10 within each stratum into transcript-only and ledger-visible, each labelled once. Prediction: transcript-only false acceptance on invisible failures ≥ 0.5; ledger-visible ≤ 0.2. Judge (TaskCompletion; ledger-instructed 32B) vs human per condition, with confusion matrices, raw agreement and κ with CIs | false acceptance per condition × stratum; κ | Transcript-only false acceptance on invisible failures < 0.3 | **P1** |
| H12 | VOI(a) = E[L \| no verification] − E[L \| a] − C(a). Six policies: never, always, fixed rate, random at equal cost, amount threshold, expected risk. Pareto frontier; raw exposure reported separately; sensitivity over loss weights | frontier | Expected-risk is no better than random at equal cost | P2 (deferred if P0 slips) |
| H9 | Local verification gives ≥ 90% correct writes but ≤ 70% cross-service invariants under partial or cross-service faults; cross-service clauses restore ≥ 95% | invariant satisfaction | Local already ≥ 95% global | P2 |
| H8 | Trained agents' recovery ignores what their read showed; a deterministic engine reaches ≥ 0.9 recovery on resolvable cases and ≥ 0.9 escalation recall on unresolvable ones | recovery; recall | Engine < 0.9 | supporting |
| H10 | Stale replicas and forged receipts in non-authoritative output are caught by version freshness and corroboration. A lying authoritative source without corroboration gives AMBIGUOUS and escalation, which is reported as a limit | false finalization | Finalizes success on forged evidence | supporting |
| H11 | Exposure-weighted loss changes at least one policy ranking (with weight sensitivity) | expected loss; raw exposure | Rankings identical | supporting |
| H14 | Remaining false commitment under report gating vs the H5 leak union | — | — | **engineering consistency check, not a claim** (it may follow algebraically from the shared scorer) |
| H6, H16 | Aya-drafted contracts; transfer to a held-out service | — | — | P3 |
| H13, H15, H17 | — | — | — | deferred |

## E. Product strategy

| | A. Execution verification and recovery gateway | B. Proof-carrying execution | C. Recovery infrastructure | D. Multi-party commerce trust |
|---|---|---|---|---|
| Feasible in 5 days | high | medium | medium-high | low |
| Novelty | medium; not claimed (Cordon, Atomix, Janus, 2611.23353) | high in principle | low | high |
| Commercial relevance | high | medium | high, only inside A | high, early |
| Video potential | high | low | medium | slides only |
| Fit with research | direct | inverse (A5 certificates failed) | needs A | none |

**Recommendation: A**, with C as its recovery engine and gateway-issued (not agent-issued) proofs. D is deferred.
- **Initial user:** a platform engineer putting an LLM agent in front of purchase-to-pay systems.
- **Value:** the agent can act but cannot declare success. Transactions finalize only on verified effects. Unknown outcomes are reconciled or escalated, never retried blindly. Every money movement has an evidence trail.
- **Display rule:** a committed irreversible payment whose compensation failed is shown as *unresolved* with residual exposure, never as success.

**Three modes.** The names are the same in the product and in the experiment.

| Mode | Gateway | What the agent receives |
|---|---|---|
| **Observe** | Verifies and logs; never changes execution | Unchanged tool responses |
| **Enforce** | Enforces grants and limits; blocks unsafe downstream calls and false finalization; runs recovery rules | On a blocked call, an ordinary in-schema refusal `{"status":"error","error":"operation_not_permitted"}` with no verification detail. Recovery effects become visible only through the agent's own later reads |
| **Enforce+Inform** | Enforce, plus structured evidence | The acknowledgement is replaced in schema by `{"status":"error","error":"not_committed","verdict":<state class>,"evidence_id":<id>,"observed":<authoritative projection row(s)>}`, or by `"status":"ok","verdict":"VERIFIED_SUCCESS"`. Same tool schema, no free text |

Passthrough (no gateway) is the baseline condition of the experiment only.

The demo and evaluation page show **one table generated from real executions**: the same task, model, seed and injected fault in each mode. Rows are false finalization, duplicate payment, verified completion, recovery, tool calls, latency and exposure. **Live** Observe-mode overhead is benchmarked separately.

**Scope under research first:**
- an SDK client;
- the gateway as a **separate local process** (FastAPI);
- the CLI;
- a dashboard with three views: the three-lane timeline, the fault playground, and mode comparison plus evaluation.

## F. Proposed architecture

```
 Agent (Tiny Aya + LoRA | Aya Expanse 8B/32B prompt-only | scripted reference policies)
   |  tool calls (same Cohere JSON format and parser as training)
   v
 SDK Transaction.tools()  (duck-types the env: call / tools / done, so rollout/episode.py runs unchanged)
   v
 GATEWAY (sole owner of service credentials; no bypass path):
   intercept -> bind contract (immutable hash, fail-closed) -> policy and risk gate (incl. shared limits across transactions)
   -> idempotency manager (op+entity scoped keys) -> executor -> op tracker
   |                                                | events (actor-attributed)
   v                                                v
 ServiceAdapters (sim | ecom)              EVIDENCE LEDGER (append-only SQLite, SHA-256 hash chain, heads anchored externally)
   | writes        ^ reads (separate read path)     ^                                  -> SSE -> REST, CLI, dashboard
   v               |                                |
 Systems of record (5 ProcureSim services)   VERIFIER (reads versioned projections only; never acks or agent text) -> Verdict
                                                    v
                                             RECOVERY ENGINE (decision table; compensation tracked as its own operation)
```

**Process boundary and interfaces** (revision 3).
- **Demo and product.** The gateway runs as a **separate local process** (FastAPI). It alone holds the service credentials, contracts, policy and verification. The agent process holds only a scoped gateway client token and reaches services only through the gateway. The SDK is a thin HTTP client: `Gateway(url, token).transaction(contract).tools()` returns the same tool names and signatures, so the agent loop is unchanged.
- **Training and evaluation rollouts.** For throughput these use an **in-process** adapter with the same verifier code. This path is labelled "research harness, not a security boundary".
- **What we claim.** No production-grade security from the in-process wrapper, and no prevention of a second network route an operator leaves open. The demonstrated boundary is "one local process to another".
- **Interfaces:**
  - REST: `POST /v1/transactions`, `POST /v1/transactions/{tx}/calls`, `POST /v1/transactions/{tx}/finalize`, `GET /v1/transactions[/{tx}/events]`, `GET /v1/stream` (SSE), `GET /v1/ledger/verify`, and test mode `POST /v1/test/runs`.
  - CLI: `actionproof serve|run|ledger verify|anchor|tamper|contract validate|eval|label`.

**Modules.** All new code goes in new files. Frozen files are only imported.
- `actionproof/integrity/`:
  - `models.py`: pydantic v2; exports the JSON Schema.
  - `contracts.py`, `catalog.py`, `statemachine.py`, `gateway.py`, `policy.py`, `risk.py`, `idempotency.py`, `verifier.py`, `recovery.py`, `ledger.py`, `cli.py`, `logging.py`.
  - `adapters/{base,sim,ecom}.py`.
  - `sim/{clock,service,supplier,oms,payments,inventory,accounting,faults,world,scenarios,concurrency,truth}.py`.
  - `api/`, `dashboard/` (reuses the `demo/theme.css` tokens), `contracts/*.toml`.
- Research side:
  - `bench/generator_proc.py`, `rewards/proc_rewards.py`, `train/proc_demos.py`.
  - `train/grpo_proc.py`: rebinds `make_env`/`episode_record`, as `eval/run_eval_a4.py:60-61` does.
  - `scripts/{freeze_a11,freeze_a12,classify_faults,s2_eval,s2_stats,s2_report,s2_numcheck,reference_verifier_proc}.py`.
  - `tests/integrity/`, `tests/golden/`.
  - `docs/stage2/lifecycle.tla` (if a TLA+ checker can be used; see M4).

**Data models.** Money is integer minor units plus currency.
- **`Contract`**:
  - identity and versioning: `contract_id, version, content_hash` (immutable once bound);
  - parties and intent: `principal, agent_id, intent_text` (shown to the agent, never used in decisions);
  - scope and limits: `grants, targets, limits{per_op_max, total_max, max_irreversible, shared_budget_id}`;
  - conditions: `preconditions, postconditions: [Predicate], allowed_effects, forbidden_effects, commit_points`;
  - policies: `idempotency{scope, key_template, retention_s}`, `recovery{max_retries, settle_wait_max_s, allow_compensation, escalate_to}`, `risk{approval_above, block_above}`, `expires_at`.
  - **Fail-closed:** a call not covered by a grant, or an unknown predicate, means rejection, never allowance.
- **`Predicate`**: `row_exists | field_eq | count_eq | sum_le | same_value`, evaluated on versioned read projections. This replaces the 7 fixed predicates of `env/verifier.py:57-74`.
- **`OperationSpec`**:
  - `service, op, kind, args_schema, entity_arg, money_arg`;
  - idempotency: `idempotent (none|by_key|natural_key)`, `key_param`, `key_retention_s`;
  - retry and undo: `retry_safe (never|after_verified_absent|always)`, `reversible`, `rollback_op`, `compensator{op, arg_map}`, `irreversible`;
  - observation: `read_projection`, `status_endpoint`, `settle_bound_s`, **`consistency (linearizable|read_your_writes|eventual(bound))`**.
- **`Evidence`**:
  - `source{service, read_op, authoritative|replica}`, `query`, `observed`;
  - **`as_of_version`**: the service's monotonic commit sequence at read time;
  - **`fresh`**: `as_of_version ≥ dispatch_version`, the high-water mark returned when the write was dispatched;
  - `supports (committed|absent|partial|unknown)`.
  - Wall-clock time is only shown, never used as the freshness test. Each adapter declares its consistency assumption in the catalog.
- **`Verdict`**:
  - state axis: `VERIFIED_SUCCESS | VERIFIED_FAILURE | PARTIAL | COMPENSATED | AMBIGUOUS | POLICY_VIOLATION`;
  - claim axis: `SUPPORTED | UNVERIFIED_ASSERTION | CONTRADICTED_CLAIM`;
  - `residual_exposure_minor`, `reasons`, `ledger_head`.
  - A compensated transaction keeps both the original and the compensating effect. It is **never** classified as if the original had not happened.
- **`RecoveryDecision`**: `SAFE_RETRY | NOOP_ALREADY_COMMITTED | WAIT_AND_PROBE | RECONCILE | ROLLBACK (only where the provider truly supports undo before commit) | COMPENSATE | ESCALATE | TERMINAL_FAILURE`.
- **`RiskAssessment`**: `exposure_minor` (irreversible money committed without verified goals plus worst-case pending), `reversible_share`, `in_doubt_count`, `tier`.
- **`LedgerEvent`**:
  - fields: `seq, tx_id, op_id, kind, actor, t_logical, t_wall, body, prev_hash, hash`;
  - `hash = sha256(prev_hash || canonical_json(event))`;
  - **anchoring:** `actionproof ledger anchor` appends `(seq, head)` to a separate append-only anchor log, which can be pushed to git or HF.
  - **Claim:** tamper *detection* of modification, deletion, reordering and truncation *after the last anchor*, relative to a trusted anchor store. Not tamper-proof storage. No signatures claimed.

**Operation lifecycle** (an explicit table; illegal transitions raise; checked by model-based tests):
```
PROPOSED -> REJECTED | AWAITING_APPROVAL -> AUTHORIZED -> DISPATCHED -> ACK_SUCCESS | ACK_ERROR | ACK_ACCEPTED | NO_RESPONSE
any ack -> VERIFYING -> COMMITTED | ABSENT | PARTIAL | IN_DOUBT
IN_DOUBT -> (wait, probe, settle bound) -> COMMITTED | ABSENT | ESCALATED
ABSENT -> RETRYING (same key, only if retry_safe) -> DISPATCHED | FAILED_TERMINAL
PARTIAL -> RECONCILING -> COMMITTED | ESCALATED
COMMITTED -> COMPENSATION_REQUESTED -> COMPENSATION_COMMITTED | COMPENSATION_FAILED -> ESCALATED   (residual exposure recorded)
```

**Recovery decision table.** Retry-safe and idempotent are different properties.

| Case | Evidence (versioned) | Action |
|---|---|---|
| Phantom ack | ack success; fresh probe shows absent | discrepancy; SAFE_RETRY with same key |
| Timeout before commit | ack error; fresh probe absent after settle bound, or status failed | SAFE_RETRY with same key |
| Timeout after commit (lost ack) | ack error; fresh probe committed | NOOP_ALREADY_COMMITTED; suppress the agent's retry |
| Late commit, bounded | pending; bound not reached | WAIT_AND_PROBE |
| Late commit, unbounded or never | pending; no bound | no resend; ESCALATE with exposure |
| Partial | some sub-effects present | RECONCILE; otherwise COMPENSATE the committed part (tracked as its own op) |
| Duplicate | ≥ 2 irreversible commits for (op, entity, tx) | COMPENSATE the extra one; POLICY_VIOLATION |
| Stale read | `as_of_version` < `dispatch_version`, or source is a replica | read the authoritative source or wait; never conclude |
| Cross-service mismatch | captured, no journal entry | PARTIAL; RECONCILE the system that lags |
| Untrustworthy evidence | authoritative read contradicts a corroborating service | AMBIGUOUS; ESCALATE (never finalize) |
| Compensation failed | compensator ack error or absent | COMPENSATION_FAILED; ESCALATE with residual exposure |

**Guarantees, stated precisely** (these wordings go into `docs/stage2/PRODUCT.md`):
- **At most one committed irreversible effect per (op, entity, tx)**, holding only under stated provider assumptions:
  - the key is scoped to (op, entity);
  - the key is retained for at least the transaction horizon;
  - no concurrent in-flight call shares the key;
  - after the key expires, the gateway escalates and does not retry.
- **No exactly-once claim.** Under unbounded late commits, verification alone cannot give it (LIMBO Prop. 1), and the gateway escalates.
- **No false finalization:** the gateway never reports VERIFIED_SUCCESS without fresh authoritative evidence for every required effect.
- **Limits:**
  - we do not claim to detect a lying system of record unless it is corroborated;
  - **no atomicity across services:** it verifies after execution and controls later operations; it is not a distributed transaction manager, and Cordon's staged-effect outbox is the comparison;
  - a failed compensation leaves the transaction unresolved, with its residual exposure;
  - no liveness guarantee;
  - no cross-organisation non-repudiation.

**Threat model.**

| Threat | Covered | Not guaranteed |
|---|---|---|
| Lying tool | acks are never evidence; separate read path | lying system of record without corroboration (AMBIGUOUS, then escalate) |
| Malicious or injected agent | fail-closed grants, limits, shared budgets; credentials only in the separate gateway process; payee bound to master data; no decision reads free text | allowed but unwise actions within limits; credentials or network routes outside the gateway; the in-process research harness is not a boundary |
| Replayed responses | evidence bound to op_id and dispatch version | |
| Ledger tampering | hash chain plus external anchors | changes before the first anchor if the anchor store is compromised |
| Stale reads | version-based freshness | sources whose consistency is misdeclared in the catalog |
| Concurrent transactions | reservations against shared inventory and budgets; serialized commit points | cross-gateway races (one gateway assumed) |

**ProcureSim** (one simulator for both the product and GRPO).
- **How we describe it:** "a separately implemented procurement task environment with multiple independent simulated services". It is **not** an independent external environment, because it shares the ActionProof codebase and lifecycle ideas. Generalization to other systems needs a later experiment on an independently developed backend.
- **What it is:**
- Each service has its own SQLite database and a **monotonic commit sequence**.
- One logical clock, ported from `env/tools_a5.py:86-132`.
- A per-service ops registry with the A5 lifecycle.
- Faults live in the transport: `FaultPlan[(service, op, k)]` with ok, phantom, partial, lost_ack, precommit_timeout, late_commit(delay), never_commit, stale_ack (replay), replica_lag. It supports **combinations and timing**: several faults per transaction, at chosen call indices and logical delays.
- Deterministic per-service RNG.
- `sim/truth.py` scores after a hidden-horizon `settle()` and shares no code with `integrity/verifier.py`.
- **`sim/concurrency.py`**: deterministic interleavings of two transactions (seeded schedule), in four scenarios:
  1. two agents buying the last stock;
  2. two refunds against one order;
  3. capture while another agent edits the PO;
  4. two POs that each fit a shared budget but together exceed it.

| Service | Writes (properties) | Authoritative read |
|---|---|---|
| supplier (external, **no** status endpoint) | `place_order` (idempotent by key; rollback `cancel_order` before shipment) | `find_order(po_ref)` |
| oms | `create_po` (natural key), `approve_po` (limit-gated), `close_po` (commit point) | `get_po` |
| payments | `authorize` (idempotent; rollback `void`), `capture` (irreversible; compensator `refund`, itself an operation that can fail) | `list_payments(po_id)` |
| inventory | `record_goods_receipt` (**not idempotent**; compensator `reverse_receipt`) | `list_receipts(po_id)` |
| accounting | `post_entry` (keyed; compensator `post_reversal`) | `get_entry` (primary); `list_entries` (lagging replica) |

**Constraints so GRPO can run on ProcureSim:**
- at most 14 agent tools;
- the oracle finishes in at most 16 calls;
- prompt plus schemas at most about 3k tokens (`batched_vllm` caps context at 7,168);
- phantom and real acks are byte-identical;
- prompts never depend on the fault.

**Task families and held-out sets:**
- Families: clean, phantom_supplier, precommit_timeout_capture, lost_ack_receipt, late_bounded_capture, never_commit_capture, partial_capture_oms, gl_replica_lag, cross_service_gl_missing, over_limit, payee_injection, and **two-fault combinations**.
- Held out of training and development: never_commit_capture, cross_service_gl_missing, payee_injection, faults on `accounting.post_entry`, and a set of unseen two-fault combinations.

## G. Research roadmap (revision 3)

Four pre-registrations, each registered before any run of its family:

| Registration | Covers | When |
|---|---|---|
| **A11** | evaluation factorial in the e-commerce environment (H1, H4, H7, judge) | Fri Nov 20 |
| **A12** | procurement training (H2, H3) | by Sat 14:00, gated |
| **A13** | contract coverage (H5) | Mon 09:00, before H5 runs |
| *(no registration)* | H8 to H12 | exploratory CPU, labelled as such |

### G1. Multi-axis fault taxonomy (frozen in A11 and A12 before any evaluation)

`scripts/classify_faults.py` labels each task algorithmically from its contract, its catalog projections and its fault plan. It never looks at outcomes. The labels and the script hash are frozen.

| Axis | Values | Question it answers |
|---|---|---|
| State observability | Full / Partial / None | Can the actual execution state of every required effect be determined from authoritative, fresh projections within the horizon? |
| Contract coverage | Complete / Incomplete | Does the contract state the relevant invariant? |
| Recoverability | Safe / Conditional / Impossible | Is there a valid recovery action? Conditional means it depends on provider semantics or a bound. |
| Agent observability | Gateway-only / Agent-visible | Can the policy see the evidence? Set by the mode: Agent-visible only in Enforce+Inform or through the agent's own reads. |

**Examples:**

| Situation | State | Contract | Recovery | Agent sees |
|---|---|---|---|---|
| Timeout before payment commit | Full | Complete | Safe, under the key semantics | depends on mode |
| Supplier op in flight, no status endpoint | None within horizon | Complete | Unresolved | no authoritative evidence |
| Unrecoverable refund | Full | Complete | Impossible | depends on mode |

A contract can be complete about a property and still be unable to establish whether that property holds. Cells that are empty in a given environment are reported as empty. The None-observability cells are tested on ProcureSim with scripted and prompt-only agents.

### G2. A11: evaluation family (e-commerce v2)

- **Splits.** New untouched splits, built through runtime entries to `generator.SPLITS` (`bench/generator.py` is not edited):
  - `test_iid_c2`: 200 tasks;
  - `v2_stress_c2`: 100 timeouts before commit.
- **H1 factorial: 4 modes × 7 policies.**
  - Modes: Passthrough, Observe, Enforce, Enforce+Inform. In Observe the system outcome equals Passthrough's, and detection is computed offline on the Passthrough trajectories; live Observe overhead is benchmarked separately.
  - Policies: warm start; PCSTR-d s0 and s1; ack_outcome s0 and s1; tool_trust s0; Aya 32B prompt-only.
  - Agent-level metrics (report, read-back, the agent's own retries) and system-level metrics (final state and finalization) are reported separately.
- **Estimands.** Δ_policy, Δ_runtime and Δ_information, per cell, as defined in §D. Each gets a paired task bootstrap CI. For policies with 2 seeds, every seed is reported separately.
- **Primary test.** The two-sided interaction test in §D. **Engineering check:** Δ_runtime > 0 on full-observability, safe-recovery cells.
- **H4 protocol.**
  - Random allocation, stratified 10/10, between subjects.
  - `actionproof label` hides the judge's verdict and the truth.
  - A second annotator labels the same items independently, if one becomes available.
- **H7.** Deterministic tables on shared fault traces (v2 and A5), plus the ledger-instructed 32B judge.

### G3. A12: procurement training (gated by Sat Nov 21 14:00)

**Readiness gate:**
- the oracle solves 100% of every split;
- byte-identical acknowledgements (a test);
- the reward unit tests reproduce the reward-truth table;
- split hashes recorded.

**Reward-truth table** (a mandatory registration artefact; computed by code on hand-built representative trajectories, *before* training, and never edited to force a finding):
- **Rows** are (fault class) × (agent behaviour). Fault classes: clean; phantom success; partial; lost-ack (committed, error ack); precommit timeout; bounded late; never commits; duplicate delivery. Behaviours:
  - proceed without reading;
  - read then proceed;
  - read, then retry with the same key;
  - blind retry with the same key;
  - blind retry with a new key;
  - read, then stop and report failure;
  - read, then escalate.
- **Columns:**
  - R_V;
  - R_A;
  - the turn cost;
  - the implied sign of verification's relative advantage under each reward.
- **Scoring rules, fixed in advance:**
  - A required operation (op, entity) maps to every attempt with that (op, entity).
  - R_A uses the **last acknowledgement** among those attempts. A missing acknowledgement (no response) counts as not success.
  - R_V = 1[every required (op, entity) is VERIFIED_SUCCESS in the final state **and** no duplicate irreversible effect].
  - Duplicates are visible to R_V (state channel) and invisible to R_A (acknowledgement channel). This is part of the channel difference and is stated as such.
  - Neither reward reads the report. Neither has a state-derived mask.

**Primary endpoints.**
- **Case A (primary):** phantom read-back. On a write whose first acknowledgement is success but which did not commit, the agent makes a fresh authoritative read of the entity before its next write or its report.
- **Secondary:** verified completion.
- **Case B diagnostics** (reported, not tested as primary):
  - lost-ack read-back;
  - precommit recovery (read, then retry after an error acknowledgement and verified absence);
  - false success rate;
  - duplicate rate.

**Splits** (fresh seeds, disjoint ids):
- `proc_demo` (SFT only);
- `proc_train` 400;
- `proc_dev` 200 and `proc_val` 100, for development and model selection only;
- **confirmatory, untouched:** `proc_conf` 300, `proc_combo` 100 (unseen two-fault combinations), `proc_stress` 100;
- `proc_heldout_service` 100 (P3).

**Checkpoint protocol.**
- **Default model:** the final step-200 checkpoint of each seed.
- Intermediate checkpoints (steps 50, 100, 150) are evaluated **only** on `proc_dev`/`proc_val`, for diagnostics.
- No selection by any metric unless a development-only rule is registered in A12. None is planned.
- **Checkpoint freeze:** adapter sha256 values are committed before any confirmatory run.
- **Confirmatory evaluation:** each frozen model is run **exactly once** per confirmatory split. Re-runs are forbidden; a crashed run is resumed from its saved per-task file, never repeated.

**Warm start and gate.**
- Scripted-oracle demonstrations, built like `train/a6_demos.py`, with 30% verifying demonstrators and three report templates.
- Gate (T = 1 × 4, 100 `proc_train` tasks):
  - phantom read-back in [0.10, 0.60];
  - verified completion in [0.25, 0.75];
  - valid calls ≥ 0.95;
  - predicted non-zero-variance groups ≥ 30% under both rewards.
- One re-mix is allowed; if it fails again, stop and report.

**Training:** identical to A6/A7 (200 × 4 × 8, lr 2e-5, β 0.02, T = 1, ≤ 16 turns, LoRA r16), with the rollout log on.

**Seeds.**
- **Minimum 3 paired seeds (0, 1, 2); target 5.**
- Paired means the same warm start, the same seed index in both arms, and identical tasks.
- Seeds 3 and 4 start under a registered rule: only if they can finish, including the confirmatory run, by Mon Nov 23 12:00.
- Seeds 0–2 run on the cluster's two H200s. HF Jobs are used only after the full remote training path has passed a smoke test.

**Statistics.**
- Per seed: paired task bootstrap (10k) and exact McNemar.
- Between-seed SD.
- The exact paired sign-flip test over seeds. Its minimum one-sided p is 1/8 at 3 seeds and 1/32 at 5, and it is reported as such.
- A two-level hierarchical bootstrap is reported as **descriptive** only. With 3 seed clusters it says little about training variability in the population.
- Episodes are never pooled as if they were independent policies.
- Primary family, Holm over 2: phantom read-back R_V − R_A, and verified completion R_V − R_A.
- **The key result is a consistent change in behaviour across seeds,** not the mean reward.

**Training diagnostics** (all saved and plotted per seed):
- reward curves;
- read-back curves on `proc_dev` checkpoints;
- per-class reward variance;
- informative-group counts and per-class group advantages (H3);
- loss and KL;
- checkpoint stability;
- rollout-level action distributions.

**Optional K1** (P3): R_A on the pool without the predicted classes, only after seeds 0–2 are complete.

### G4. A13: contract coverage (registered Mon 09:00)

**Fault groups on ProcureSim:**
- **G-rep:** faults the contract represents.
- **G-comb:** combinations of known primitives.
- **G-adv:** adversarial violations of the intended transaction semantics, written by an **independent agent**. It gets only a plain description of the semantics, never the contract, catalog or verifier. Examples include wrong payee, amount rounding, split payments, entity aliasing, out-of-order commits, partial line items, stale master data, and over-receipt.

**Measurements:**
- clause necessity (single removal);
- redundancy;
- pairwise interactions;
- composition generalization;
- the specification gap (G-adv violations that no clause can express, listed one by one).

**Primary test:** full-contract false acceptance on G-adv is greater than on G-rep (one-sided, Wilson CIs).

A descriptive table of clause necessity on existing e-commerce episodes is included, labelled "engineering validation". H14 is a consistency check only.

### G5. Verifier validation (truth is not assumed)

1. **Independent implementation.** `reference_verifier_proc.py` is written by a separate agent from `docs/stage2/VERIFIER_SPEC.md` alone.
2. **Independent golden traces.** At least 60 hand-specified transactions with expected verdicts, written by a *different* separate agent. You spot-check 15. They cover every fault class and boundary:
   - missing, duplicated and reordered events;
   - wrong entity;
   - wrong amount;
   - partial writes;
   - delayed commits;
   - illegal transitions;
   - collateral writes;
   - compensation failure;
   - stale evidence.
3. **Mutation testing.** At least 20 mutants; we report the kill rate.
4. **Random histories.** Hypothesis-generated valid and invalid histories; both verifiers must agree.
5. **What we report:** soundness (false VERIFIED_SUCCESS, target 0), completeness, and agreement on every confirmatory episode.

The claim is "agreement on independently specified cases", never "ground truth".

### G6. Metrics

Definitions go in `docs/stage2/METRICS.md` and in the benchmark card. Agent-level and system-level values are reported separately where they differ.

| Metric | Definition | Failure it catches |
|---|---|---|
| **Phantom read-back** (primary) | Fresh authoritative read of the entity after an initial success acknowledgement on a write that did not commit, before the next write or report | not looking where only state can tell |
| Benign read-back | The same, after a real success acknowledgement (observation-equivalence check) | — |
| Lost-ack read-back; precommit recovery | Case B diagnostics (§G3) | blind retry; duplicates |
| Verified completion | Postconditions and invariants hold, no duplicate irreversible effect, every required write committed, status matches | false success |
| Strict completion | Verified completion and no committed write outside the required set | extra writes |
| False finalization / commitment | P(completed reported or finalized \| goal not met), conditional and absolute | lying outcomes |
| Unauthorized side effect; collateral mutation | Committed op outside grant, scope or limit; an in-scope write that was not required | policy breach; extra writes |
| Evidence precision and use | Useful reads ÷ reads; P(correct next action \| read shows a discrepancy) | read spam; ignoring evidence |
| Recovery success (CRSR-comparable) | Over recoverable faults: goal met, no duplicate, correct status | not recovering |
| Duplicate transactions | Irreversible (op, entity) with ≥ 2 commits, plus the amount | paying twice |
| Raw exposure; residual exposure | Unweighted irreversible amount without verified goals, plus pending; the uncompensated amount after compensation attempts | money at risk |
| Expected economic loss | Weighted sum with sensitivity over a weight grid | money harm |
| Time to detect; escalation accuracy; verification cost | as in revision 2 | late detection; silence or refusal; over-verification |
| Latency | Batch-1 p50/p95 per episode; live gateway overhead per write (Observe and Enforce) | operational cost |
| **Track 3** | Task success (verified next to claimed); tool-call validity and argument correctness; average and p50/p95 latency; judge–human agreement (H4) | checklist |

### G7. Claims and kill criteria

| Claim | Made if | Otherwise |
|---|---|---|
| C1. Reward observability replicates | Phantom read-back R_V − R_A has the same sign in every paired seed; per-seed CIs exclude 0 in at least 2 of 3 seeds (or 4 of 5); the descriptive hierarchical CI excludes 0; and H3 signs agree in ≥ 80% of classes | Headline negative: "did not replicate", kept and reported |
| C2. Contract coverage | The G-adv vs G-rep test holds; the specification gap and the interaction table are reported | Reported as measured, with no generalization claim |
| C3. Policy–runtime contributions | Estimates with CIs for all cells; the interaction test result as it comes out; the engineering check passes | If the engineering check fails, the runtime is reported as not working as designed |
| Supporting (H4, H7, H12) | As measured | As measured |

**Not run:**
- more e-commerce seeds, M5 or M3′;
- A5 reruns;
- grounded-judge GRPO;
- interpretability, DPO, calibration swaps;
- zero-shot e-commerce→procurement;
- Aya 8B policy training;
- markets;
- fault-rate sweeps;
- dependency-aware reward training;
- more leaderboard models;
- new training algorithms.

**Claims to reword** in Stage-2 materials (the Stage-1 abstract is not edited):
1. Strict VSC 0.424 comes from one collateral habit.
2. The reference verifier is an independent implementation of our own specification.
3. Judge: 0.960 → 0.715 on the same 298 episodes, Aya 32B with the unchanged AuditKIT prompt; 8B accepts 1.00.
4. "Selects read-back; recovery incomplete".
5. Scope the reward-source claim to its environment and mechanism.
6. +0.678 comes from one run per cell, with the interval over tasks.
7. Latency is exploratory, on A100.
8. The AuditKIT bridge shows parity, not independence.
9. Scripted results are labelled as such.
10. No "exactly-once", "tamper-proof" or "ACID".
11. ProcureSim is "separately implemented", not "independent".

## H. Execution roadmap (revision 3)

### H.1 Effort allocation and priorities

**Effort:**
- research experiments 45%;
- gateway correctness (verifier, contract, recovery, tests) 30%;
- evaluation and reproducibility (statistics, labels, artefacts, CI) 15%;
- demo and communication 10%.

**No new features beyond this plan.**

| Keep (never cut) | Defer first, in this order |
|---|---|
| A12 with 3 paired seeds; the reward-truth table; H3 predictions | 1. H6 drafted contracts |
| H1 four-mode estimands | 2. K1 |
| H5 coverage generalization (A13) | 3. extra dashboard views |
| H4 audit | 4. H12, if P0 slips |
| verifier validation (§G5) | 5. live 32B endpoint |
| two complete hero scenarios | 6. REST beyond the demo |
| claim-to-evidence ledger and benchmark card | 7. concurrency beyond 2 scenarios (shared budget, last stock) |
| | 8. TLA+ (the Python model check is kept) |
| | 9. seeds 3–4 |

### H.2 Workstreams

Parallel agents produce code faster but add integration risk. Every agent's output is merged only behind tests, and I integrate one workstream at a time.
- **WS-A (critical, main session):** ProcureSim, the generator, oracle and demos, R_V/R_A, the reward-truth table, `grpo_proc.py`, A12.
- **WS-B (worktree agent):** the e-commerce gateway adapter (4 modes), c2 splits, `classify_faults.py`, the A11 draft. I review and register it.
- **WS-C (independent agents with no access to the implementation):** (1) the reference verifier; (2) the golden traces; (3) the G-adv fault set. Each works from its own written input only.
- **WS-D (main session, after WS-A reaches its gate):** the gateway process, recovery and compensation, ledger anchoring, the 2 concurrency scenarios, the model check, CLI, dashboard, CI.

### H.3 Milestones

Each milestone follows the same loop: objective → files → implement → tests → inspect → report works and failed → update STATUS and DECISIONS → next action.

**M0: Bookkeeping, specs, queue** (Fri Nov 20, 01:30–04:30)
- **Docs:** `docs/stage2/` with PLAN, DECISIONS, EXPERIMENTS registry, RISKS, STATUS, METRICS, HF_CREDITS, RELATED_WORK (Appendix Y), HYPOTHESES (§D), **VERIFIER_SPEC.md**, **TRANSACTION_SEMANTICS.md** (the only input for the G-adv agent), COMPLIANCE.md.
- **`claims/claims.yaml` skeleton.** Each claim maps to its pre-registration, training runs, evaluation files, metric code, statistics script, verifier, figures and source commit.
- **Benchmark card skeleton** (`docs/stage2/BENCHMARK_CARD.md`): transaction semantics, fault taxonomy, data generation, split construction, verifier assumptions, metrics, limits, intended and prohibited interpretations.
- **Housekeeping:**
  - archive Appendix Z;
  - update the memory files;
  - untracked results go to the HF traces dataset or `.gitignore` (never deleted);
  - check the official page for the compliance items and confirm the HF credit balance.
- **Workers:** submit **two H200 workers** and one A100 worker. The co-user's jobs share our QOS limits and are never touched.
- **Acceptance:** docs committed; workers in the queue.

**M1: A11 registered, H1 queued** (Fri 04:30–16:00; WS-B, plus CPU work in the main session)
- **Deliverables:**
  - c2 splits;
  - the multi-axis classifier and its frozen labels;
  - the e-commerce adapter with 4 modes (Enforce+Inform payload exactly as in §E);
  - `PREREGISTRATION_A11.md` and `freeze_a11.py`;
  - slurm tasks;
  - CPU: the H7 tables.
- **Acceptance:**
  - `freeze_a11 --check` passes before any A11 job;
  - the gateway verdict matches the frozen `verify()` for every scripted agent;
  - Observe never changes the agent's observations (a byte-equality test).
- **Fallback:** H1 not started by 20:00 → HF A100, after an HF smoke test.

**M2: ProcureSim and A12** (Fri 04:30 – Sat 14:00, WS-A, critical)
- **Deliverables:**
  - `integrity/sim/*` (5 services, commit versions, fault combinations, truth) and `integrity/models.py`;
  - `generator_proc.py`, the oracle, `proc_demos.py`, `proc_rewards.py`, `grpo_proc.py`;
  - **`results/a12/reward_truth_table.md`**, generated by `scripts/a12_reward_truth.py`;
  - tests: byte-identical acks, oracle 100%, determinism, token budget, reward table reproduced, version freshness;
  - `PREREGISTRATION_A12.md` (H3 predictions included) and `freeze_a12.py`.
- **Acceptance:** the gate passes and A12 is registered by Sat 14:00.
- **Fallback:** evaluation-only procurement (H7, H9, H1 cells with None observability using scripted and prompt-only agents). C1 is reported as "not replicated in time".

**M3: Training** (Sat 14:00 – Mon 12:00)
1. Warm start and gate on H200, target done by Sat 19:00.
2. An **HF Jobs full-path smoke test** (warm start load, 5 GRPO steps, adapter save, evaluation on 10 tasks) runs in parallel. HF is used later only if it passes.
3. **Wave 1** (target Sat 20:00): seed 0 of R_V and R_A on the two H200s.
4. **Wave 2**: seed 1. **Wave 3**: seed 2. About 4 h per wave, so seeds 0–2 are done around Sun 08:00.
5. **Waves 4–5**: seeds 3–4 by the registered rule, on the cluster or on HF after the smoke test.
6. Each seed is evaluated on `proc_dev`/`proc_val` at steps 50, 100, 150 and 200.
7. **Checkpoint freeze:** hashes committed.
8. **Confirmatory evaluation**, once per frozen model: `proc_conf`, `proc_combo`, `proc_stress`.
9. Procurement prompt-only baselines (Aya 32B, 8B, Tiny) in the three modes.
10. H4 labelling sessions on Sat and Sun evening (between subjects).
- **Acceptance:** seeds 0–2 in both arms are complete with all diagnostics saved.
- **Stop rule:** more than 80% zero-variance groups at step 50 → stop and report. No reward redesign unless it is pre-registered first.

**M4: Gateway process and correctness** (Sat – Mon, WS-D)
- **Gateway:**
  - FastAPI process; SDK client; contracts (fail-closed, immutable hash);
  - verifier; recovery with compensation states; risk with shared budgets;
  - ledger with anchoring; JSON logs with actor attribution.
- **Concurrency:** 2 scenarios.
- **Lifecycle model check** (Python, exhaustive, bounded) with these invariants:
  - FinalizedSuccess ⇒ every required effect committed;
  - IrreversibleCommit ⇒ authorization satisfied;
  - UnknownOutcome ⇒ no blind retry;
  - DuplicateIrreversible ⇒ PolicyViolation;
  - CompensationFailed ⇒ not displayed as success.
- **Tests:** model-based transition tests, Hypothesis properties, one regression per fault and per combination.
- **A13:** registered Mon 09:00, then the H5 runs.
- **Acceptance:** green tests; the model check is clean, or each violation it finds is fixed and kept as a regression test.

**M5: Analysis, validation, dashboard, CI** (Mon Nov 23)
- **Analysis:** `s2_stats.py` (per seed, sign-flip, descriptive hierarchical, diagnostics plots); the verifier validation report (§G5); `s2_report.py` → `results/s2/`.
- **Claims ledger:** `claims/claims.yaml` filled in, and `s2_numcheck.py` resolves every number in the slides, script, README and site to a claims entry.
- **Dashboard:** the three-lane timeline, the fault playground, and the mode-comparison table generated from live runs. Evaluation is file-backed, and a missing file shows "pending".
- **CI:** ruff, `mypy --strict` on `integrity/`, CPU pytest, a frozen-file guard, a clean-venv install plus CLI run.
- **Acceptance:** CI green; numcheck clean.

**M6: Freeze, slides, video** (Tue Nov 24; freeze 12:00; recording 15:00–21:00)
- **Deliverables:** a 12-slide deck and a 6–7 minute script; you narrate, Claude drives.
- **Script order:**
  1. The problem, and Stage-1 results marked as pre-Stage-2.
  2. **C1** (per-seed results, the reward-truth table, the predicted vs observed mechanism).
  3. The same task, model, seed and fault under Observe / Enforce / Enforce+Inform, with actual system state, recovery and exposure.
  4. Compensation failure shown as unresolved.
  5. Tamper detection.
  6. C2 and C3.
  7. The Track 3 metrics.
  8. Limits.
- **Acceptance:** every claim resolves through `claims.yaml`.

**M7: Compliance and submission** (Wed Nov 25; submit by 12:00 IST)
- **`COMPLIANCE.md`**, each item linked to its artefact:
  - before/after on the same Aya model;
  - task success and verified completion;
  - tool-call validity and argument correctness;
  - average and p50/p95 latency;
  - judge–human agreement with its protocol;
  - hyperparameters and seeds;
  - ablations and CIs;
  - negative results and limits;
  - compute and ROI;
  - public HF model and dataset collections;
  - public GitHub repo with reproduction commands;
  - **a declaration separating work done before Stage 2 (Nov 1–15) from Stage 2 (Nov 20–25)**;
  - an LLM-usage statement and team contribution note;
  - a public demo or repository;
  - the benchmark card.
- Anything public is confirmed with you first.
- Final run of every `freeze_*.py --check`, all tests, and numcheck.

## I. Resource plan

**Capacity:**
- Reliable capacity is the **two cluster H200s** (QOS: 2 GPUs, 2 jobs, no wall limit).
- A100 (2 GPUs) and H100 (1 GPU) share their per-user limits with the co-user's pending jobs.
- HF Jobs are used only after the full-path smoke test.

| Item | GPU-h | Pool (fallback) | Worst-case HF $ |
|---|---|---|---|
| H1 factorial (7 policies × 3 GPU modes × 300 tasks, plus dev smoke) | 7.0 | A100/H200 (HF A100) | 18 |
| Live overhead benchmark and latency | 0.7 | A100 | 2 |
| Ledger-instructed 32B judge and H4 preparation | 1.3 | H200 | 7 |
| Procurement prompt-only baselines (3 models × 3 modes) | 2.0 | A100 | 5 |
| HF full-path smoke test | 0.5 | HF H200 | 3 |
| Warm start and gate | 2.0 | H200 | 10 |
| **GRPO seeds 0–2 × 2 arms** (about 4 h each) | **24.0** | 2 cluster H200 (HF H200) | 0 expected, 60 worst |
| Development-checkpoint and confirmatory evaluation (6 models) | 4.5 | H200 | 0 expected |
| **Core total** | **≈ 42** | | **≈ 105 worst** |
| Seeds 3–4 × 2 arms plus evaluation (target) | 18.0 | cluster H200 (HF H200) | 0–80 |
| Live Aya 32B endpoint (P3) | — | HF Inference Endpoint | 40 |

**HF rules:**
- Expected spend about $10–60.
- Jobs are capped at $150 cumulative, the endpoint at $40, with a hard stop at $195.
- Every spend is logged in `HF_CREDITS.md`.

**Decision tree:**
- **Fri 20:00:** H1 not started → HF A100 (if the smoke test passed).
- **Sat 14:00:** no A12 → M2 fallback.
- **Sat 18:00:** warm start not started → HF H200.
- **Gate fails twice** → stop and report.
- **Sun 08:00:** seeds 0–2 not progressing → move the remaining runs to HF H200.
- **Mon 12:00:** last start of any training or confirmatory run.
- **Tue 12:00:** freeze.

## J. Final recommendation

**Optimise for outcome B:** a reproducible empirical finding plus a credible product, not a large feature set.

**Research contribution (C1, the novelty):** reward-information observability.
- Under phantom writes that look identical to success, a verdict-grounded reward selects authoritative read-back and an acknowledgement-grounded reward of identical form does not.
- Replicated with ≥ 3 paired seeds (target 5) in a separately implemented procurement environment.
- The mechanism is predicted in advance from a reward-truth table and step-0 differentials.
- We also report where learned acquisition fails to become recovery.

**Measurements that make the product honest:**
- **C2, contract coverage:** what a contract establishes, misses and cannot express, on independently generated faults.
- **C3, policy–runtime contributions:** Δ_policy, Δ_runtime and Δ_information per mode and fault cell.

**Product:** ActionProof, verifiable execution infrastructure for agentic workflows. It is a separate-process gateway with:
- fail-closed contracts over versioned projections;
- a verifier validated against independent goldens and mutations;
- recovery that separates retry-safe from idempotent and tracks compensation as its own operation that can fail;
- an anchored hash-chained ledger;
- three modes compared on the same run.

**Strongest demonstration:** the same Aya agent, task, seed and injected supplier phantom under Observe, Enforce and Enforce+Inform. The table comes from real executions, the three-lane timeline shows each step, and the actual system state, recovery and exposure are on screen. Then the per-seed C1 chart, with the predicted vs observed mechanism.

**Top technical risks:**
1. ProcureSim not ready by Sat 14:00. Mitigation: gate, then fallback.
2. A wrong verifier or recovery decision. Mitigation: independent goldens and mutations, the model check, fail-closed defaults.
3. Too much work for 5 days. Mitigation: the effort allocation, the cut order, no new features, and integration behind tests.

**Top scientific rejection risks:**
1. A seed-specific result. Mitigation: paired seeds, sign-flip, per-seed reporting, protected confirmatory splits.
2. The R_A design lets reading pay off through error-acknowledgement retries. Mitigation: phantom-success read-back as the primary endpoint, Case B kept separate, and the reward-truth table registered before training.
3. Overlap with LIMBO, Cordon and 2611.23353. Mitigation: a narrow research claim (C1) and the gateway explicitly not claimed as novel.

**Not built:**
- real payments;
- PKI or blockchain;
- a general policy language;
- distributed or multi-tenant infrastructure;
- markets;
- an LLM in the gateway's decisions;
- a React pipeline;
- elaborate dashboard features;
- refactors of frozen code;
- history rewrites;
- ACID across services.

## Verification (end to end)

1. `pytest -q` (existing tests plus `tests/integrity/` and `tests/golden/`), and every `scripts/freeze_*.py --check`.
2. CI: ruff, mypy on `integrity/`, CPU pytest, the frozen-file guard (`git diff --exit-code 2795c06 -- <hashed paths>`), a clean-venv `pip install .[integrity]` and `actionproof run --scenario procure --seed 1 --agent naive --mode enforce`.
3. `results/a12/reward_truth_table.md` is reproduced by the reward unit tests, and its hash matches A12.
4. The confirmatory-split access log shows exactly one run per frozen model and split.
5. The verifier validation report: goldens, mutation kill rate, random-history agreement, confirmatory agreement, soundness.
6. Lifecycle model check: no violation within the bound.
7. Demo: start the gateway process (`actionproof serve --test-mode`) and the agent as a separate process. Run both hero scenarios with unseen seeds and faults in the three modes, and confirm the generated table, timeline, recovery, the unresolved display after compensation failure, and tamper detection.
8. `s2_stats.py` output shows each seed, the sign-flip p and the descriptive hierarchical CI.
9. `s2_numcheck.py` resolves every published number through `claims/claims.yaml`.

Appendices: related-work matrix in `docs/stage2/RELATED_WORK.md`; the 2026-11-10 research review in `docs/audit/research_review_2026-11-10.md`.
