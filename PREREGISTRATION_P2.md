# PREREGISTRATION P2: observable ledger, deterministic guard and the full arm set

Registered 2026-11-24, before any P2 training or evaluation output. Hashes: `results/p2_hashes.json` (`scripts/freeze_p2.py`). Branch `quality-upgrade`.

**State of knowledge at registration.**
- P1 is registered and frozen (`PREREGISTRATION_P1.md`); its development results are known (`docs/research/P1_DEV_DIAGNOSIS.md`).
- Some P1 frozen-test outputs exist on the cluster. **None has been read**, except each summary's runtime field, which was read to size this study.
- P2's design therefore uses development evidence only.

**Not a new algorithm.** P2 tests an observation format, a runtime gate and known training recipes.

## 1. Question

P1's development data showed that agents read the evidence but fail to act on it. A bare "outstanding" line made one agent top up with the old key: 12 of 14 false successes.

P2 asks two questions:
- Does a structured, fact-only ledger of what the agent has seen help, at inference and when trained with it?
- Does a deterministic gateway that enforces read-before-retry and verified completion make the agents safe?

## 2. Arms

The prompt's arms A–E map to these tags. A, C and D are P1's models; their P1 test outputs are reused unchanged.

| Tag | Model | At inference |
|---|---|---|
| A | RSC seeds 0–2 (P1 arm A) | — |
| B2 | A | ledger |
| C | P1 arm C (A + recovery SFT) | — |
| D | P1 arm D (recovery SFT warm start + RSC recipe) | — |
| **E** | **D's warm-start rows with the ledger, D's SFT settings; then the RSC recipe with ledger rollouts**, seeds 0–2 | ledger |
| Ag, Cg, Dg | A, C, D | guard |
| Eg | E | ledger + guard |
| W | warm start `rft_proc` | — (reference) |
| M | class-matched control `s3_matched_s0–2` (S3 addendum 1) | — (reference) |
| O | scripted oracle (`scripts/p2_oracle.py`) | greedy only (reference) |

**E vs D.**
- Identical SFT targets: 4,961 rows whose completions are byte-identical (sha256 `946f44ef…`).
- The same SFT settings: 2 epochs, lr 1e-4, seed 0, max length 6,144. No row exceeds it: the longest is 5,583 tokens with the ledger, 3,740 without.
- The same RL: R_V; `s3_trainb_half` + `s3_rec_train_half`; 200 steps; 4 × 8 rollouts; lr 2e-5; β 0.02.
- The only difference is the ledger text in every tool result. E's prompts are longer (median 3,399 vs 2,924 tokens), so more of its episodes may end at the context limit; that is reported.

## 3. The ledger (`actionproof/p2/ledger.py`)

A pure function of the conversation text: the request plus the tool results already shown. It never touches the simulator, the database, the fault log or the task metadata.

**What it lists:**
- the PO total and quantity, from the request or a `get_purchase_order` result;
- the last read of payments, receipts, journal and orders, and whether a write came after it ("stale");
- idempotency keys already used, with the payment each acknowledgement named, and which of those payments are visibly refunded;
- writes not read back since.

Anything not read is printed as "unknown". It states **no recommended action**.

It is inserted into every tool-result message, in SFT rows, RL rollouts (`actionproof/p2/rollout.py`, a copy of `stage3/rollout.py` whose only changes are marked and tested) and evaluation.

## 4. The guard (`actionproof/p2/guard.py`)

Deterministic and visible-only. It uses the ledger state, plus the workflow it fronts: the requested steps and the PO total and quantity stated in the request.

| Rule | What it blocks |
|---|---|
| R1 | Read before retry: a write to a record type whose previous write has not been read back |
| R2 | No over-fill: a write when the latest fresh read already meets the PO |
| R3 | No refunded key: re-paying with the key of a visibly refunded payment |
| R4 | Verified completion: "completed" unless a read after each requested step's last write shows it complete |

A blocked call is not executed; the policy sees the guard's reply. Reports of "failed" or "escalated" are never blocked. The guard is not defined on the renamed-tool API, so it is not run there.

## 5. Evaluation (`scripts/p2_eval.py`)

The episode loop is P1's (greedy plus 4 samples at T = 1, seeded per task, sample and turn; 16 turns; 256 tokens; 8,192 context). It adds only the ledger and guard hooks, plus a provenance file per output: git commit, code, split and adapter hashes, decoding settings, and expected vs actual counts.

**Equivalence check (smoke, not analysed).** With both hooks off, P2's loop is compared episode by episode with P1's on `p1_dev_rec` (`scripts/p2_equivalence.py`).

| Phase | Splits | Who |
|---|---|---|
| dev | p1_dev_full, p1_dev_rec | E (in its training task), B2, Ag. Descriptive; no choice is made from it |
| test | the five p1_test_* splits (450 tasks) | B2, Ag, Cg, Dg, E, Eg. Each model and split **once**; the script refuses to overwrite |
| ref | the five test splits | W, M |
| shift | s4_lang_hi, s4_lang_hinglish, s4_api (150 each) | A, C, D, E. Once |

The oracle O runs on the test and language splits.

## 6. Metrics

- **Primary:** VSC from the deterministic verifier.
- **Also, per arm, split and workflow:** false success, unrefunded duplicates, honest failure, short-payment repair, the share of new keys among post-refund re-pays, reads and repeated reads, policy calls per success (blocked calls included), context endings, and guard blocks per episode with their rules. Greedy and sampled are reported separately.

**Secondary, descriptive: the J1 committee** (`evaluation/llm_judge_committee/committee_p2.py`).
- Gemma 4 31B, Qwen3 235B, gpt-oss-120b and DeepSeek V3.2, using J1's v1 rubric, transcript view.
- The sample is the seed-0 greedy episodes of A, C, D, E and Eg on all 450 test tasks (2,250 cases), blinded.
- It reports each judge's accuracy against the verifier, and its acceptance of verifier failures.
- It stops before J1 + P2 spend passes the author's $15. No judge answer changes an outcome.

## 7. Hypotheses and analysis (`scripts/p2_stats.py`, frozen with this file)

**Gate.** If any required output is missing, incomplete or duplicated, the analysis stops (`actionproof/reporting/coverage.py`).

**Primary family, Holm over four:**

| | Contrast | Question |
|---|---|---|
| H1 | E > D | Does the ledger add to the same training? |
| H2 | E > A | Does the full candidate beat the existing agent? |
| H3 | B2 > A | Does the ledger at inference alone help? |
| H4 | Ag > A | Does the guard help the existing agent? |

**Secondary family, Holm over three:** Cg > C, Dg > D, Eg > E.

**Statistics.**
- Sampled VSC on the test union.
- A paired task bootstrap over seed-averaged task scores (B = 10,000, seed 20261125), with a one-sided p.
- **Supported** iff Holm p < 0.05 **and** all three seed-indexed differences are > 0.
- Seeds are paired by index. E's seeds do not descend from A's, so for H2 and H3 the seed condition is a consistency guard, not a pairing.

**Descriptive.**
- The shift splits: each model's difference from A, with a 95% task-bootstrap interval.
- The references W, M and O.
- Greedy VSC.

## 8. Rules and known limits

- No test or shift split is used for any choice, and nothing is tuned after any output.
- Three seeds per trained arm: there is not time for five before the deadline. The seed condition is the stated guard.
- On the renamed-tool API the ledger recognises no tool, so it prints only "unknown". That is reported, not fixed.
- The oracle encodes the fault taxonomy by hand. It shows what written checks achieve; it is not a learned policy.
- A failed arm is reported with a diagnosis. Deviations go in `docs/stage2/DECISIONS.md` with a separate freeze.
