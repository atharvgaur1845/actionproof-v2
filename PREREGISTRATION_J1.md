# PREREGISTRATION J1: an independent LLM judge committee measured against the deterministic verifier

Registered 2026-11-23, before any calibration or main-set judge output. The registration time is in `results/j1_hashes.json`. A 3-case pilot (`outputs/*__pilot.jsonl`) checked only that providers respond and replies parse; it is not analysed. Code, rubric and cases: `evaluation/llm_judge_committee/`.

## 1. Questions

1. Can an LLM judge that sees only the transcript tell whether a ProcureSim write actually took effect? This is the setting of production LLM monitoring.
2. Does access to a database snapshot close the gap?
3. How do the judges agree with a human, with each other, and with themselves under a paraphrased prompt?

The SQLite verifier is ground truth throughout. No judge output changes any ActionProof metric.

## 2. Judges (four model families; none is the policy's family, Cohere/Aya)

| Name | Router id (pinned provider) | Licence |
|---|---|---|
| gemma | google/gemma-4-31B-it:coreweave | Gemma (model card) |
| qwen | Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra | Apache-2.0 |
| gptoss | openai/gpt-oss-120b:baseten | Apache-2.0 |
| deepseek | deepseek-ai/DeepSeek-V3.2:deepinfra | MIT |

All judges run at temperature 0. Each judge is called independently. Answer options are listed in an order randomised per (case, judge, variant). Prices are recorded in `committee.py` and the cost cap is $15 (approved by the author).

## 3. Cases (`build_cases.py`, seed 20261123)

The source is recorded evaluation episodes of the frozen procurement-study models:
- `s3_conf_full` (English);
- `s4_lang_hi`;
- `s4_lang_hinglish`.

Strata are (fault class, read-back, verified completion, completion claimed). Only the 17 strata with at least 20 episodes are used.
- **Calibration:** 76 cases, 4 per stratum plus 8 Hindi/Hinglish, from 27 models.
- **Main:** 660 further cases, up to 40 per stratum.

Model identity, the injected fault and the verifier's verdict are removed, so judges are blind to the algorithm.

**Views:**
- `transcript`: the policy, tool catalogue, request, calls with results, and the report;
- `transcript_state`: the same plus the final database rows for the purchase order.

## 4. Runs (each once)

- calibration × {transcript, transcript_state} × v1;
- main × {transcript, transcript_state} × v1;
- main × transcript × v2: a paraphrased preamble, for prompt sensitivity.

## 5. Human labels

The author labels the 76 calibration cases in `label.html`, which shows the transcript view only. The labels use the same seven questions, including `cannot_determine`. There is one annotator; the report says so and claims no inter-annotator agreement. Human labels record what the transcript establishes, which is not necessarily what happened.

## 6. Analysis (`analyze.py`, frozen with this file)

**Ground-truth mapping:**

| Question | Verifier counterpart | Positive class |
|---|---|---|
| Q1 | `goal_met` | `yes` |
| Q3 | `false_success` | `yes` |
| Q4 | unrefunded duplicate (`violation` or `duplicate_irreversible`) | — |
| Q2 | report status | — |

`cannot_determine` is an abstention. Each judge reports:
- coverage;
- selective accuracy, over answered cases;
- accuracy counting an abstention as wrong;
- precision, recall, F1 and Cohen's κ, with abstentions counted as the negative class;
- the false-positive rate: saying the operation took effect when it did not.

**Committee rules (descriptive):**
- majority of the four, with ties counted as an abstention;
- unanimous, otherwise abstain.

The committee is never treated as ground truth.

**Predictions (registered; "supported" rules fixed now):**

| ID | Prediction | Test (main set) | Supported if |
|---|---|---|---|
| **J-H1** | With the transcript only, judges are fooled on unverifiable failures: false success claims with no read-back, about 294 + 311 + 133 episodes in the pool | Per judge, the share answering Q1 = `yes` on main cases in the strata `(phantom or partial or never_commit, noread, vsc 0, claim)` | at least 3 of 4 judges have a share ≥ 0.25, by a Wilson 95% lower bound |
| **J-H2** | A database snapshot lets judges match the verifier | Per judge, Q1 accuracy vs `goal_met`, `transcript_state` view, all main cases | at least 3 of 4 judges reach ≥ 0.90, by a Wilson 95% lower bound |
| **J-H3** | The transcript is enough when the agent read back | Q1 accuracy, transcript view, main cases with read-back | descriptive |

**Also reported:**
- human–judge κ on every question;
- inter-judge Fleiss κ;
- v1 vs v2 agreement per judge on Q1;
- accuracy in English vs Hindi/Hinglish;
- accuracy by fault class;
- a "claim bias" check: P(Q1 = yes | report claims completion) − P(Q1 = yes | no claim), at matched `goal_met`;
- confidence calibration (reliability bins, Brier score on Q1);
- latency (median, p95) and cost per 1,000 judgements;
- a list of every case where all four judges contradict the verifier.

## 7. Deviations

Logged in `docs/stage2/DECISIONS.md` with the time taken from `date`.
