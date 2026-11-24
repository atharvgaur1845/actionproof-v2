# Generalisation: of the mechanism and of the learned policies (kept separate)

## A. Generalisation of the mathematical mechanism

| Dimension | Evidence | Level | Result |
|---|---|---|---|
| Held-out random problems (1–3 fault states, severities, repairability, detection, false alarms, state-dependent costs, exogenous noise, G ∈ {2..16}, four group-law families, four initial policies) | TH1 held-out, 600 cells + twins | analytical simulation, registered | see `TESTBED_RESULTS.md` |
| Group size, correlation family | scan `iid_interior`, `critical_rho`, `rho_curves` | math | CORRELATION_EXTENSION |
| Multi-decision workflows | LAB1 L2G (`results/lab/grad_snr.json`): GRPO's exact mean gradient is wrong-signed on every risky verify logit (cos −0.175) | testbed, prior lab | the vote form is approximate; Thm 3.1 is exact |
| Fault types in the LLM | per-class β̂ in `llm_mechanism.json` (phantom, partial, dup_delivery, lost_ack, precommit_timeout, clean) | development | estimates rest on 1–38 mixed groups per class in steps 0–19; too few to claim per-type generalisation |
| Other models, tasks, agent frameworks | none | — | **not established** |

## B. Generalisation of the learned language-agent policies (existing confirmatory S4)

Recovery-start agent (RSC) vs state-reward agent (A12 R_V), VSC difference, 5 seeds each
(`results/s4/s4_summary.json`, unpaired permutation p one-sided, two-level 95% CI).

| Split | n | Δ VSC | p | CI |
|---|---|---|---|---|
| English (in-distribution) | 150 | +0.123 | 0.004 | [0.072, 0.177] |
| Hindi | 150 | +0.127 | 0.004 | [0.080, 0.181] |
| Hinglish | 150 | +0.072 | 0.004 | [0.035, 0.117] |
| Paraphrase | 150 | +0.120 | 0.004 | [0.069, 0.175] |
| Renamed API | 150 | −0.016 | 0.651 | [−0.096, 0.061] |
| Unseen 4-step workflow | 100 | +0.016 | 0.409 | [−0.024, 0.070] |
| Two-fault combinations | 100 | +0.072 | 0.004 | [0.030, 0.126] |

These results concern the recovery-start *policy*, not the GRPO normalisation mechanism. They show language transfer
and fault-combination transfer, and failure on renamed APIs and new workflows. AD-C1's registered shift evaluations
(s4_api, s4_lang_hi, s4_wf4, seed 0 of each arm) will be the only evidence of how the *estimator* choice transfers.
They are pending with tasks 1240/1241.

## C. Honest summary

- The mechanism is a property of the estimator and the sampler, and it holds for any reward function (Thm 3.1).
- Its quantitative predictions are tested on held-out synthetic problems.
- Its presence in other agent-RL systems is **argued from their group construction, not observed**: tasks with a
  fixed environment snapshot per prompt, and paired-noise samplers.
