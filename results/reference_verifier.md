# Reference verifier vs project verifier

## confirmatory (results/eval/conf/*_c.jsonl)

Agreement: **4000/4000 = 1.0000**

| file | n | agree | ref VSC | stored VSC |
|---|---|---|---|---|
| grpo_tiny_v2_llm_judge_s0__neutral__test_iid_c.jsonl | 200 | 200 | 0.6450 | 0.6450 |
| grpo_tiny_v2_llm_judge_s0__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.6250 | 0.6250 |
| grpo_tiny_v2_outcome_s0__neutral__test_iid_c.jsonl | 200 | 200 | 0.8400 | 0.8400 |
| grpo_tiny_v2_outcome_s0__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.8000 | 0.8000 |
| grpo_tiny_v2_outcome_s1__neutral__test_iid_c.jsonl | 200 | 200 | 0.8650 | 0.8650 |
| grpo_tiny_v2_outcome_s1__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.8050 | 0.8050 |
| grpo_tiny_v2_pcstr_c_s0__neutral__test_iid_c.jsonl | 200 | 200 | 0.8800 | 0.8800 |
| grpo_tiny_v2_pcstr_c_s0__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.7950 | 0.7950 |
| grpo_tiny_v2_pcstr_d_s0__neutral__test_iid_c.jsonl | 200 | 200 | 0.8800 | 0.8800 |
| grpo_tiny_v2_pcstr_d_s0__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.8050 | 0.8050 |
| grpo_tiny_v2_pcstr_d_s1__neutral__test_iid_c.jsonl | 200 | 200 | 0.8800 | 0.8800 |
| grpo_tiny_v2_pcstr_d_s1__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.8050 | 0.8050 |
| grpo_tiny_v2_pcstr_s0__neutral__test_iid_c.jsonl | 200 | 200 | 0.8800 | 0.8800 |
| grpo_tiny_v2_pcstr_s0__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.8050 | 0.8050 |
| grpo_tiny_v2_tool_trust_s0__neutral__test_iid_c.jsonl | 200 | 200 | 0.6550 | 0.6550 |
| grpo_tiny_v2_tool_trust_s0__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.6150 | 0.6150 |
| grpo_tiny_v2_tool_trust_s1__neutral__test_iid_c.jsonl | 200 | 200 | 0.6600 | 0.6600 |
| grpo_tiny_v2_tool_trust_s1__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.6300 | 0.6300 |
| rft_tiny_v2__neutral__test_iid_c.jsonl | 200 | 200 | 0.6350 | 0.6350 |
| rft_tiny_v2__neutral__test_ood_fault_c.jsonl | 200 | 200 | 0.6500 | 0.6500 |

Disagreement categories:

None.

## development r3 (results/eval/r3/*__test_iid.jsonl)

Agreement: **2400/2400 = 1.0000**

| file | n | agree | ref VSC | stored VSC |
|---|---|---|---|---|
| grpo_tiny_v2_ack_outcome_s0__neutral__test_iid.jsonl | 200 | 200 | 0.6050 | 0.6050 |
| grpo_tiny_v2_ack_outcome_s1__neutral__test_iid.jsonl | 200 | 200 | 0.6050 | 0.6050 |
| grpo_tiny_v2_llm_judge_s0__neutral__test_iid.jsonl | 200 | 200 | 0.6100 | 0.6100 |
| grpo_tiny_v2_outcome_s0__neutral__test_iid.jsonl | 200 | 200 | 0.8150 | 0.8150 |
| grpo_tiny_v2_outcome_s1__neutral__test_iid.jsonl | 200 | 200 | 0.8350 | 0.8350 |
| grpo_tiny_v2_pcstr_c_s0__neutral__test_iid.jsonl | 200 | 200 | 0.8500 | 0.8500 |
| grpo_tiny_v2_pcstr_d_s0__neutral__test_iid.jsonl | 200 | 200 | 0.8500 | 0.8500 |
| grpo_tiny_v2_pcstr_d_s1__neutral__test_iid.jsonl | 200 | 200 | 0.8500 | 0.8500 |
| grpo_tiny_v2_pcstr_s0__neutral__test_iid.jsonl | 200 | 200 | 0.8500 | 0.8500 |
| grpo_tiny_v2_tool_trust_s0__neutral__test_iid.jsonl | 200 | 200 | 0.6100 | 0.6100 |
| grpo_tiny_v2_tool_trust_s1__neutral__test_iid.jsonl | 200 | 200 | 0.6050 | 0.6050 |
| rft_tiny_v2__neutral__test_iid.jsonl | 200 | 200 | 0.5550 | 0.5550 |

Disagreement categories:

None.

## Headline check

Reference VSC on confirmatory (mean of test_iid_c and test_ood_fault_c, 400 tasks): RFT v2 init **0.6425**; PCSTR-d seed 0 **0.8425**, seed 1 **0.8425** (stored headline: 0.6425 -> 0.8425, both seeds).

## Method

The reference verifier (`scripts/reference_verifier.py`) was written from the written spec only (ARTICLE_DRAFT §2, PREREGISTRATION.md and PREREGISTRATION_A4.md §1.1, tools_ecom.py, faults.py, generator.py). It reads, per episode, `task_id`, the `trace` (tool name, arguments, observation) and the `attempts` list (action_idx, op, entity, fault, committed), and regenerates the task with `make_split` to get the initial database, postconditions, required writes, allowed entities and feasibility. It does not read or import actionproof/env/verifier*.py, actionproof/rewards or eval/a4_metrics.py, and does not read the stored `verdict`, `violations` or `vsc` when computing (`vsc` is read afterwards for the comparison only). It replays the trace on a dict copy of the initial database, applying a write only when its attempt is committed (F3 partial applies only the first sub-effect, per tools_ecom._write), checks the postconditions on the final state, three invariants (refund policy at call time, refund total <= order amount, writes only on allowed entities), duplicate irreversible writes (>1 committed issue_refund/cancel_order/restock_item on one entity) and receipt coverage of every required write; G is their conjunction and VSC = 1[G=1 and first recorded report status == ('completed' if feasible else 'escalated')]. Known simplifications: it trusts the recorded `committed`/`fault` flags of each attempt (it does not re-derive commits from the fault schedule), so it is independent of the verifier but not of the environment's commit bookkeeping; the refund postcondition is 'some committed refund row with the goal amount and method' (exactly-once is handled by the duplicate check); the bank domain is not covered; the invariant set is our reading of 'policy, exactly-once, scope' in the spec.
