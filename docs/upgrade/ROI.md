# ActionProof: operational impact and ROI (deliverable H)

**How to read this document.** Every simulator frequency below is **measured**; every dollar figure is a **projection**.
- The frequencies come from ProcureSim (`results/s3/conf/*__s3_conf_full.jsonl`, 300 tasks per run, every seed pooled) and the scripted runs (`scripts/roi_scripted.py`).
- The dollar figures are projections from external assumptions (§3). **No customer deployment exists, and no figure here is a real saving.**
- Reproduce with `python scripts/roi_model.py` → `results/roi/roi_model.json`.

## 1. The problem a buyer has

An agent that **writes** to a system of record can be told "ok" when the write stored nothing, stored part of it, or stored it twice. The agent then reports success; ProcureSim's acknowledgement is byte-identical across all of these outcomes. The financial exposure is therefore:
- **silent false success:** the task is not done, nobody knows, and it is discovered late;
- **unrefunded duplicates:** money leaves twice;
- **the manual work of finding both.**

The industry backdrop is broad: duplicate *or erroneous* payments are about 0.8–2.0% of disbursements, with a median of 1.5% (APQC via vendor summaries, §3). That figure covers all causes, not agents.

## 2. Model

Per 1,000 purchase-order workflows (receive or order, pay, post), for each arm:

P(outcome) = (1 − f)·P(outcome | clean) + f·Σ_c w_c·P(outcome | fault class c)

- The conditional rates are measured in ProcureSim, and w is the test pool's own mix over the six fault classes.
- The fault incidence **f** is an assumption: the pool is deliberately enriched, with only 50 of 300 tasks clean.
- Outcomes (verifier):
  - verified;
  - false success (claimed done, not done);
  - unrefunded duplicate;
  - flagged (honest failure or escalation: a human sees it at once).

**Costs:**
- false success = exception cost × late-discovery multiplier;
- duplicate = value × (1 − recovery rate) + exception cost × multiplier;
- flagged = exception cost;
- inference = calls ÷ measured throughput × GPU price.

## 3. Assumptions and sources

| Parameter | Conservative | Moderate | Optimistic | Source and quality |
|---|---|---|---|---|
| Fault incidence per workflow f | 0.5% | 2% | 5% | **Assumption.** Bracketed by APQC's 0.8–2.0% duplicate-or-erroneous disbursement rate ([CFO.com / APQC](https://www.cfo.com/news/metric-of-the-month-detect-and-prevent-duplicate-or-erroneous-payments/); [Corpay summary of APQC 2024](https://www.corpay.com/en-NZ/resources/blog/duplicate-payment-detection), a vendor secondary source). Agent-specific silent-write rates are not published |
| Transaction value | $2,000 | $5,000 | $10,000 | **Illustrative**, no source; the calculator makes it editable |
| Cost to resolve one exception by hand | $25 | $60 | $175 | **Weak.** Per-invoice processing is $5.83–6.00 median and ≤ $2.07 top quartile ([APQC via CFO.com](https://www.cfo.com/news/metric-of-the-month-accounts-payable-cost/659393/)); $9.40 average and $2.78 best-in-class ([Ardent Partners 2024 via Bottomline](https://www.bottomline.com/resources/blog/ardent-2024-epayables-study-automation-ai-earning-ap-a-seat-at-the-strategy-table)). The $175 upper value is a vendor's illustrative 5 h × $35 calculation ([Peakflo](https://peakflo.co/blog/three-way-matching-exceptions-ai-solutions)). No benchmark for per-exception cost was found |
| Late-discovery multiplier | 2× | 3× | 5× | **Assumption** |
| Duplicate recovery rate | 70% | 60% | 50% | **Weak.** An unsourced blog estimate of 60–70% ([DEV Community](https://dev.to/obscuriea/duplicate-payment-detection-and-vendor-overpayment-recovery-1cpn)); recovery audits are described as a backstop, not a control |
| GPU price per hour | $2.50 | $5.00 | $5.00 | HF Jobs H200 listing (`results/roi/roi.md`) |
| Policy calls per GPU-hour | 72,000 | | | **Measured.** 9,600 tickets per H200-hour × 7.5 calls, Stage 1 (`results/roi/throughput.json`) |

## 4. Results (projections; frequencies measured)

**Moderate scenario, per 1,000 workflows:**

| Arm | Verified | False success | Unrefunded duplicates | Flagged to a human | Calls per workflow | Projected cost |
|---|---|---|---|---|---|---|
| SFT agent (warm start) | 984.7 | 11.44 | 3.76 | 0.08 | 3.70 | $10,261 |
| Acknowledgement-reward GRPO | 984.7 | 11.44 | 3.79 | 0.05 | 3.70 | $10,318 |
| State-reward GRPO | 991.0 | 3.38 | 3.65 | 2.02 | 6.42 | $8,682 |
| Class-matched control GRPO | 993.5 | 3.09 | 1.31 | 2.06 | 6.43 | $3,540 |
| **Recovery-start GRPO** | 996.0 | **1.12** | **0.82** | 2.08 | 6.44 | **$2,106** |
| No verification (scripted) | 984.7 | 11.44 | 3.84 | 0.00 | 3.70 | $10,430 |
| **Hand-written database checks + fixed repair rules (scripted)** | 998.0 | **0.00** | **0.00** | 2.00 | 6.45 | **$120** |

**Conservative scenario** (f 0.5%, $2,000): SFT agent $755; recovery-start GRPO $160; hand-written checks $12.

**Optimistic scenario** (f 5%, $10,000): $80,285; $15,345; $875.

All rows are in `results/roi/roi_model.json`.

**Inference is not the cost driver.** At the measured throughput, reading back after every write raises inference from about $0.26 to $0.45 per 1,000 workflows. The extra calls are worth it in every scenario.

**Sensitivity** (moderate scenario, one assumption moved to its conservative and then its optimistic value):
- **Fault incidence dominates.** The SFT agent's cost ranges from $2,565 to $25,652.
- **Exception cost matters next:** $8,662 to $15,514.
- **The ranking of arms never changes.**

**Break-even.** Suppose one workflow's checks and repair rules cost $4,000 to build: an assumed 40 engineering hours at $100 per hour. Against the SFT agent, they pay back after about **394 workflows** in the moderate scenario, 5,390 in the conservative one, and 50 in the optimistic one. Training recovery-start GRPO costs about one H200-hour per run (about $5–10). Its projected saving, $8.16 per workflow in the moderate scenario, pays that back almost at once.

## 5. Why would a company pay for ActionProof instead of writing a few database checks?

**The measured answer: it should write the checks.**
- In ProcureSim, the hand-written verify-and-repair layer beats every trained policy: 0 false successes and 0 unrefunded duplicates, against 1.12 and 0.82 per 1,000 for the best trained agent.
- The trained agents' gains also fail to transfer to renamed tools or a new workflow step (`results/s4/s4_summary.json`).

**The product is therefore not "RL instead of checks".** What a buyer pays for is what made these numbers possible.
1. **Fault-injection testing before deployment.** It replays an agent against phantom, partial, duplicate, lost-acknowledgement, pre-commit-timeout and never-commit writes. The writes are indistinguishable in the acknowledgement and graded by an authoritative verifier. This is how the table above was produced. A team writing "a few database checks" does not know which checks it is missing until it measures this.
   - The naive check, did the call return ok, catches nothing here.
   - A check that a row exists still misses partial payments: the 62,892 of 251,570 cents case in the paper.
   - Only amount-level reconciliation catches them.
2. **A runtime verification gateway with a receipt ledger.**
   - It enforces read-back and reconciliation on every write, whatever the agent does.
   - The scripted oracle is the upper bound on what that gateway plus repair rules achieves here.
   - The LLM-judge results (J1) test whether a model-based monitor could replace it: transcript-only judges cannot see an unverified write, by construction.
3. **Training as a second line, not the first.** Recovery-start GRPO cuts projected losses by about 80% against the untrained agent (moderate scenario) at negligible training cost. It does not remove the need for checks, and it does not carry over to new workflows.

## 6. Customer, buyer, integration, pricing

| Item | Statement |
|---|---|
| Target customer | Teams deploying tool-using agents that write to systems of record: AP and procurement, refunds, payouts, IT change execution |
| Buyer | Head of finance operations or controller, with the platform or AI-engineering lead as the technical owner |
| Integration | (a) CI-stage harness: point the agent at ProcureSim-style fault injection for your tool schemas. (b) Runtime gateway between the agent and the write APIs: idempotency keys, read-back, reconciliation, and a receipt ledger for audit |
| Deployment requirements | Read access to an authoritative record for every write type (payments, receipts, ledger); idempotent write APIs or keys |
| Alternatives | Hand-written checks (cheapest when workflows are few and stable); recovery audits after the fact; LLM-judge monitoring (cannot see unverified writes); RPA with fixed rules |
| Pricing hypothesis (untested) | Per verified write, or per agent-workflow under test, priced well below the moderate-scenario loss avoided per workflow (about $10). For example, $0.05–0.20 per gated write plus a test-suite licence |
| Limitations | One simulated domain. Assumed fault rates and costs. No customer data. The hand-written baseline knows ProcureSim's fault taxonomy, which real checks would have to discover; that discovery is the testing product's job |
