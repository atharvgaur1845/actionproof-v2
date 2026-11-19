# Stage 2 related-work matrix (scan of 2026-11-19/20)

**Column key:**
- **Ack** = assumes tools and acknowledgements report truthfully.
- **AS** = checks authoritative state.
- **TB/TA** = handles timeout before / after commit.
- **P** = handles partial commit; **Dup** = handles duplicates.
- **Read** = how much was read: full text, abstract, docs or snippet.

**Our differentiation codes:**
- **d1**: phantom acknowledgements that are byte-identical to success.
- **d2**: trains the actor against the integrity layer.
- **d3**: a deterministic check of authoritative state after the write.
- **d4**: money and policy contracts over multi-step workflows.
- **d5**: a measured LLM-judge failure even when the judge is given the ledger.

| Item | Year | Contribution and mechanism | Limitation against our thesis | Ours | Read |
|---|---|---|---|---|---|
| LIMBO 2609.29095 (Microsoft) | 26 | 12 boundary faults; effect-ledger grader; key contracts, verify-before-retry, wait, guard; 9 frontier models | Faults are signalled; no phantom success; no training; Prop. 1: verification alone fails under late commits | d1 d2 d4 | full |
| Stale Does Not Mean Unsafe 2609.29522 | 26 | Commit-time guards (epoch, read-set, semantic predicate); leave-one-family-out | Blocks rather than recovers; infrastructure domain | d2 d4 | full |
| EGR 2610.41453 | 26 | Belief over hidden commit states; capability contracts; query/wait/retry/compensate/handoff | Controller never reads authoritative state; author-written model; 37 assignments | d2 d3 | full |
| Atomix 2602.14849 | 26 | Epochs, effect classes, compensation | Trusts tools; no postcondition check; cross-endpoint partial commits possible | d1 d3 | full |
| Cordon 2606.17573 (EuroSys'27) | 26 | Shadow state, outbox, idempotency table, scoped approval, invariants | Pre-commit policy only; partial receipts go to manual review | d1 d3 | full |
| Tool Receipts 2603.10060 | 26 | HMAC receipts for tool claims | Checks against tool output, not state; a lying tool still gets a valid receipt | d1 d3 | full |
| Notarized Agents 2606.04193 | 26 | Service-signed receipts; Merkle transparency log | Trusts the service; no timeouts or partials; microbenchmarks | d3 | full |
| Proof-or-Stop 2607.14890 | 26 | Output as a claim; admissible evidence; bounded retry, then escalate | Coding domain only | d4 | full |
| Harness value 2609.20474 | 26 | Planning vs terminal verifier on tau2; dollar liability | Verifier reads text only; rejects 17% of correct episodes | d3 d5 | full |
| Janus 2609.38266 | 26 | Evidence logged before effect; signed hash chain; risk-limit gates; sagas | Trusts declared facts (paid wrong amounts); 1 model | d1 d3 | full |
| Idempotency Is Not Recovery 2611.23353 | 26 | Auth + reservation + keys + reconciliation + sealed evidence; classes duplicate/orphaned/ambiguous/unresolved | Frozen responses; no live or trained policy; no money contracts | d1 d2 | full |
| Verified Tool Calls 2608.02645 | 26 | Postcondition wrapper (true/false/unknown) on authoritative state; verify-before-retry | Hand-written postconditions; 2 workflows, 1 model | d2 d4 | full |
| UndoBench 2610.05622 | 26 | Competence vs recovery (CRSR); payments; lost-ack boundary | No phantom acks; no training | d1 d2 | full |
| AgentFaultBench 2611.20711 | 26 | Real GitLab; forged receipt and forged verification faults | No training; 15 instances | d2 d4 | full |
| Mnemosyne 2607.00269 | 26 | Agentic transaction processing; append-only log; compensation | Ambiguous commits not described | d1 | abstract |
| SagaLLM 2503.11951 (VLDB'25) | 25 | Sagas for multi-agent LLM planning | Unclear whether it checks authoritative state | d3 | abstract |
| RAC 2605.03409 | 26 | Log-based compensation via MCP annotations | Assumes failures are signalled | d1 | snippet |
| Temporal / DBOS / Restate / LangGraph | docs | Durable execution, event history | At-least-once; idempotency left to the developer; trusts tools | d1 d3 | docs |
| AgentSpec 2503.18666 (ICSE'26) | 25 | Runtime rule DSL (trigger, predicate, enforce) | Pre-action only | d3 | snippet |
| ShieldAgent 2503.22738; Pro2Guard 2508.00500 | 25 | Policy circuits; probabilistic risk guard | No commit check | d3 | snippet |
| ContrAgent 2609.18128 | 26 | LTLf contracts compiled to automata | Checks the trace, not state | d3 | abstract |
| CoVer 2610.04699 | 26 | LLMs cannot decide which rules a check can verify | — | d5 | abstract |
| EBTE 2607.25364; 2609.14744 | 26 | Typed action claims checked against server facts; resource quarantine | Pre-execution only | d3 | abstract |
| Mandato 2608.14074 | 26 | Signed mandates at an MCP proxy; hash-chained log | Covers authorization, not commit | d3 d4 | abstract |
| AIREP 2608.21363 | 26 | Decision / control / execution / effect evidence; "indeterminate" state | Does not establish whether events are true | d3 | abstract |
| EvidenceNet 2609.10181 | 26 | Post-change observations checked against completion contracts | Network domain; no training | d2 d4 | abstract |
| Revocation outcome 2610.15991 | 26 | Transport ack ≠ enforcement (49.78% still admitted) | Revocation only | d4 | abstract |
| Invariant Guardrails; NeMo; AEGIS 2603.12621 | 25–26 | Tool-call policy proxies | No postcondition checks found | d3 | docs |
| 2609.15906; IETF adaptive authz draft | 26 | Risk-adaptive step-up approval | Authorization only | d3 | snippet |
| Google AP2 | 25–26 | Checkout/payment mandates; signed receipts | Receipts issued after authorization; no timeout handling | d3 d4 | docs |
| ACP (OpenAI/Stripe/Meta) | 25–26 | Agentic checkout; required Idempotency-Key; webhooks | Merchant is the source of truth; agent claim not reconciled | d3 | docs |
| Stripe idempotency; Shared Payment Tokens | docs | Saved response ≥ 24 h; scoped tokens | Concurrent conflicting requests are not saved | d2 | docs |
| Coinbase x402; 2605.30998 | 25–26 | HTTP 402 payments; verify/settle | Duplicate-settlement race | d4 | docs, abstract |
| Visa TAP; Mastercard Agent Pay; UCP | 25–26 | Agent identity, intent, tokens | Identity and intent only | d3 | press |
| AWS AgentCore Payments; Beltic | 26 | Spend caps; agent identity | No outcome verification found | d3 | snippet |
| A402 2603.01179 | 26 | Payment finalizes only if service delivered (TEE adaptor signatures) | Needs a TEE; machine-to-machine only | d4 | abstract |
| Agentic Settlement Protocol 2609.02208 | 26 | Escrow, fulfilment ladder, partial refunds | Stablecoin rails; specification only | d2 | abstract |
| AP2 security 2608.23858; 2609.11757; 2602.06345 | 26 | Threats; injected product text steers carts | Manipulation before authorization | d3 | abstract |
| AIP-Bench 2607.21824 | 26 | 33 commerce protocol vulnerabilities | Security only | d3 | abstract |
| tau2-bench 2506.07982 | 25 | Dual-control support; state goal check | Trusts tools | d1 | snippet |
| AppWorld 2407.18901 | 24 | State unit tests including collateral damage | Trusts tools | d1 | snippet |
| AgentDojo 2406.13352 | 24 | Adversarial tool outputs (injection) | No false acks | d1 | snippet |
| CRMArena-Pro 2505.18878; ToolSandbox 2408.04682 | 24–25 | Business and stateful tasks | Trust tools | d1 | abstract |
| 2610.10062; SilentProbe 2609.00035 | 26 | Silent failures detected 58.8% vs loud 91.3% | Evaluation only | d2 | abstract |
| Calibrate-Then-Act 2602.16699 and VoI papers | 26 | Cost-aware exploration and selective verification | Information gathering, not money writes | d4 | abstract |
| VeriCUA 2610.26469 | 26 | RL-trained proactive verifier | Trains the verifier, not the actor | d2 | snippet |
| Magentic Marketplace 2510.25779 | 25 | Two-sided agent market | No settlement faults | d1 d4 | abstract |
| AgenticPay 2602.06008; A2A negotiation 2506.00073; collusion 2601.01279, 2507.01413 | 25–26 | Bargaining and collusion | No execution faults | d1 | snippet |
| OTel GenAI semconv; LangSmith; Langfuse; Phoenix; AgentOps | 26 | Tool spans, online evaluators | No claimed-vs-actual reconciliation found | d3 | docs |
| Earlier scouts (VeriGUI, SENTINEL, 2601.22607, AgentRobustBench, FTA, ECT, RetryBench, Confident Closing, honesty-RL papers) | 25–26 | See §7 of Appendix Z and `docs/research/novelty_scouts_2026-11-11.md` | — | — | — |

**Ideas that look novel but already exist** (do not claim them):
- idempotency keys for agent tools;
- verify-before-retry;
- signed hash-chained receipts and ledgers;
- signed mandates and money limits;
- an "ambiguous/indeterminate" outcome class;
- "agent output is a claim";
- MCP-intercepting gateways;
- choosing between retry, reconcile, compensate and escalate;
- commit-time precondition re-checks;
- risk-adaptive approval;
- payment bound to delivery;
- "a transport ack is not the effect";
- LLM judges accepting false completions.
