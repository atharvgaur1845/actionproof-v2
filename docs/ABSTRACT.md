# ActionProof: agents say "done", we check the database

**Track 3, Agentic Fine-tuning · AgentTune → AuditKIT (+ CuratorKIT) · Cohere Tiny Aya Global and Aya Expanse 8B / 32B**

Tool-using agents report the result of a write ("refund issued") by repeating the tool's acknowledgement. Real tools lie: writes are acknowledged but never committed, partially applied, stale, timed out or ambiguous. ActionProof is a customer-support and refund environment backed by SQLite with eight injected fault types and a receipt ledger. Every metric is computed from database state, never from the transcript. The primary metric is **Verified Safe Completion (VSC)**: every goal met in the database, no policy violation, and a final report that matches what actually happened.

**Measure.** Untrained Cohere models below 32B score VSC ≈ 0. Aya Expanse 32B reaches 0.515. Prompting it to "verify every write" *lowers* VSC to 0.33. Across 3,200 episodes it never once re-read state after a *successful* acknowledgement. LLM judges are fooled the same way. On 476 false "completed" episodes, both Aya 32B judges, ours and AuditKIT's TaskCompletion, accept 76–95% of the false claims, and they accept 100% of ambiguous-ack failures.

**Fix.** From one shared warm start (verifier-filtered RFT), we ran GRPO with six rewards; only the reward differs between arms. The transcript-based rewards (tool-trusting, LLM judge) end at 0.61 test-IID VSC, and they *remove* the init's occasional post-acknowledgement checks. All four state-based rewards end at 0.815–0.85. The two we measured, outcome and PCSTR-d, re-check after every refund acknowledgement. Our reward, **PCSTR-d**, also penalises silent failure. Over two seeds it beats the tool-trusting reward by **+24.2 pp VSC** (95% CI [+18.5, +30.2]) in-distribution and by **+14.2 pp** on held-out fault types. The tiny trained model beats the untrained 32B by 34 pp. The decisive variable is *where the reward looks*. PCSTR-d's dense terms add only +2.5 pp over a binary outcome reward. The gains transfer unchanged to Hindi, Hinglish and pressure prompts ("just confirm it's done").

**Understand.** A linear probe decodes "this task failed" from the residual stream, but a text classifier on the visible transcript decodes it equally well. Used as an inference-time guard, the probe raises post-acknowledgement checks tenfold but leaves VSC unchanged: nudged models ignore the nudge or ignore what they read. Prompts and probes change what the model *calls*; only the state reward changes what it *concludes*.

**Limits, reported as results.**
- Verification is learned per tool, not as a general skill. On a held-out banking domain every trained policy fails (VSC ≈ 0, more policy violations).
- Remaining failures are still reported as "completed".
- The policy's per-service verification rates are a memorised prior, not estimated from observed faults.

**ROI.** On one H200 the trained model handles about 9,600 support tickets per hour. At 100k tickets a month that costs about $52 self-hosted, against about $228 (Gemini 3.1 Flash-Lite) and $684 (GPT-5.4 Mini) at list API prices, on the same measured workload.

**Rigour.**
- Splits, scoring code and hypotheses were pre-registered with hashes. Three amendments are logged, including a sampling-seed bug that we found and disclosed.
- The confirmatory test runs once, on untouched splits, after the experiment freeze.
- All models, traces and logs are released on Hugging Face.
