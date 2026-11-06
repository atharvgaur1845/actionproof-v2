# ActionProof: agents say "done", we check the database

**Track 3, Agentic Fine-tuning · AgentTune → AuditKIT (+ CuratorKIT) · Cohere Tiny Aya Global and Aya Expanse 8B / 32B**

Tool-using agents report a write's result by repeating the tool's acknowledgement, but real tools lie: writes are acknowledged yet never committed, partial, stale, timed out or ambiguous. ActionProof is a SQLite-backed customer-support environment with eight injected fault types and a receipt ledger. Every metric is computed from database state, never from the transcript. The primary metric is **Verified Safe Completion (VSC)**: every goal met in the database, no policy violation, and a final report that matches what actually happened.

**Measure.** Untrained Cohere models below 32B score VSC ≈ 0; Aya Expanse 32B reaches 0.515. Prompting it to "verify every write" *lowers* VSC to 0.33. Across 3,200 episodes it never once re-read state after a *successful* acknowledgement. LLM judges are fooled too: on 476 false "completed" episodes, Aya 32B judges (ours and AuditKIT's TaskCompletion) accept 76–95% of the false claims, and 100% on ambiguous acks.

**Fix.** From one shared warm start (verifier-filtered RFT), we ran GRPO with six rewards; only the reward differs between arms. The transcript-based rewards (tool-trusting, LLM judge) end at 0.61 test-IID VSC, and *remove* the init's occasional post-ack checks. All four state-based rewards reach 0.815–0.85; outcome and PCSTR-d re-check after every refund ack. Our reward, **PCSTR-d**, also penalises silent failure. Over two seeds it beats the tool-trusting reward by **+24.2 pp VSC** (95% CI [+18.5, +30.2]) in-distribution and by **+14.2 pp** on held-out fault types. It beats the untrained 32B by 34 pp. The decisive variable is *where the reward looks*: PCSTR-d's dense terms add only +2.5 pp over a binary outcome reward. The gains hold in Hinglish and under pressure prompts (0.85–0.86) and stay +16 pp over tool-trusting in Hindi (0.775).

**Understand.** A linear probe decodes "this task failed" from the residual stream, but a text classifier on the visible transcript decodes it equally well. As an inference-time guard, the probe raises post-ack checks tenfold but leaves VSC unchanged: nudged models ignore the nudge or what they read. Prompts and probes change what the model *calls*; only the state reward changes what it *concludes*.

**Limits, reported as results.**
- Verification is learned per tool, not as a general skill. On a held-out banking domain every trained policy fails (VSC ≈ 0, more policy violations).
- Remaining failures are still reported as "completed".
- The policy's per-service verification rates are a memorised prior, not estimated from observed faults.

**ROI.** One H200 serves about 9,600 tickets per hour. At 100k tickets a month: about $52 self-hosted, against $228 (Gemini 3.1 Flash-Lite) and $684 (GPT-5.4 Mini) at list API prices on the measured workload.

**Rigour.**
- Splits, scoring code and hypotheses were pre-registered with hashes; three amendments are logged, including a self-found sampling-seed bug.
- The confirmatory test runs once, on untouched splits, after the freeze.
- Models, traces and logs are on Hugging Face.
