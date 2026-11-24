# Final decision

2026-11-24. AD-C1 is still running; its result is added when `results/ad/c1_summary.json` exists.

1. **Ship D as the demonstration agent.** It is registered (P1 H3: +0.306 over the recovery-start agent, all seeds positive), and it won development selection.
2. **Do not ship the ledger or the action guard.** Both fail their registered tests (P2). The guard's failure is a design fault: its read-before-retry rule blocks the taught refund-then-re-pay repair. A revised guard would need a new registration.
3. **State the limits plainly.**
   - The scripted oracle still beats every trained agent (0.924 against 0.819).
   - The unseen workflow is not solved (0.486), and every agent skips the new step there.
   - All results come from one base model and one simulator.
4. **Contribution framing.** A fault-injection harness and verifier that measure what tool agents do when an acknowledgement lies. Registered evidence about which training changes help (recovery demonstrations, a state reward) and which do not (inference-time hints, ledgers, a naive guard, randomised tool names, our own credit methods).
5. **Author-only before submission:**
   - confirm the deadline on the portal;
   - the §30 declaration and team contributions;
   - the calibration labels / second annotator;
   - approval of the paper v2 restructure and of the HF Space redeploy;
   - approval to push the Phase II branch.
