# AD-C1 addendum 1: candidate D3 enters by the same rule

Registered 2026-11-23 16:35 IST, together with AD-D3.
- **Seen at registration:** no C1 output, no cw_conf output, no D2 output, and no D3 output. The seed-1 outputs seen
  so far are awrb s1 only.
- **Hash file:** `results/ad_c1a1_hashes.json`. `PREREGISTRATION_AD_C1.md` and its hash file are not edited.

**Changes.**
1. **Inclusion.** `nostd_ema_irc` (AD-D3) enters C1 iff it passes G1–G4 (`scripts/ad/d3_gates.py`).
2. **Hypotheses added if it enters,** both inside the same Holm family as PREREGISTRATION_AD_C1.md §3:
   - H_D3: VSC(nostd_ema_irc) > VSC(grpo);
   - H_D3b: J_c(nostd_ema_irc) > J_c(nostd_ema). This one is tested only if nostd_ema also enters.
   - The test and decision rule are those of C1 §3, applied to the stated metric.
3. **Order.** The C1 training tasks for grpo seeds 2–4 now follow the D3 runs (1216, 1217) in the two chains, so at
   most two GPUs are used.
