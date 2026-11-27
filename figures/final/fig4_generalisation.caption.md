**Figure 4. Generalisation by task group and evidence regime on txn_test.** Left block: verified transaction completion
(VTC) per task group. Right block: safe terminal outcome (STR: verified, or an escalation the task marks correct with no
harm) per evidence regime. Rows separate the prompted base model, warm starts (SFT only), SEL1 SFT continuations of
Dw_aug, RL policies (SFT + GRPO) and the scripted reference B4.

- **Sources:** `results/txn/<tag>__txn_test.summary.json`, `results/sel1/<tag>__txn_test.summary.json`, blocks
  `txn.by_group` and `txn.by_regime`; table view `figures/final/fig4_generalisation.csv` (with per-seed rates).
- **Split and denominators:** txn_test, 440 locked tasks; each cell shows episodes (num/den). Task counts per column are
  in the header (from B4's single run).
- **Seed scope:** Dw_aug, SEL, CTL, D and Dn pool 3 seeds; base, W and Dw are single runs; B4 is deterministic.
- **Uncertainty:** none drawn. Small columns (renamed tools, new domain and the other OOD groups: 20 tasks each) move by
  several points between seeds; the CSV lists every seed.
- **Reading note:** the quadruple group mixes the standard and unresolvable regimes; an agent that hands off correctly
  when records are unreadable scores lower VTC there by design (B4). The post-hoc validity analysis
  (PREREGISTRATION_TXN1_ADDENDUM3.md) found that keyed unresolvable tasks can be settled by a same-key resend, which is
  why some models show VTC above B4 on the quadruples.
