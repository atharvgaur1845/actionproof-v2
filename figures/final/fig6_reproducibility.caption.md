**Figure 6. Reproducibility map.** Two registered studies, TXN1 and SEL1, traced from source code to the documents
that quote their numbers. Every hash and count in the figure is recomputed by `scripts/build_final_figures.py` when it
runs; a box turns red if a live file no longer matches its manifest or registration record.

- **Sources:** `data/bench/proc/txn_manifest.json`, `data/bench/proc/sel1_manifest.json`, `results/txn1_hashes.json`,
  `results/sel1_hashes.json`, `claims/claims.yaml`, git history (registration commits),
  `docs/submission/REPRODUCIBILITY_FINAL.md` (command times from the two clean-room runs).
- **Split:** txn_test (440 tasks) and sel1_test (240 tasks, sealed until the SEL1 evaluation).
- **Uncertainty:** not applicable (provenance figure).

| Study | Step | Content |
|---|---|---|
| TXN1 (transaction recovery benchmark) | Source | actionproof/txn/; env, truth, policies, tasks; docs/txn/TXN_SPEC.md; spec sha256 d7c2dd8850b9 |
| TXN1 (transaction recovery benchmark) | Registration | PREREGISTRATION_; TXN1.md; added 36e5a66 2026-11-26 00:45 IST; + addenda 1-3 (dated) |
| TXN1 (transaction recovery benchmark) | Task hash | data/bench/proc/; txn_test.jsonl (440); sha256 01437c0b3418; manifest matches; registration matches |
| TXN1 (transaction recovery benchmark) | Raw traces | results/txn/ (26 episode files); released: HF dataset; atharvgaur18/; actionproof-traces |
| TXN1 (transaction recovery benchmark) | Analysis | scripts/txn_stats.py; -> results/txn/; stats__txn_test.json; sha256 11ef7b13d17f |
| TXN1 (transaction recovery benchmark) | Claims ledger | claims/claims.yaml; 4 TXN claims, 75 numbers; each re-read from its file by; scripts/s2_numcheck.py |
| TXN1 (transaction recovery benchmark) | Surfaces | README, write-up, slides, video captions, paper (scanned by s2_numcheck); site: built from results |
| SEL1 (selective escalation, registered 2026-11-27) | Source | actionproof/sel1/; env, tasks, policy, build; sha256 recorded in; results/; sel1_hashes.json |
| SEL1 (selective escalation, registered 2026-11-27) | Registration | PREREGISTRATION_; SEL1.md; added 795e7cd 2026-11-27 05:39 IST; + addendum 1 (post-hoc scope, dated) |
| SEL1 (selective escalation, registered 2026-11-27) | Task hash | data/bench/proc/; sel1_test.jsonl (240, sealed); sha256 644f7530a250; manifest matches; registration matches |
| SEL1 (selective escalation, registered 2026-11-27) | Raw traces | results/sel1/ episodes; released: HF dataset; atharvgaur18/; actionproof-traces |
| SEL1 (selective escalation, registered 2026-11-27) | Analysis | scripts/sel1_stats.py; -> results/sel1/; sel1_summary.json; sha256 17ab18c3e205 |
| SEL1 (selective escalation, registered 2026-11-27) | Claims ledger | claims/claims.yaml; SEL1 block, 45 numbers; each re-read from its file by; scripts/s2_numcheck.py |
| SEL1 (selective escalation, registered 2026-11-27) | Surfaces | README, write-up, slides, video captions (scanned by s2_numcheck); site: built from results |
