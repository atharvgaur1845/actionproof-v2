"""SEL1 scripted references (descriptive; PREREGISTRATION_SEL1.md): B5 (selective), B4 (escalates on the first
unavailable read), A5 (never escalates), B3 (bounded retry with idempotency), B0 (acknowledgement-trusting) and ALWAYS
(escalates at once) on sel1_test (SEL1 env) and txn_test (TXN's canonical env, as TXN's own policy runs). One
deterministic episode per task. CPU only; seconds.

Output: results/sel1/policy_<P>__<split>.jsonl and .summary.json (never overwritten).

    python scripts/sel1_policies_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.sel1.policy import SEL_POLICIES, run_policy
    from actionproof.txn.truth import breakdown, summarise
    out_dir = ROOT / "results/sel1"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("sel1_test", "txn_test"):
        tasks = load_tasks(DATA / f"{split}.jsonl")
        for name in SEL_POLICIES:
            out = out_dir / f"policy_{name}__{split}.jsonl"
            if out.exists():
                print(f"skip {out.name}")
                continue
            rows = []
            for t in tasks:
                rec, _ = run_policy(name, t, txn_env=(split == "txn_test"))
                rec["tag"] = f"policy_{name}"
                rows.append(rec)
            out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
            sc = [r["txn"] for r in rows]
            s = {"tag": f"policy_{name}", "split": split, "n_tasks": len(tasks), "all": summarise(sc),
                 "by_group": breakdown(sc, "group"), "by_regime": breakdown(sc, "regime")}
            out.with_suffix(".summary.json").write_text(json.dumps(s, indent=1))
            print(split, name, "STR", s["all"]["STR"], "CER", s["all"]["CER"])


if __name__ == "__main__":
    main()
