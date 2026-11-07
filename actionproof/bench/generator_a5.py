"""Amendment 5 scenario slice (draft; not registered): evidence-aware recovery under partially observable outcomes.

Every task is the Domain A refund workflow on a fresh world. Its `meta["a5"]` sets the write outcome, evidence
freshness and external actors. `resolvable` says whether the evidence available within the budget determines the
outcome. Matched pairs share a world, observations and `pair_id`, but differ in hidden truth.

    python -m actionproof.bench.generator_a5 --out data/bench/a5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from actionproof.bench.generator import PROMPTS, Task, _goal, _world
from actionproof.env.faults import FaultSchedule

UNBOUNDED_DELAY = 100_000  # far past any reachable report time (18 turns x 5 s + waits of <= 300 s)

SCENARIOS: dict[str, dict[str, Any]] = {
    "benign":              {"faults": {}, "resolvable": True},
    "lost_ack":            {"faults": {"issue_refund:0": {"kind": "lost_ack"}}, "resolvable": True},
    "stale_ledger":        {"faults": {}, "ledger_lag": 60, "resolvable": True},
    "bounded_late":        {"faults": {"issue_refund:0": {"kind": "late_commit", "delay": 60}},
                            "bounded_tools": ["issue_refund"], "resolvable": True},
    "unbounded_late":      {"faults": {"issue_refund:0": {"kind": "late_commit", "delay": UNBOUNDED_DELAY}},
                            "resolvable": False, "pair": "inflight"},
    "never_commit":        {"faults": {"issue_refund:0": {"kind": "never_commit"}}, "resolvable": False, "pair": "inflight"},
    "phantom":             {"faults": {"issue_refund:0": {"kind": "phantom"}}, "resolvable": True},
    "partial":             {"faults": {"issue_refund:0": {"kind": "partial"}}, "resolvable": True},
    "external_during":     {"faults": {}, "external": [{"at": 10, "actor": "billing_bot"}], "resolvable": True,
                            "pair": "external"},
    "external_during_lag": {"faults": {}, "external": [{"at": 10, "actor": "billing_bot"}], "ledger_lag": 60,
                            "resolvable": True, "pair": "external"},
    "pre_satisfied":       {"faults": {}, "external": [{"at": 0, "actor": "billing_bot"}], "resolvable": True},
}
# Held out of TRAINING (evaluation only): the combinations a policy cannot have memorised.
HELD_OUT = ("external_during_lag", "never_commit")
SPLITS = {"a5_train": {"seed": 32, "per": 36, "exclude": HELD_OUT},
          "a5_val": {"seed": 33, "per": 8, "exclude": HELD_OUT},
          "a5_eval": {"seed": 31, "per": 30, "exclude": ()}}


def make_a5_split(split: str) -> list[Task]:
    c = SPLITS[split]
    out, i = [], 0
    for name, sc in SCENARIOS.items():
        if name in c["exclude"]:
            continue
        for j in range(c["per"]):
            # paired scenarios share the world draw: same seed string keyed by pair, not by scenario
            key = sc.get("pair", name)
            rng = random.Random(f"actionproof-a5/{split}/{c['seed']}/{key}/{j}")
            state, ids = _world(rng, "refund", False)
            task_id = f"{split}_{i:04d}"
            a5 = {"scenario": name, "pair_id": f"{key}/{j}" if "pair" in sc else None,
                  "faults": sc["faults"], "ledger_lag": sc.get("ledger_lag", 0),
                  "bounded_tools": sc.get("bounded_tools", []), "resolvable": sc["resolvable"],
                  "external": [{**e, "order_id": ids["order"]} for e in sc.get("external", [])]}
            prompt = PROMPTS["refund"].format(ticket=ids["ticket"], cust=ids["cust"], order=ids["order"])
            out.append(Task(task_id, split, "refund", prompt, state, FaultSchedule(), _goal(task_id, "refund", ids, True),
                            meta={"faults": [name], "services": [], "ids": ids, "benign": name == "benign",
                                  "language": "en", "domain": "a5", "a5": a5}))
            i += 1
    return out


def split_bytes(split: str) -> bytes:
    return "".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in make_a5_split(split)).encode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench/a5")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (Path(a.out) / f"{s}.jsonl").write_bytes(split_bytes(s))
        print(s, len(make_a5_split(s)))


if __name__ == "__main__":
    main()
