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

UNBOUNDED = (50_000, 150_000)  # far past any reachable report time (18 turns x 5 s + waits of <= 300 s)
BOUNDED = (20, 110)            # strictly inside the documented 120 s settle bound
LAG = (30, 90)                 # replicated-ledger lag (s)
EXT_AT = (5, 30)               # when the billing job runs (s); it is idempotent (skips if a refund op exists)

# family -> (outcome of the agent's first refund call, evidence/actor configuration, resolvable)
SCENARIOS: dict[str, dict[str, Any]] = {
    "benign":              {"kind": "ok", "resolvable": True},
    "lost_ack":            {"kind": "lost_ack", "resolvable": True},
    "stale_ledger":        {"kind": "ok", "lag": True, "resolvable": True},
    "bounded_late":        {"kind": "late_bounded", "resolvable": True},
    "unbounded_late":      {"kind": "late_unbounded", "resolvable": False, "pair": "inflight"},
    "never_commit":        {"kind": "never_commit", "resolvable": False, "pair": "inflight"},
    "phantom":             {"kind": "phantom", "resolvable": True},
    "partial":             {"kind": "partial", "resolvable": True},
    "external_during":     {"kind": "ok", "external": "during", "resolvable": True, "pair": "external"},
    "external_during_lag": {"kind": "ok", "external": "during", "lag": True, "resolvable": True, "pair": "external"},
    "pre_satisfied":       {"kind": "ok", "external": "pre", "resolvable": True},
}
# Held out of ALL training (RFT demos and GRPO pools): evaluation only.
HELD_OUT = ("external_during_lag", "never_commit")
SPLITS = {"a5_train": {"seed": 32, "per": 36, "exclude": HELD_OUT},
          "a5_val": {"seed": 33, "per": 8, "exclude": HELD_OUT},
          "a5_eval": {"seed": 31, "per": 30, "exclude": ()}}
PROMPTS_A5 = (
    PROMPTS["refund"],
    "Ticket {ticket}: order {order} from customer {cust} came back. Refund it to the original payment method, log an "
    "audit note on the ticket saying what you did, then close the ticket.",
    "Customer {cust} returned order {order} (ticket {ticket}). Please refund the order to its original payment method, "
    "record an audit note, and close ticket {ticket}.",
)


def _params(rng: random.Random, sc: dict[str, Any], order: str) -> dict[str, Any]:
    kind = sc["kind"]
    f: dict[str, Any] = {}
    if kind == "late_bounded":
        f = {"issue_refund:0": {"kind": "late_commit", "delay": rng.randint(*BOUNDED)}}
    elif kind == "late_unbounded":
        f = {"issue_refund:0": {"kind": "late_commit", "delay": rng.randint(*UNBOUNDED)}}
    elif kind != "ok":
        f = {"issue_refund:0": {"kind": kind}}
    ext = []
    if sc.get("external") == "pre":
        ext = [{"at": 0, "actor": "billing_bot", "order_id": order}]
    elif sc.get("external") == "during":
        ext = [{"at": 5 * rng.randint(EXT_AT[0] // 5, EXT_AT[1] // 5), "actor": "billing_bot", "order_id": order}]
    return {"faults": f, "external": ext, "ledger_lag": rng.randint(*LAG) if sc.get("lag") else 0,
            "bounded_tools": ["issue_refund"] if kind == "late_bounded" else []}


def make_a5_split(split: str) -> list[Task]:
    c = SPLITS[split]
    out, i = [], 0
    for name, sc in SCENARIOS.items():
        if name in c["exclude"]:
            continue
        for j in range(c["per"]):
            # Paired families share ONE stream (world, ids, prompt, env seed, actor timing), keyed by the pair, so
            # paired tasks are observation-equivalent; the family only sets the hidden outcome.
            key = sc.get("pair", name)
            rng = random.Random(f"actionproof-a5/{split}/{c['seed']}/{key}/{j}")
            state, ids = _world(rng, "refund", False)
            prompt = rng.choice(PROMPTS_A5).format(ticket=ids["ticket"], cust=ids["cust"], order=ids["order"])
            env_seed = rng.randrange(2**31)
            prm = _params(rng, sc, ids["order"])
            if name == "external_during_lag":   # its pair partner differs ONLY in ledger lag
                prm["ledger_lag"] = rng.randint(*LAG)
            task_id = f"{split}_{i:04d}"
            a5 = {"scenario": name, "pair_id": f"{key}/{j}" if "pair" in sc else None, "env_seed": env_seed,
                  "resolvable": sc["resolvable"], **prm}
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
