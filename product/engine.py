"""ActionProof product engine: deterministic ProcureSim scenarios, execution-truth timelines, reward laboratory and
recovery arena. Pure Python, CPU only, no model calls (policy comparisons replay recorded evaluation episodes).

Information layers are kept separate in every timeline row:
  visible  what the agent saw (its own call and the tool's response)
  state    the database transition the write caused (environment)
  truth    the evaluator-only verdict (never shown to a policy; the product shows it to the human user)
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Any

import actionproof.bench.generator_proc as G
from actionproof.integrity.sim.agents import _args, _po
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL, TABLES
from actionproof.integrity.sim.truth import score, step_verdict
from actionproof.stage3.env import make_env_s3

WORKFLOWS = {"order_prepay": "Order, then pay, then post", "receive_pay": "Receive goods, then pay, then post",
             "pay_post": "Pay, then post"}
FAULTS = {"none": "No fault", "phantom": "Phantom write (ok acknowledgement, nothing stored)",
          "partial": "Partial commit (ok acknowledgement, part stored)",
          "lost_ack": "Lost acknowledgement (stored, timeout returned)",
          "precommit_timeout": "Timeout before commit (nothing stored, timeout returned)",
          "dup_delivery": "Duplicate delivery (stored twice, ok returned)",
          "never_commit": "Never commits (every attempt acknowledged ok, nothing stored)"}
ACTIONS = ("read", "retry", "retry_new_key", "compensate", "continue", "escalate", "stop")
ACTION_LABELS = {"read": "Read", "retry": "Retry (same key)", "retry_new_key": "Retry (new key)", "compensate": "Compensate",
                 "continue": "Continue", "escalate": "Escalate", "stop": "Stop"}


@dataclass(frozen=True)
class Scenario:
    workflow: str = "pay_post"
    fault: str = "partial"
    fault_step: str = "pay"
    ratio: float = 0.5
    seed: int = 7

    def validate(self) -> None:
        assert self.workflow in WORKFLOWS and self.fault in FAULTS
        assert self.fault_step in G.WORKFLOWS[self.workflow] and self.fault_step != "post"
        assert 0.05 <= self.ratio <= 0.95
        if self.fault == "partial":
            assert self.fault_step in ("pay", "receive")


def build_task(sc: Scenario) -> Any:
    sc.validate()
    rng = random.Random(sc.seed)
    while True:  # deterministic: first generated task with the requested workflow
        t = G.make_task("product", rng.randrange(10**6), rng)
        if t.workflow == sc.workflow:
            break
    t.meta.update(faults=[], never_commit=None, fault_class=sc.fault)
    tool = STEP_TOOL[sc.fault_step]
    if sc.fault == "never_commit":
        t.meta["never_commit"] = tool
    elif sc.fault != "none":
        f = {"tool": tool, "k": 0, "kind": sc.fault}
        if sc.fault == "partial":
            f["ratio"] = sc.ratio
        t.meta["faults"] = [f]
    return t


def _rows_for(state: dict, pid: str) -> dict[str, list[dict]]:
    return {t: [r for r in state[t] if r.get("po_id") == pid] for t in TABLES if t not in ("pos", "suppliers")}


def timeline(task: Any, env: Any, n_before: int = 0) -> list[dict[str, Any]]:
    """One row per agent call: visible call and response, database diff for the task PO, evaluator verdicts."""
    pid = task.meta["po_id"]
    po, payee = _po(task), None
    sup = next((s for s in task.initial_state["suppliers"] if s["supplier_id"] == po["supplier_id"]), None)
    payee = (sup or {}).get("payee_account")
    replay = make_env_s3(task, "timeline", seed=0)  # every product environment uses seed 0, so ids replay exactly
    rows = []
    for i, step in enumerate(env.trace):
        before = _rows_for(replay.state, pid)
        obs = replay.call(step["tool"], step["args"], by=step.get("by", "agent"))
        after = _rows_for(replay.state, pid)
        diff = {t: [r for r in after[t] if r not in before[t]] for t in after if after[t] != before[t]}
        att = [a for a in replay.attempts if a["idx"] == i]
        verdicts = {s: step_verdict(s, replay.state, po, payee) for s in task.meta["requested"]}
        rows.append({"i": i, "by": step.get("by", "agent"), "scripted": i < n_before,
                     "visible": {"call": {"tool": step["tool"], "args": step["args"]}, "response": obs},
                     "state": {"new_rows": diff, "commit": att[0]["committed"] if att else None,
                               "fault": att[0]["fault"] if att else None},
                     "truth": {"verdicts": verdicts,
                               # the acknowledgement disagrees with what this call did (a replay from the idempotency
                               # store of an earlier committed write is a truthful "ok", not a divergence)
                               "diverged": bool(att) and att[0]["obs_status"] == "ok" and not att[0]["from_idempotency_store"]
                               and att[0]["committed"] in ("none", "partial", "double")}})
    return rows


def rewards(task: Any, env: Any) -> dict[str, Any]:
    """The same trajectory under every reward in the project, with the verifier evidence each one reads."""
    from actionproof.stage3 import rewards as R3
    rec = {"task": task.to_json(), "trace": env.trace, "attempts": env.attempts, "report": env.final_report,
           "final_state": env.state}
    s = score(rec)
    e = {"score": s, "report": env.final_report, "task": task.to_json()}
    return {"R_V (state reward)": float(s["r_v"]), "R_A (acknowledgement reward)": float(s["r_a"]),
            "safe (lexicographic, exploratory)": R3.safe_reward(None, [0], ap_episodes=[e])[0],
            "VSC (evaluation metric)": float(s["vsc"]), "evidence": {
                "step_verdicts": s["steps"], "report": env.final_report, "false_completion": s["false_success"],
                "duplicate": s["duplicate_irreversible"], "collateral": s["collateral"],
                "last_acknowledgements": {st: next((x["obs"].get("status") for x in reversed(env.trace)
                                                    if x["tool"] == STEP_TOOL[st]), None) for st in task.meta["requested"]}}}


def run_policy(sc: Scenario, policy: str) -> dict[str, Any]:
    """Scripted policies: 'trusting' (never reads, reports completed) and 'verifying' (the reference oracle)."""
    from actionproof.integrity.sim import agents
    task = build_task(sc)
    env = make_env_s3(task, f"product-{policy}", seed=0)
    {"trusting": agents.naive, "verifying": agents.oracle}[policy](env, task)
    return {"task": task.to_json(), "timeline": timeline(task, env), "rewards": rewards(task, env)}


class Arena:
    """A faulted workflow the user steps through by choosing the next action class."""

    def __init__(self, sc: Scenario) -> None:
        self.task = build_task(sc)
        self.sc = sc
        self.env = make_env_s3(self.task, "arena", seed=0)
        self.po = _po(self.task)
        self.steps = list(self.task.meta["requested"])
        self.i = 0  # index of the current step
        for s in self.steps:  # run up to and including the faulted write
            self.env.call(STEP_TOOL[s], _args(s, self.po, f"{s}-{self.po['po_id']}"))
            if s == sc.fault_step:
                break
            self.env.call(STEP_READ[s], {"po_id": self.po["po_id"]})
            self.i += 1
        self.n0 = 0
        self.n_new = 0

    @property
    def step(self) -> str:
        return self.steps[min(self.i, len(self.steps) - 1)]

    def act(self, action: str) -> dict[str, Any]:
        assert action in ACTIONS and not self.env.done
        pid, s = self.po["po_id"], self.step
        if action == "read":
            obs = self.env.call(STEP_READ[s], {"po_id": pid})
        elif action == "retry":
            obs = self.env.call(STEP_TOOL[s], _args(s, self.po, f"{s}-{pid}"))
        elif action == "retry_new_key":
            self.n_new += 1
            obs = self.env.call(STEP_TOOL[s], _args(s, self.po, f"{s}-{pid}-r{self.n_new}"))
        elif action == "compensate":
            if s == "pay":
                pays = self.env.state["payments"]
                refunded = {p["refund_of"] for p in pays if p["kind"] == "refund"}
                live = [p for p in pays if p["po_id"] == pid and p["kind"] == "payment" and p["payment_id"] not in refunded]
                obs = self.env.call("refund_payment", {"payment_id": live[-1]["payment_id"]}) if live else \
                    {"status": "error", "error": "nothing to compensate"}
            else:
                obs = {"status": "error", "error": "compensation is defined for payments only in this arena"}
        elif action == "continue":
            self.i += 1
            if self.i < len(self.steps):
                nxt = self.steps[self.i]
                obs = self.env.call(STEP_TOOL[nxt], _args(nxt, self.po, f"{nxt}-{pid}"))
            else:
                obs = self.env.call("submit_report", {"status": "completed", "summary": "all steps done"})
        elif action == "escalate":
            self.env.call("escalate_to_human", {"po_id": pid, "reason": "could not confirm the change"})
            obs = self.env.call("submit_report", {"status": "escalated", "summary": "escalated to a human"})
        else:  # stop
            obs = self.env.call("submit_report", {"status": "failed", "summary": "stopped"})
        return {"action": action, "response": obs, "done": self.env.done,
                "timeline": timeline(self.task, self.env), "rewards": rewards(self.task, self.env) if self.env.done else None}
