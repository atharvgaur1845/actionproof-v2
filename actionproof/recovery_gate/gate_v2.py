"""Recovery Gate v2 (PREREGISTRATION_RG2.md): the frozen RG1 gate (gate.py, unchanged) plus bounded patience under
an outage. When a read of a step that was written comes back `service_unavailable`, the gate reads it again, up to
PATIENCE unavailable reads in all since the step's last write, before the step counts as unresolvable. PATIENCE = 3
is SEL1's registered read budget (actionproof/sel1/policy.py READS_BEFORE_ESCALATING, used by the selective script
B5); it was not chosen from any test outcome. Every extra read is an ordinary agent-side call tagged `gate_reread`.
Everything else is the RG1 gate."""
from __future__ import annotations

from typing import Any, Callable

from actionproof.integrity.sim.env import STEP_TOOL, TOOL_STEP
from actionproof.recovery_gate.gate import NOT_SENT, READ_STEP, Execute, Gate, Intent, Rules

PATIENCE = 3


class GateV2(Gate):
    def __init__(self, intent: Intent, rules: Rules = Rules(), name: Callable[[str], str] = lambda s: s,
                 patience: int = PATIENCE) -> None:
        super().__init__(intent, rules, name)
        self.patience = patience
        self.unavail = {s: 0 for s in STEP_TOOL}

    def observe(self, tool: str, args: dict[str, Any], obs: dict[str, Any]) -> None:
        super().observe(tool, args, obs)
        if args.get("po_id") not in (None, self.intent.po_id):
            return
        if tool in TOOL_STEP and obs.get("error") not in NOT_SENT:
            self.unavail[TOOL_STEP[tool]] = 0
        elif tool in READ_STEP and obs.get("error") == "service_unavailable":
            self.unavail[READ_STEP[tool]] += 1

    def _run(self, execute: Execute, tool: str, args: dict[str, Any], tag: str) -> dict[str, Any]:
        obs = super()._run(execute, tool, args, tag)
        if not (self.rules.escalate and tool in READ_STEP and args.get("po_id") == self.intent.po_id):
            return obs
        s = READ_STEP[tool]
        n0 = self.unavail[s]
        while obs.get("error") == "service_unavailable" and self.steps[s].writes and self.unavail[s] < self.patience:
            obs = super()._run(execute, tool, args, "gate_reread")
        if self.unavail[s] > n0 or (n0 and obs.get("status") == "ok"):
            self._note("escalate", "reread", None, f"{self.N(tool)} was unavailable; the gate read it again "
                                                   f"({self.unavail[s]} unavailable of at most {self.patience}): "
                                                   f"{'served' if obs.get('status') == 'ok' else 'still unavailable'}")
        return obs
