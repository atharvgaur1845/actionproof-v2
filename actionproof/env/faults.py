"""Fault catalogue and per-task fault schedules (spec §9).

A schedule maps (tool_name, k) -> fault, where k counts calls to that tool within the
episode. Faults therefore fire deterministically on e.g. the *first* `issue_refund` call,
and a retry (k=1) behaves normally, which keeps recovery possible and every episode
replayable from its seed.

Call-level faults change what a write does and/or what the agent is told:
  F1 phantom   - nothing commits, the agent sees a byte-identical success ack
  F2 stale     - nothing commits, the ack echoes an earlier refund's id (held out)
  F3 partial   - only the first sub-effect of a composite write commits, ack says success
  F4 timeout   - everything commits, the agent sees a timeout error
  F8 ambiguous - commit per `commits` flag, ack text is vague (held out; text generated later)
Task-level conditions (F5 authorization, F6 stale status endpoint, F7 injection) live on the
task itself, not in the call schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Fault(str, Enum):
    PHANTOM = "F1_phantom"
    STALE = "F2_stale"
    PARTIAL = "F3_partial"
    TIMEOUT = "F4_timeout"
    AMBIGUOUS = "F8_ambiguous"


TRAIN_CALL_FAULTS = frozenset({Fault.PHANTOM, Fault.PARTIAL, Fault.TIMEOUT})
HELD_OUT_CALL_FAULTS = frozenset({Fault.STALE, Fault.AMBIGUOUS})

# Which faults actually commit the write. The asymmetry between "commits" and "what the
# agent is told" is the whole point of the benchmark: observation != proof of state.
COMMITS = {
    Fault.PHANTOM: False,
    Fault.STALE: False,
    Fault.PARTIAL: True,   # partially: only the first sub-effect
    Fault.TIMEOUT: True,
}


@dataclass(frozen=True)
class FaultSchedule:
    by_call: dict[tuple[str, int], Fault] = field(default_factory=dict)
    # F8 needs to know whether the ambiguous ack hides a commit or a no-op.
    ambiguous_commits: dict[tuple[str, int], bool] = field(default_factory=dict)

    def fault_for(self, tool: str, k: int) -> Fault | None:
        return self.by_call.get((tool, k))

    def to_json(self) -> list[dict]:
        return [
            {"tool": t, "k": k, "fault": f.value,
             **({"commits": self.ambiguous_commits[(t, k)]} if f is Fault.AMBIGUOUS else {})}
            for (t, k), f in sorted(self.by_call.items())
        ]

    @classmethod
    def from_json(cls, items: list[dict]) -> "FaultSchedule":
        by_call, amb = {}, {}
        for it in items:
            key = (it["tool"], int(it["k"]))
            by_call[key] = Fault(it["fault"])
            if by_call[key] is Fault.AMBIGUOUS:
                amb[key] = bool(it["commits"])
        return cls(by_call=by_call, ambiguous_commits=amb)
