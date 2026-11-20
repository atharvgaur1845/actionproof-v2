"""OpsSim: cross-domain transfer environments for A17 (IT operations, healthcare administration).

Each domain module (`it_ops`, `care_admin`) is self-contained and follows the ProcureSim pattern
(`actionproof/integrity/sim/`): one write chokepoint applies a per-(tool, attempt) fault plan, the agent never
sees the plan, and a state verifier reads only the final state. The shared failure taxonomy is a specification
(docs/stage2/OPSIM_SPEC.md), not shared code, so a bug in one domain cannot silently enter the other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[3] / "data" / "bench" / "opsim"

FAULT_KINDS = ("ok", "phantom", "partial", "lost_ack", "precommit_timeout", "dup_delivery", "wrong_target",
               "replica_lag")
TRANSFER_CLASSES = ("clean", "phantom", "partial", "lost_ack", "duplicate", "never_commit")
OUT_OF_TAXONOMY = ("wrong_target", "stale_read", "conflicting", "unauthorised")
TIMEOUT = {"status": "error", "error": "timeout", "detail": "The request timed out. Its outcome is unknown."}


@dataclass
class OpsTask:
    task_id: str
    split: str
    domain: str
    template_id: str
    prompt: str
    initial_state: dict[str, list[dict[str, Any]]]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def workflow(self) -> str:
        return self.template_id

    def to_json(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "split": self.split, "domain": self.domain, "template_id": self.template_id,
                "prompt": self.prompt, "initial_state": self.initial_state, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "OpsTask":
        return cls(d["task_id"], d["split"], d["domain"], d["template_id"], d["prompt"], d["initial_state"], d["meta"])


def load_tasks(path: str | Path) -> list[OpsTask]:
    return [OpsTask.from_json(json.loads(l)) for l in Path(path).read_text().splitlines() if l.strip()]


def write_tasks(path: str | Path, tasks: list[OpsTask]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in tasks))
