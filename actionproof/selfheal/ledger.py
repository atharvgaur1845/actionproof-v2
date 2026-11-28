"""Loop ledger: which task ids each round used, for what, so no round is evaluated on data that built it.

Roles:
- `diagnose`: episodes classified (descriptive; does not by itself make a task unusable for evaluation);
- `correct`: failed episodes a teacher continued (their tasks fed corrections);
- `train`, `holdout`: tasks whose corrections went into the SFT/DPO data, or into its held-out part.

`loop_task_ids()` is every task id with role correct, train or holdout in any round; the gate refuses an evaluation
file containing one of them (`assert_fresh`). Gate splits are registered with `register_gate_split`; `correct` skips
their tasks and `use` refuses to record them as loop data, so a gate split can never become correction data after
the fact. Loop-pool tasks may be corrected again in a later round (that is training data, not evaluation).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROTECTING = ("correct", "train", "holdout")


class LedgerViolation(RuntimeError):
    pass


@dataclass
class Ledger:
    path: Path
    rounds: dict[str, dict[str, Any]] = field(default_factory=dict)
    gate_splits: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        d = json.loads(p.read_text())
        return cls(path=p, rounds=d.get("rounds", {}), gate_splits=d.get("gate_splits", {}))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"rounds": self.rounds, "gate_splits": self.gate_splits}, indent=1,
                                        sort_keys=True) + "\n")

    def use(self, round_: str, role: str, task_ids: Iterable[str], sources: Iterable[str] = ()) -> None:
        ids = sorted(set(task_ids))
        clash = self.gate_task_ids() & set(ids)
        if clash and role in PROTECTING:
            raise LedgerViolation(f"{len(clash)} gate-split task ids would become {role} data, e.g. {sorted(clash)[:3]}")
        r = self.rounds.setdefault(round_, {})
        entry = r.setdefault(role, {"task_ids": [], "sources": [], "updated": None})
        entry["task_ids"] = sorted(set(entry["task_ids"]) | set(ids))
        entry["sources"] = sorted(set(entry["sources"]) | {str(s) for s in sources})
        entry["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def protected_task_ids(self) -> set[str]:
        return {t for r in self.rounds.values() for role in PROTECTING for t in r.get(role, {}).get("task_ids", [])} \
            | self.gate_task_ids()

    def loop_task_ids(self) -> set[str]:
        return {t for r in self.rounds.values() for role in PROTECTING for t in r.get(role, {}).get("task_ids", [])}

    def gate_task_ids(self) -> set[str]:
        return {t for g in self.gate_splits.values() for t in g["task_ids"]}

    def register_gate_split(self, name: str, path: str | Path) -> None:
        text = Path(path).read_text()
        ids = [json.loads(l)["task_id"] for l in text.splitlines() if l.strip()]
        clash = self.loop_task_ids() & set(ids)
        if clash:
            raise LedgerViolation(f"gate split {name} overlaps loop data: {sorted(clash)[:3]}")
        self.gate_splits[name] = {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "task_ids": sorted(ids)}

    def assert_fresh(self, task_ids: Iterable[str]) -> None:
        clash = self.loop_task_ids() & set(task_ids)
        if clash:
            raise LedgerViolation(f"{len(clash)} evaluation tasks were used to build corrections, e.g. "
                                  f"{sorted(clash)[:3]}")
