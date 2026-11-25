"""ProcTask and loader; vendored from actionproof/bench/generator_proc.py (dataclass and load only)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProcTask:
    task_id: str
    split: str
    workflow: str
    prompt: str
    initial_state: dict[str, list[dict[str, Any]]]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "split": self.split, "workflow": self.workflow, "prompt": self.prompt,
                "initial_state": self.initial_state, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ProcTask":
        return cls(d["task_id"], d["split"], d["workflow"], d["prompt"], d["initial_state"], d["meta"])


def load_tasks(path: str | Path) -> list[ProcTask]:
    """Load a .jsonl task file (one task per line) or a .json list."""
    p = Path(path)
    if p.suffix == ".json":
        return [ProcTask.from_json(d) for d in json.loads(p.read_text())]
    return [ProcTask.from_json(json.loads(line)) for line in p.read_text().splitlines() if line.strip()]
