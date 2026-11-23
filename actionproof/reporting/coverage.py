"""Evaluation coverage validator (docs/quality/EXPERIMENT_PIPELINE_AUDIT.md §C).

`check(path, split_file, samples)` checks one evaluation output, one JSON episode per line, against its split:

| Problem | Meaning |
|---|---|
| missing_tasks | split tasks with no episode |
| unexpected_tasks | episodes for tasks not in the split |
| missing_samples | a task lacks the greedy run (sample −1) or one of samples 0..n−1 |
| duplicate_episodes | the same (task, sample) recorded twice |
| missing_verdict | no `vsc` or no `score.steps` |
| unparsed_turns | the episode ended with no parseable tool call (ended_by = no_call), counted and reported |
| incomplete_rollouts | the episode never reported and did not hit max_turns or the context limit |

The last two are behaviours, not file defects: they are counted, and fail only when `strict_behaviour`.
`matrix(spec)` runs `check` over a whole registered matrix and returns (ok, report). Analysis scripts call it and stop
if `ok` is False, so no statistic is computed on an incomplete matrix.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

FILE_DEFECTS = ("missing_file", "missing_tasks", "unexpected_tasks", "missing_samples", "duplicate_episodes",
                "missing_verdict", "bad_json")


def check(path: Path, split_file: Path, samples: int, strict_behaviour: bool = False) -> dict:
    rep = {"path": str(path), "problems": Counter(), "examples": {}}
    if not path.exists():
        rep["problems"]["missing_file"] = 1
        return _done(rep, strict_behaviour)
    want = [json.loads(l)["task_id"] for l in split_file.read_text().splitlines() if l.strip()]
    seen: Counter = Counter()
    for i, line in enumerate(path.read_text().splitlines()):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            rep["problems"]["bad_json"] += 1
            continue
        key = (r.get("task_id"), r.get("sample"))
        seen[key] += 1
        if r.get("vsc") is None or not isinstance((r.get("score") or {}).get("steps"), dict):
            rep["problems"]["missing_verdict"] += 1
            rep["examples"].setdefault("missing_verdict", key)
        if r.get("ended_by") == "no_call":
            rep["problems"]["unparsed_turns"] += 1
        if r.get("ended_by") not in ("report", "max_turns", "context", "no_call", None):
            rep["problems"]["incomplete_rollouts"] += 1
    want_set = set(want)
    got_tasks = {t for t, _ in seen}
    for t in want_set - got_tasks:
        rep["problems"]["missing_tasks"] += 1
        rep["examples"].setdefault("missing_tasks", t)
    for t in got_tasks - want_set:
        rep["problems"]["unexpected_tasks"] += 1
        rep["examples"].setdefault("unexpected_tasks", t)
    for t in want_set & got_tasks:
        for j in [-1] + list(range(samples)):
            if seen[(t, j)] == 0:
                rep["problems"]["missing_samples"] += 1
                rep["examples"].setdefault("missing_samples", (t, j))
    dups = [k for k, n in seen.items() if n > 1]
    if dups:
        rep["problems"]["duplicate_episodes"] = len(dups)
        rep["examples"]["duplicate_episodes"] = dups[0]
    rep["n_episodes"] = sum(seen.values())
    rep["expected_episodes"] = len(want) * (1 + samples)
    return _done(rep, strict_behaviour)


def _done(rep: dict, strict: bool) -> dict:
    bad = [k for k in rep["problems"] if k in FILE_DEFECTS or (strict and rep["problems"][k])]
    rep["ok"] = not bad
    rep["problems"] = dict(rep["problems"])
    rep["examples"] = {k: list(v) if isinstance(v, tuple) else v for k, v in rep["examples"].items()}
    return rep


def matrix(spec: list[tuple[Path, Path, int]]) -> tuple[bool, list[dict]]:
    """spec = [(output jsonl, split file, samples), ...]."""
    reps = [check(p, s, n) for p, s, n in spec]
    return all(r["ok"] for r in reps), reps
