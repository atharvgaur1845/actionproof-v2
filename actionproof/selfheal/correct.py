"""Corrections: cut a failed episode before its first bad turn, let a teacher continue, keep only verified results.

For each failed episode:
1. Replay the whole recorded episode and check it matches the record (otherwise skip: `replay_mismatch`).
2. Candidate cuts, in order: the primary cut (the last turn boundary before the taxonomy's `first_bad_idx`), then
   every earlier turn boundary down to the start. Later cuts keep more of the agent's own behaviour (more on-policy
   prefixes); earlier cuts are a fallback when the primary prefix leaves no safe way forward or too few turns.
3. At each cut the teacher continues on a freshly replayed env. The continuation is then replayed from scratch
   with the same parser and scored by the verifier (`replay.verify_continuation`). The first cut whose
   continuation reaches the env spec's verified-correct outcome is kept.

Splits whose name contains "test" are refused: corrections are built from development or loop-pool data only.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from actionproof.selfheal import taxonomy
from actionproof.selfheal.envs import EnvSpec, spec_for, surface_of, task_from_record
from actionproof.selfheal.replay import ReplayMismatch, parser, replay, turn_before, verify_continuation
from actionproof.selfheal.teachers import Context, Teacher


class SplitRefused(RuntimeError):
    pass


def refuse_test(rec: dict[str, Any]) -> None:
    split = str(rec["task"].get("split", ""))
    if "test" in split:
        raise SplitRefused(f"{rec['task']['task_id']}: split {split} is a test split; corrections use dev data only")


def correct_episode(rec: dict[str, Any], teacher: Teacher, spec: EnvSpec | None = None, parse=None,
                    max_cuts: int | None = None) -> dict[str, Any]:
    """One failed episode -> a result dict with `status` accepted | rejected | skipped and, when accepted, the
    corrected conversation and the verifier's score."""
    refuse_test(rec)
    task = task_from_record(rec)
    spec = spec or spec_for(task)
    parse = parse or parser()
    base = {"task_id": task.task_id, "split": task.split, "sample": rec.get("sample"), "env": spec.name,
            "teacher": teacher.name, "source_tag": rec.get("tag")}
    if surface_of(task):
        return {**base, "status": "skipped", "reason": "renamed_surface"}  # teachers act on canonical names
    succ = spec.success(spec.annotate(rec))
    cls = taxonomy.classify(rec, succ)
    base.update(primary=cls["primary"], first_bad_idx=cls["first_bad_idx"])
    if succ:
        return {**base, "status": "skipped", "reason": "not_a_failure"}
    try:
        full = replay(rec, spec, check=True, parse=parse)
    except ReplayMismatch as e:
        return {**base, "status": "skipped", "reason": "replay_mismatch", "detail": str(e)}
    primary = turn_before(full.turn_ends, cls["first_bad_idx"], full.turns)
    cuts = [primary] + list(range(primary - 1, -1, -1))
    tried = []
    for k, cut in enumerate(cuts[:max_cuts]):
        rep = replay(rec, spec, upto_turn=cut, check=True, parse=parse)
        ctx = Context(spec=spec, rep=rep)
        teacher.continue_from(ctx)
        if not ctx.actions:
            tried.append({"cut": cut, "result": "no_actions"})
            continue
        try:
            out, conv = verify_continuation(rec, spec, cut, ctx.actions, parse=parse)
        except ReplayMismatch as e:
            tried.append({"cut": cut, "result": "verify_replay_mismatch", "detail": str(e)})
            continue
        ok = spec.success(out)
        tried.append({"cut": cut, "result": "verified" if ok else "not_verified",
                      "report": (out.get("report") or {}).get("status"), "verdicts": out["score"]["steps"]})
        if not ok:
            continue
        n_prefix = len(rep.conv)
        rejected = rec["conversation"][n_prefix] if n_prefix < len(rec["conversation"]) else None
        return {**base, "status": "accepted", "cut": cut, "cut_kind": "primary" if k == 0 else "backoff",
                "primary_cut": primary, "recorded_turns": full.turns, "tries": tried,
                "prefix_len": n_prefix, "conversation": conv, "rejected_turn": rejected,
                "verified": {"vsc": out["score"]["vsc"], "goal_met": out["score"]["goal_met"],
                             "report": (out.get("report") or {}).get("status"), "verdicts": out["score"]["steps"],
                             "txn": {k2: out["txn"][k2] for k2 in ("vtc", "str", "fcr", "urr", "harm")}
                             if "txn" in out else None},
                "teacher_turns": sum(m["role"] == "assistant" for m in conv[n_prefix:])}
    return {**base, "status": "rejected", "reason": "no_cut_verified", "primary_cut": primary,
            "recorded_turns": full.turns, "tries": tried}


def run(records: Iterable[dict[str, Any]], teacher: Teacher, out_path: Path, max_cuts: int | None = None,
        exclude_task_ids: set[str] | None = None) -> dict[str, Any]:
    """Correct every failed episode; write accepted and non-accepted results to `out_path` (jsonl) and return stats."""
    parse = parser()
    stats: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            if exclude_task_ids and rec["task"]["task_id"] in exclude_task_ids:
                stats["excluded_by_ledger"] += 1
                continue
            r = correct_episode(rec, teacher, parse=parse, max_cuts=max_cuts)
            if r["status"] == "skipped" and r.get("reason") == "not_a_failure":
                stats["successes_skipped"] += 1
                continue
            stats[r["status"]] += 1
            if r["status"] != "accepted":
                stats[f"{r['status']}:{r.get('reason')}"] += 1
            else:
                stats[f"accepted:{r['cut_kind']}"] += 1
                stats[f"accepted_outcome:{r['verified']['report']}"] += 1
            by_class.setdefault(r.get("primary", "?"), Counter())[r["status"]] += 1
            f.write(json.dumps(r, default=str) + "\n")
    return {"counts": dict(stats), "by_primary_class": {k: dict(v) for k, v in sorted(by_class.items())}}
