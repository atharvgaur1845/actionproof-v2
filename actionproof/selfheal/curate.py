"""Curate verified corrections into AgentTune/TRL training data.

Steps, in order:
1. Keep accepted corrections only (each one already passed simulator replay and the verifier).
2. Hold out by task id: a task goes to the hold-out set iff sha256(seed:task_id) falls below `holdout_frac`, so all
   corrections of a task land on one side.
3. Deduplicate with CuratorKIT, as `actionproof/p1/build.py` does: ExactDeduplicator, then MinHashDeduplicator
   (threshold 0.95), over (request + agent prefix actions, teacher continuation). Without CuratorKIT (local tests),
   an internal exact + shingle-Jaccard pass stands in and the stats say so.
4. Balance: at most `per_task_cap` corrections per task, then at most `per_class_cap` per primary failure class,
   in a seeded hash order (no look at outcomes).
5. Write
   - `sft_train.jsonl` / `sft_holdout.jsonl`: one row per teacher assistant turn, `{"prompt": messages before it,
     "completion": [that turn]}` (the format of `actionproof/train/rft.py`; loss on the completion only, so the
     agent's own prefix turns are context, not targets);
   - `dpo_train.jsonl` / `dpo_holdout.jsonl`: at the primary cut, `{"prompt", "chosen": [teacher's first turn],
     "rejected": [the agent's original turn]}` when the two differ (the format of `actionproof/train/dpo.py`);
   - `stats.json`.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CurateConfig:
    holdout_frac: float = 0.2
    seed: int = 20261129
    per_class_cap: int = 150
    per_task_cap: int = 2
    minhash_threshold: float = 0.95
    use_curatorkit: bool = True


def _h(*parts: Any) -> float:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def is_holdout(task_id: str, cfg: CurateConfig) -> bool:
    return _h(cfg.seed, "holdout", task_id) < cfg.holdout_frac


def _texts(r: dict[str, Any]) -> tuple[str, str]:
    conv, n = r["conversation"], r["prefix_len"]
    prefix = " | ".join(m["content"] for m in conv[2:n] if m["role"] == "assistant")
    cont = " | ".join(m["content"] for m in conv[n:] if m["role"] == "assistant")
    return f"{conv[1]['content']}\n{prefix}", cont


def _shingles(s: str, k: int = 5) -> set[str]:
    s = re.sub(r"\s+", " ", s)
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def dedup(rows: list[dict[str, Any]], cfg: CurateConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts = [_texts(r) for r in rows]
    if cfg.use_curatorkit:
        from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
        from curatorkit.schema import DataSample
        samples = [DataSample(id=str(i), source_uri=f"selfheal/{r['task_id']}", instruction=ins, output=out,
                              task_type="instruction_following") for i, (r, (ins, out)) in enumerate(zip(rows, texts))]
        ex = ExactDeduplicator().run(samples)
        mh = MinHashDeduplicator(threshold=cfg.minhash_threshold).run(ex)
        keep = sorted(int(s.id) for s in mh)
        return [rows[i] for i in keep], {"dedup": "curatorkit", "exact_removed": len(samples) - len(ex),
                                         "minhash_removed": len(ex) - len(mh)}
    seen, kept, sh, exact, near = set(), [], [], 0, 0
    for r, (ins, out) in zip(rows, texts):
        key = ins + "\x00" + out
        if key in seen:
            exact += 1
            continue
        seen.add(key)
        s = _shingles(key)
        if any(len(s & t) / max(1, len(s | t)) >= cfg.minhash_threshold for t in sh):
            near += 1
            continue
        sh.append(s)
        kept.append(r)
    return kept, {"dedup": "internal (CuratorKIT not importable)", "exact_removed": exact, "near_removed": near}


def balance(rows: list[dict[str, Any]], cfg: CurateConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    order = sorted(rows, key=lambda r: _h(cfg.seed, "order", r["task_id"], r.get("source_tag"), r.get("sample")))
    per_task: Counter[str] = Counter()
    step1 = []
    for r in order:
        if per_task[r["task_id"]] < cfg.per_task_cap:
            per_task[r["task_id"]] += 1
            step1.append(r)
    per_class: Counter[str] = Counter()
    out = []
    for r in step1:
        if per_class[r["primary"]] < cfg.per_class_cap:
            per_class[r["primary"]] += 1
            out.append(r)
    return out, {"per_task_cap_removed": len(rows) - len(step1), "per_class_cap_removed": len(step1) - len(out)}


def sft_rows(r: dict[str, Any]) -> list[dict[str, Any]]:
    conv, n = r["conversation"], r["prefix_len"]
    meta = {"task_id": r["task_id"], "failure_class": r["primary"], "cut_kind": r["cut_kind"],
            "source_tag": r.get("source_tag"), "teacher": r["teacher"], "env": r["env"]}
    return [{"prompt": conv[:j], "completion": [conv[j]], **meta}
            for j in range(n, len(conv)) if conv[j]["role"] == "assistant"]


def dpo_pair(r: dict[str, Any]) -> dict[str, Any] | None:
    rej = r.get("rejected_turn")
    if r["cut_kind"] != "primary" or not rej or rej["role"] != "assistant":
        return None
    chosen = r["conversation"][r["prefix_len"]]
    if chosen["content"].strip() == rej["content"].strip():
        return None
    return {"prompt": r["conversation"][:r["prefix_len"]], "chosen": [chosen], "rejected": [rej],
            "task_id": r["task_id"], "kind": r["primary"], "turn": r["prefix_len"], "source_tag": r.get("source_tag")}


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build(rows: list[dict[str, Any]], out_dir: Path, cfg: CurateConfig = CurateConfig()) -> dict[str, Any]:
    acc = [r for r in rows if r.get("status") == "accepted"]
    stats: dict[str, Any] = {"config": asdict(cfg), "input_rows": len(rows), "accepted": len(acc)}
    sides = {"train": [r for r in acc if not is_holdout(r["task_id"], cfg)],
             "holdout": [r for r in acc if is_holdout(r["task_id"], cfg)]}
    out_dir.mkdir(parents=True, exist_ok=True)
    for side, rs in sides.items():
        d, ds = dedup(rs, cfg)
        b, bs = balance(d, cfg)
        sft = [x for r in b for x in sft_rows(r)]
        dpo = [p for p in (dpo_pair(r) for r in b) if p]
        _write(out_dir / f"sft_{side}.jsonl", sft)
        _write(out_dir / f"dpo_{side}.jsonl", dpo)
        stats[side] = {"corrections_in": len(rs), **ds, **bs, "corrections_kept": len(b), "sft_rows": len(sft),
                       "dpo_pairs": len(dpo), "tasks": len({r["task_id"] for r in b}),
                       "by_class": dict(Counter(r["primary"] for r in b)),
                       "by_cut_kind": dict(Counter(r["cut_kind"] for r in b)),
                       "by_outcome": dict(Counter(r["verified"]["report"] for r in b))}
        stats[f"{side}_task_ids"] = sorted({r["task_id"] for r in rs})
    assert not set(stats["train_task_ids"]) & set(stats["holdout_task_ids"]), "hold-out leaked into train"
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats
