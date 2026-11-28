"""Deployment gate: candidate vs incumbent on fresh splits, under criteria fixed in a JSON file before any result.

Input: evaluation jsonl files (scripts/p1_eval.py or scripts/txn_eval.py records) for the candidate and the
incumbent, one or more seeds each, on the splits the criteria file names. Per task, every metric is the mean over
all of an arm's episodes (greedy and sampled, all seeds). Differences are candidate minus incumbent, paired by task,
with a cluster bootstrap (clusters: TXN template id where the task has one, else the task id).

Checks before any number:
- every evaluated task id is fresh: none was used to build corrections in any loop round (ledger);
- both arms cover exactly the same tasks on every split the criteria name.

Criteria (all must pass to promote):
- primary: verified completion on the pooled primary splits is non-inferior: lower 95% bound of the difference
  above -margin (and "superior" is reported when the lower bound is above 0);
- safety: for each safety metric (false completion report, unsafe final state), the upper 95% bound of the
  increase is below its margin;
- retention: on each retention (OOD) split, the lower 95% bound of the verified-completion difference is above
  -max_drop.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from actionproof.selfheal.envs import harm, spec_for, task_from_record

METRICS = ("vtc", "success", "false_completion", "unsafe_mutation")


def episode_metrics(rec: dict[str, Any]) -> dict[str, Any]:
    task = task_from_record(rec)
    spec = spec_for(task)
    rec = spec.annotate(rec)
    s = rec["score"]
    tx = rec["task"]["meta"].get("txn") or {}
    return {"task_id": task.task_id, "split": task.split, "cluster": tx.get("template_id") or task.task_id,
            "vtc": float(bool(s["vsc"])), "success": float(spec.success(rec)),
            "false_completion": float(bool(s["false_success"])), "unsafe_mutation": float(harm(s))}


def _read(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    with (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_arm(paths: list[str]) -> list[dict[str, Any]]:
    return [episode_metrics(r) for p in paths for r in _read(p)]


def per_task(eps: list[dict[str, Any]], metric: str, splits: set[str]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for e in eps:
        if e["split"] in splits:
            acc[e["task_id"]].append(e[metric])
    return {t: float(np.mean(v)) for t, v in acc.items()}


def cluster_bootstrap(diff: dict[str, float], cluster: dict[str, str], b: int, seed: int) -> dict[str, Any]:
    tasks = sorted(diff)
    cl = sorted({cluster[t] for t in tasks})
    ci = {c: i for i, c in enumerate(cl)}
    member = np.array([ci[cluster[t]] for t in tasks])
    x = np.array([diff[t] for t in tasks])
    sums = np.bincount(member, weights=x, minlength=len(cl))
    counts = np.bincount(member, minlength=len(cl)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cl), size=(b, len(cl)))
    w = np.zeros((b, len(cl)))
    for j in range(len(cl)):
        w[:, j] = (draws == j).sum(axis=1)
    boot = (w @ sums) / (w @ counts)
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return {"diff": float(x.mean()), "ci95": [lo, hi], "n_tasks": len(tasks), "n_clusters": len(cl)}


def _compare(cand: list[dict], inc: list[dict], metric: str, splits: set[str], cfg: dict[str, Any]) -> dict[str, Any]:
    a, b = per_task(cand, metric, splits), per_task(inc, metric, splits)
    if set(a) != set(b) or not a:
        raise ValueError(f"{metric} on {sorted(splits)}: arms cover different tasks ({len(a)} vs {len(b)})")
    cluster = {e["task_id"]: e["cluster"] for e in cand + inc}
    r = cluster_bootstrap({t: a[t] - b[t] for t in a}, cluster, cfg["bootstrap"]["B"], cfg["bootstrap"]["seed"])
    r.update(candidate=float(np.mean(list(a.values()))), incumbent=float(np.mean(list(b.values()))))
    return r


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decide(criteria_path: Path, candidate: list[str], incumbent: list[str], ledger: Any | None) -> dict[str, Any]:
    cfg = json.loads(criteria_path.read_text())
    cand, inc = load_arm(candidate), load_arm(incumbent)
    ids = {e["task_id"] for e in cand + inc}
    fresh = {"checked": ledger is not None}
    if ledger is not None:
        ledger.assert_fresh(ids)  # raises: a gate on correction data is never computed
        fresh["n_tasks"] = len(ids)
    named = set(cfg["primary"]["splits"]) | set(cfg["retention"]["splits"])
    seen = {e["split"] for e in cand} & {e["split"] for e in inc}
    missing = sorted(named - seen)
    if missing:
        raise ValueError(f"splits named by the criteria but not evaluated in both arms: {missing}")
    out: dict[str, Any] = {"criteria_file": str(criteria_path), "criteria_sha256": _sha(criteria_path),
                           "criteria_version": cfg.get("version"), "freshness": fresh,
                           "inputs": {"candidate": {p: _sha(p) for p in candidate},
                                      "incumbent": {p: _sha(p) for p in incumbent}},
                           "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "criteria": {}}
    prim = set(cfg["primary"]["splits"])
    r = _compare(cand, inc, cfg["primary"]["metric"], prim, cfg)
    m = cfg["primary"]["noninferiority_margin"]
    out["criteria"]["primary_noninferior"] = {**r, "rule": f"lower95 > -{m}", "pass": r["ci95"][0] > -m,
                                              "superior": r["ci95"][0] > 0}
    for s in cfg["safety"]:
        r = _compare(cand, inc, s["metric"], prim, cfg)
        out["criteria"][f"safety_{s['metric']}"] = {**r, "rule": f"upper95 < {s['max_increase']}",
                                                    "pass": r["ci95"][1] < s["max_increase"]}
    for sp in cfg["retention"]["splits"]:
        r = _compare(cand, inc, cfg["retention"]["metric"], {sp}, cfg)
        d = cfg["retention"]["max_drop"]
        out["criteria"][f"retention_{sp}"] = {**r, "rule": f"lower95 > -{d}", "pass": r["ci95"][0] > -d}
    out["decision"] = "promote" if all(c["pass"] for c in out["criteria"].values()) else "reject"
    return out
