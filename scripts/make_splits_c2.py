"""Amendment 11 splits (PLAN.md section G2): two NEW untouched e-commerce splits in the v2 environment format. CPU.

    python scripts/make_splits_c2.py           # -> data/bench/c2/test_iid_c2.jsonl, data/bench/c2/v2_stress_c2.jsonl
    python scripts/make_splits_c2.py --check   # rebuild in memory and compare bytes with the files on disk

How they are built:
- `bench/generator.py` is imported, not edited. Two runtime entries are added to `generator.SPLITS` and
  `generator.SPLIT_SEEDS`, and tasks come from the unchanged `generator.make_task`:
  * `test_iid_c2` (seed 9): the `test_iid` recipe (F1 phantom, F3 partial, F4 timeout, F5 authorization, 40%
    benign), first 200 indices.
  * `v2_stress_c2` (seed 10): call-fault menu restricted to F4 timeout, no task faults. The generator still draws
    benign tasks, so indices are scanned in order and the first 100 tasks with a timeout are kept (the kept index
    is stored in meta.c2.source_index). Every timeout in these tasks is a timeout BEFORE commit.
- v2 format exactly as `scripts/make_bench_v2.py`: meta gets env_version 2 and precommit_timeouts
  (empty in test_iid_c2, every timeout call in v2_stress_c2).
- Entity disjointness. About 80% of the 4-digit customer, order and ticket id space is already used by the
  existing splits, so generated ids would collide. Every C-/O-/T- id in a new task is therefore renamed, through a
  per-task bijection, to an id that appears in no file under data/bench (excluding data/bench/c2) and in no other
  new task. Ids keep the generator's format (prefix plus 4 digits, drawn from a seeded shuffle of the free ids), so
  the task distribution is unchanged; only the labels differ. SKUs are a shared 100-value catalogue by design and
  are not renamed. The script asserts that task ids and C/O/T entity ids are disjoint from all existing splits,
  that each task still round-trips through `Task.from_json`, and that the reference policy solves it.
- Output is canonical JSON (sorted keys), one task per line. `--check` rebuilds against the current contents of
  data/bench, so it is only meaningful until another split is added there; after registration the frozen hashes
  (scripts/freeze_a11.py) are the reference.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actionproof.bench import generator as G  # noqa: E402
from actionproof.env.faults import Fault  # noqa: E402

OUT = ROOT / "data/bench/c2"
SPLITS_C2 = {
    "test_iid_c2": {"seed": 9, "n": 200, "cfg": {**G.SPLITS["test_iid"], "n": 200}},
    "v2_stress_c2": {"seed": 10, "n": 100, "cfg": {"n": 100, "call": [Fault.TIMEOUT], "task": [], "combo": False,
                                                   "rho": G.TRAIN_RHO}},
}
ID_RE = re.compile(r"(?<![A-Za-z0-9])([COT])-(\d{4})(?![A-Za-z0-9])")
REMAP_SEED = "actionproof/c2/entity-remap/v1"


def register() -> None:
    """Runtime entries in the generator's registries (generator.py itself is unchanged)."""
    for name, s in SPLITS_C2.items():
        G.SPLITS[name] = s["cfg"]
        G.SPLIT_SEEDS[name] = s["seed"]


def existing() -> tuple[set[str], dict[str, set[str]]]:
    task_ids: set[str] = set()
    ids: dict[str, set[str]] = {"C": set(), "O": set(), "T": set()}
    for f in sorted((ROOT / "data/bench").rglob("*.jsonl")):
        if OUT in f.parents:
            continue
        for line in f.open():
            d = json.loads(line)
            if "task_id" in d:
                task_ids.add(d["task_id"])
            for p, n in ID_RE.findall(json.dumps(d)):
                ids[p].add(f"{p}-{n}")
    return task_ids, ids


def raw_tasks() -> dict[str, list[dict]]:
    register()
    out: dict[str, list[dict]] = {}
    for name, s in SPLITS_C2.items():
        rows, i = [], 0
        while len(rows) < s["n"]:
            t = G.make_task(name, i)
            d = t.to_json()
            timeouts = [[x["tool"], x["k"]] for x in d["schedule"] if x["fault"] == Fault.TIMEOUT.value]
            if name == "v2_stress_c2" and not timeouts:
                i += 1
                continue
            d["meta"] = {**d["meta"], "env_version": 2,
                         "precommit_timeouts": timeouts if name == "v2_stress_c2" else [],
                         "c2": {"source_index": i, "entity_ids_remapped": True}}
            rows.append(d)
            i += 1
        out[name] = rows
    return out


def remap(tasks: dict[str, list[dict]], used: dict[str, set[str]]) -> dict[str, list[dict]]:
    rng = random.Random(REMAP_SEED)
    free = {p: [f"{p}-{n:04d}" for n in range(10_000) if f"{p}-{n:04d}" not in used[p]] for p in "COT"}
    for p in "COT":
        rng.shuffle(free[p])
    out: dict[str, list[dict]] = {}
    for name in SPLITS_C2:
        rows = []
        for d in tasks[name]:
            task_id = d.pop("task_id")
            text = json.dumps(d, sort_keys=True)
            olds = sorted({f"{p}-{n}" for p, n in ID_RE.findall(text)})
            mapping = {o: free[o[0]].pop() for o in olds}
            new = json.loads(ID_RE.sub(lambda m: mapping[m.group(0)], text))
            new["task_id"] = task_id
            new["goal"]["task_id"] = task_id
            rows.append(new)
        out[name] = rows
    return out


def check(tasks: dict[str, list[dict]], task_ids: set[str], used: dict[str, set[str]]) -> None:
    from actionproof.bench.generator import Task
    from actionproof.rewards.sanity_agents import task_oracle
    from actionproof.rollout.agenttune_rollout import episode_record, make_env
    seen: dict[str, set[str]] = {"C": set(), "O": set(), "T": set()}
    for name, rows in tasks.items():
        assert len(rows) == SPLITS_C2[name]["n"], name
        for d in rows:
            assert d["task_id"] not in task_ids, d["task_id"]
            ids = {f"{p}-{n}" for p, n in ID_RE.findall(json.dumps(d))}
            for x in ids:
                assert x not in used[x[0]], (d["task_id"], x)
            m = d["meta"]["ids"]
            for k in ("order", "ticket", "cust"):
                assert m[k] in d["prompt"], (d["task_id"], k)
            for x in ids:
                seen[x[0]].add(x)
            t = Task.from_json(json.loads(json.dumps(d)))
            assert t.to_json() == d, d["task_id"]
            env = make_env(t, f"chk-{t.task_id}")
            assert type(env).__name__ == "EcomEnvV2"
            task_oracle(env, t.workflow, m["order"], m["ticket"])
            assert episode_record(t, env)["vsc"] == 1, d["task_id"]
    n_ids = sum(len({f"{p}-{n}" for p, n in ID_RE.findall(json.dumps(d))}) for rows in tasks.values() for d in rows)
    assert n_ids == sum(len(v) for v in seen.values()), "an entity id is shared between two new tasks"
    stress = tasks["v2_stress_c2"]
    assert all(d["meta"]["precommit_timeouts"] for d in stress)
    assert all(not d["meta"]["precommit_timeouts"] for d in tasks["test_iid_c2"])


def build() -> dict[str, bytes]:
    task_ids, used = existing()
    tasks = remap(raw_tasks(), used)
    check(tasks, task_ids, used)
    return {name: "".join(json.dumps(d, sort_keys=True) + "\n" for d in rows).encode() for name, rows in tasks.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data = build()
    if a.check:
        bad = [n for n, b in data.items() if not (OUT / f"{n}.jsonl").exists() or (OUT / f"{n}.jsonl").read_bytes() != b]
        print("OK: c2 splits reproduce byte for byte" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, b in data.items():
        if (OUT / f"{name}.jsonl").exists() and (OUT / f"{name}.jsonl").read_bytes() != b:
            sys.exit(f"{name}.jsonl exists with different content; refusing to overwrite an untouched split")
        (OUT / f"{name}.jsonl").write_bytes(b)
    for name in data:
        rows = [json.loads(x) for x in (OUT / f"{name}.jsonl").read_text().splitlines()]
        faults: dict[str, int] = {}
        for d in rows:
            for f in d["meta"]["faults"] or ["benign"]:
                faults[f] = faults.get(f, 0) + 1
        print(name, len(rows), "tasks;", dict(sorted(faults.items())))


if __name__ == "__main__":
    main()
