"""Fresh task splits for the self-healing loop (built once; `check()` verifies them).

Loop pools (the agent is run on them, failures are corrected, corrections are trained on):
- `sh_pool_r{1,2}_p1full` (P1 full episodes, 40 per workflow), `sh_pool_r{1,2}_p1rec` (P1 recovery starts, 120),
  `sh_pool_r{1,2}_txn` (TXN v1: 8 core classes x 15, plus 8 counterfactual groups of 8 = 184 tasks).

Gate splits (used only by the deployment gate; never by diagnose, correct or curate):
- primary: `sh_gate_p1full` (120), `sh_gate_p1rec` (120), `sh_gate_txn` (8 core classes x 20, plus 15 counterfactual groups of 8 = 280);
- retention (OOD): `sh_gate_txn_ood` (6 OOD groups x 20), `sh_gate_p1wf4` (unseen workflow, 60),
  `sh_gate_p1combo` (two faults, 60).

Generators are the existing ones (`actionproof/p1/build.py`, `actionproof/stage3/tasks.py`, `actionproof/txn/tasks.py`)
with new seeds. Task ids, PO ids and prompts are checked disjoint from every file in data/bench/proc, including
txn_* and p1_*, and from each other. A manifest with per-file sha256 is written next to the files.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from actionproof.bench.generator_proc import DATA, ProcTask

MANIFEST = DATA / "selfheal_manifest.json"
SEEDS = {
    "sh_pool_r1_p1full": 29112601, "sh_pool_r1_p1rec": 29112602, "sh_pool_r1_txn": 29112603,
    "sh_pool_r2_p1full": 29112611, "sh_pool_r2_p1rec": 29112612, "sh_pool_r2_txn": 29112613,
    "sh_gate_p1full": 29112621, "sh_gate_p1rec": 29112622, "sh_gate_txn": 29112623,
    "sh_gate_txn_ood": 29112624, "sh_gate_p1wf4": 29112625, "sh_gate_p1combo": 29112626,
}
GATE_PRIMARY = ("sh_gate_p1full", "sh_gate_p1rec", "sh_gate_txn")
GATE_RETENTION = ("sh_gate_txn_ood", "sh_gate_p1wf4", "sh_gate_p1combo")


def _txn(name: str, seed: int, n_core: int, n_quad: int, ood: int) -> list[ProcTask]:
    from actionproof.txn import tasks as T
    rng, out = random.Random(seed), []
    for cls in T.CORE if n_core else ():
        for j in range(n_core):
            out.append(T.core_task(name, len(out), rng, cls, T.TRAINED_WF[j % len(T.TRAINED_WF)]))
    for q in range(n_quad):
        out += T.quad_tasks(name, len(out), rng, T.TRAINED_WF[q % len(T.TRAINED_WF)], q)
    for g in T.OOD_GROUPS if ood else ():
        for j in range(ood):
            out.append(T.ood_task(name, len(out), rng, g, j))
    return out


def make(name: str) -> list[ProcTask]:
    from actionproof.p1 import build as P1
    from actionproof.stage3.tasks import make_pool
    seed = SEEDS[name]
    kind = name.split("_")[-1]
    if kind == "p1full":
        return P1._balanced(name, seed, 40)
    if kind == "p1rec":
        return make_pool(name, 120, seed)
    if kind == "p1wf4":
        return P1._wf4(name, seed, 60)
    if kind == "p1combo":
        return P1._combo(name, seed, 60)
    if name == "sh_gate_txn":
        return _txn(name, seed, 20, 15, 0)
    if name == "sh_gate_txn_ood":
        return _txn(name, seed, 0, 0, 20)
    if kind == "txn":
        return _txn(name, seed, 15, 8, 0)
    raise KeyError(name)


def line(t: ProcTask) -> str:
    return json.dumps(t.to_json(), sort_keys=True) + "\n"


def _existing() -> tuple[set[str], set[str], set[str]]:
    tids, pos, prompts = set(), set(), set()
    for f in sorted(DATA.glob("*.jsonl")):
        if f.name.startswith("sh_"):
            continue
        for ln in f.read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                tids.add(d.get("task_id"))
                prompts.add(d.get("prompt"))
                pos |= {p["po_id"] for p in d.get("initial_state", {}).get("pos", [])}
    return tids, pos, prompts


def check_disjoint(splits: dict[str, list[ProcTask]]) -> None:
    tids, pos, prompts = _existing()
    seen: set[str] = set()
    po_owner: dict[str, str] = {}
    prompt_owner: dict[str, str] = {}
    for tasks in splits.values():
        for t in tasks:
            owner = (t.meta.get("txn") or {}).get("quad_id") or t.task_id
            assert t.task_id not in tids and t.task_id not in seen, f"task id {t.task_id} collides"
            seen.add(t.task_id)
            assert t.prompt not in prompts, f"{t.task_id}: prompt collides with an existing split"
            assert prompt_owner.setdefault(t.prompt, owner) == owner, f"{t.task_id}: prompt shared across tasks"
            for p in t.initial_state["pos"]:
                assert p["po_id"] not in pos, f"{t.task_id}: PO id {p['po_id']} collides with an existing split"
                assert po_owner.setdefault(p["po_id"], owner) == owner, f"{t.task_id}: PO id reused"


def write() -> dict[str, Any]:
    splits = {n: make(n) for n in SEEDS}
    check_disjoint(splits)
    paths = {n: DATA / f"{n}.jsonl" for n in splits}
    clash = [str(p) for p in list(paths.values()) + [MANIFEST] if p.exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite: {clash}")
    man: dict[str, Any] = {"module": "actionproof/selfheal/splits.py", "seeds": SEEDS, "files": {}}
    for n, ts in splits.items():
        text = "".join(line(t) for t in ts)
        paths[n].write_text(text)
        man["files"][paths[n].name] = {"n": len(ts), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                       "role": "gate_primary" if n in GATE_PRIMARY else
                                       "gate_retention" if n in GATE_RETENTION else "loop_pool"}
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return man["files"]


def check() -> int:
    from actionproof.bench.generator_proc import load_tasks
    man = json.loads(MANIFEST.read_text())
    bad = []
    for n in SEEDS:
        text = (DATA / f"{n}.jsonl").read_text()
        if hashlib.sha256(text.encode()).hexdigest() != man["files"][f"{n}.jsonl"]["sha256"]:
            bad.append(f"{n}: sha256 differs from the manifest")
        if text != "".join(line(t) for t in make(n)):
            bad.append(f"{n}: does not regenerate byte-identically")
    try:
        check_disjoint({n: load_tasks(DATA / f"{n}.jsonl") for n in SEEDS})
    except AssertionError as e:
        bad.append(f"disjointness: {e}")
    print("OK: self-heal splits match the manifest and are disjoint" if not bad else "MISMATCH: " + "; ".join(bad))
    return 1 if bad else 0
