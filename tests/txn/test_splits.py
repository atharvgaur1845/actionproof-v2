"""TXN splits: sizes, composition, quadruple structure, OOD definitions, disjointness and manifest hashes."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from actionproof.bench.generator_proc import DATA, WORKFLOWS, load_tasks
from actionproof.txn.env import make_env_txn
from actionproof.txn.tasks import (COMBOS, CORE, OOD_GROUPS, OOD_WF, SEEDS, check_disjoint, existing_combos, make_split,
                                   p4_compositions, task_line)


def load(s):
    return load_tasks(DATA / f"{s}.jsonl")


def test_sizes_and_groups():
    assert len(load("txn_test")) == 440 and len(load("txn_dev")) == 120 and len(load("txn_val")) == 120
    g = Counter(t.meta["txn"]["group"] for t in load("txn_test"))
    assert g["core"] == 200 and g["quad"] == 120 and all(g[x] == 20 for x in OOD_GROUPS)
    for s in ("txn_dev", "txn_val"):
        assert {t.meta["txn"]["group"] for t in load(s)} == {"core", "quad"}
    core = Counter(t.meta["fault_class"] for t in load("txn_test") if t.meta["txn"]["group"] == "core")
    assert set(core) == set(CORE) and set(core.values()) == {25}


def test_files_regenerate_identically_and_match_manifest():
    man = json.loads((DATA / "txn_manifest.json").read_text())
    for s in SEEDS:
        text = (DATA / f"{s}.jsonl").read_text()
        assert text == "".join(task_line(t) for t in make_split(s))
        assert man["files"][f"{s}.jsonl"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
        for line in text.splitlines():
            d = json.loads(line)
            m = man["tasks"][d["task_id"]]
            assert m["sha256"] == hashlib.sha256((line + "\n").encode()).hexdigest()
            assert m["template_id"] == d["meta"]["txn"]["template_id"]


def test_disjoint_from_every_existing_split():
    check_disjoint({s: load(s) for s in SEEDS})


def test_trained_workflows_outside_ood_workflow_group():
    for t in load("txn_test") + load("txn_dev"):
        if t.meta["txn"]["group"] == "ood_workflow":
            assert t.workflow in OOD_WF
        else:
            assert t.workflow in WORKFLOWS


def test_quadruples_share_po_prompt_seed_and_ack_text():
    quads: dict[str, list] = {}
    for t in load("txn_test"):
        if t.meta["txn"]["group"] == "quad":
            quads.setdefault(t.meta["txn"]["quad_id"].rsplit("-q", 1)[1], []).append(t)
    assert len(quads) == 15
    for ts in quads.values():
        assert len(ts) == 8 and len({(t.prompt, t.meta["env_seed"], t.meta["po_id"]) for t in ts}) == 1
        assert Counter((t.meta["txn"]["reality"], t.meta["txn"]["regime"]) for t in ts) == Counter(
            {(r, g): 1 for r in ("ok_phantom", "ok_committed", "timeout_committed", "timeout_uncommitted")
             for g in ("standard", "unresolvable")})
        step = ts[0].meta["txn"]["quad_step"]
        from actionproof.integrity.sim.agents import _args, _po
        from actionproof.integrity.sim.env import STEP_TOOL
        acks = {}
        for t in ts:
            env = make_env_txn(t, "q")
            for s in t.meta["requested"]:  # the agent-visible prefix up to the faulted write is identical
                o = env.call(STEP_TOOL[s], _args(s, _po(t), f"{s}-{t.meta['po_id']}"))
                if s == step:
                    break
            acks.setdefault(o["status"], set()).add(json.dumps(o, sort_keys=True))
            assert t.meta["txn"]["escalation_correct"] == (t.meta["txn"]["regime"] == "unresolvable")
        assert all(len(v) == 1 for v in acks.values())


def test_ood_definitions():
    p4 = p4_compositions()
    assert not set(OOD_WF.values()) & p4
    seen = existing_combos()
    assert seen and all(frozenset({ka, kb}) not in seen for _, ka, _, kb, _ in COMBOS)
    ev = [t for t in load("txn_test") if t.meta["txn"]["group"] == "ood_evidence"]
    assert all(t.meta["txn"]["reorder_fields"] and t.meta["txn"]["lag_first_reads"] == 2 for t in ev)
    dom = [t for t in load("txn_test") if t.meta["txn"]["group"] == "ood_domain"]
    assert all(t.prompt.startswith("Work order") and "vendor" in t.prompt for t in dom)
    names = {s["name"] for t in dom for s in t.initial_state["suppliers"]}
    from actionproof.bench.generator_proc import SUPPLIERS
    assert not names & set(SUPPLIERS)


def test_template_ids_and_labels():
    for t in load("txn_test"):
        tx = t.meta["txn"]
        g, wf, cls, reg = tx["template_id"].split("/")
        assert (g, wf, reg) == (tx["group"], t.workflow, tx["regime"]) and cls == t.meta["fault_class"]
        assert t.meta["env_version"] == "proc3"
        kinds = {f["kind"] for f in t.meta["faults"]}
        assert tx["regime"] == ("delayed" if "visibility_lag" in kinds else
                                "unresolvable" if tx["unavailable_reads"] else "standard")
