"""P4 (PREREGISTRATION_P4.md): demonstration-level randomisation of the tool surface and of the step composition.

Dw (P1 arm D's warm start, `slurm/tasks/1153_p1_dwarm.sh`) is SFT on `proc_demo_rows` (1,151 rows) plus
`p1_recovery_rows` (3,810 rows). Dw_aug changes only *which* demonstrations those rows contain:

1. **Surface.** The P1 recovery episodes are regenerated exactly as `actionproof/p1/build.py::_episode` does
   (scripted oracle on the canonical env, actions replayed). Half of the tasks, chosen by `surface.choose(task_id,
   None)` (a fixed hash, so no selection), are rendered on one of SR1's synonym surfaces
   (`actionproof/stage4/surface.py`): system prompt, tool schema, actions and tool results all use that surface. No
   surface shares a name with the held-out S4 API surface (asserted in surface.py).
2. **Composition.** New oracle demonstrations on step sequences never used in training before, none of which
   contains order -> receive -> pay or the held-out four-step workflow: order_receive, order_pay, pay_only,
   receive_only, order_only (40 tasks each, fresh seed 1601, ids and PO ids checked against every split, prompt text
   checked against every split). Half of them are also rendered on a surface.
3. **Everything in Dw is kept.** All 1,151 original warm-start rows stay. The composition demonstrations are added
   on top, so Dw_aug has more rows than Dw (stated in the registration). Matching the row count was tried first and
   discarded before registration: it dropped 217 of the 240 original warm-start episodes, which would have changed the
   arm in a second, unintended way.

Only verified teacher episodes are kept (the same filters as P1: goal met or the registered failed report on
never-commit, idempotency keys never reused after a refund).

    python -m actionproof.p4.build      # writes data/rft/p4_aug_rows.jsonl and .stats.json; refuses to overwrite
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/rft/p4_aug_rows.jsonl"
DW_ROWS = 4961
SEED = 1601
AUG_WF = {"order_receive": ("order", "receive"), "order_pay": ("order", "pay"), "pay_only": ("pay",),
          "receive_only": ("receive",), "order_only": ("order",)}
WORLD = {"order_receive": "order_prepay", "order_pay": "order_prepay", "order_only": "order_prepay",
         "receive_only": "receive_pay", "pay_only": "pay_post"}
FAULT_MIX = (("clean", 0.20), ("phantom", 0.25), ("partial", 0.15), ("lost_ack", 0.15), ("precommit_timeout", 0.15),
             ("dup_delivery", 0.10))


def _prompt(wf: str, po: dict, sup: str, item: str) -> str:
    head = (f"Purchase order {po['po_id']} is approved: {po['qty']} x {item} ({po['sku']}) from {sup} at "
            f"{po['unit_price_cents']} cents each, total {po['amount_cents']} cents. ")
    return head + {
        "order_receive": f"Place the order with the supplier, then record the goods receipt when all {po['qty']} units arrive. Do not pay yet.",
        "order_pay": "Place the order with the supplier and pay the supplier the full amount in advance. The journal entry is handled separately.",
        "pay_only": "The goods were ordered and received earlier. Pay the supplier the full amount. The journal entry is handled separately.",
        "receive_only": f"The order was placed earlier. Record the goods receipt now that all {po['qty']} units have arrived.",
        "order_only": "Place the order with the supplier. Payment and receipt are handled later.",
    }[wf]


def _composition_tasks() -> list:
    from actionproof.bench import generator_proc as G
    from actionproof.stage4.shift import apply_fault
    rng, out = random.Random(SEED), []
    for wf, steps in AUG_WF.items():
        for i in range(40):
            t = G.make_task("p4_comp_train", len(out), rng)
            state, po, sup, item = G._world(rng, WORLD[wf])
            t.initial_state, t.workflow = state, wf
            t.task_id, t.split = f"p4_comp_train-{len(out):04d}", "p4_comp_train"
            t.meta.update(po_id=po["po_id"], requested=list(steps), faults=[], never_commit=None)
            t.prompt = _prompt(wf, po, sup, item)
            kinds = [k for k, _ in FAULT_MIX]
            kind = rng.choices(kinds, weights=[w for _, w in FAULT_MIX])[0]
            if kind == "partial" and not any(s in ("pay", "receive") for s in steps):
                kind = "phantom"
            if kind == "dup_delivery" and "pay" not in steps:
                kind = "lost_ack"
            apply_fault(t, rng, kind)
            t.meta["fault_class"] = f"p4comp:{wf}:{kind}"
            out.append(t)
    return out


def _check_disjoint(tasks: list) -> None:
    from actionproof.bench.generator_proc import DATA
    from actionproof.stage3.tasks import seen_ids
    tids, pos = seen_ids()
    assert not {t.task_id for t in tasks} & tids, "task ids collide"
    new_pos = [p["po_id"] for t in tasks for p in t.initial_state["pos"]]
    assert len(new_pos) == len(set(new_pos)) and not set(new_pos) & pos, "PO ids collide"
    prompts = set()
    for f in DATA.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                prompts.add(json.loads(line).get("prompt"))
    clash = [t.task_id for t in tasks if t.prompt in prompts]
    assert not clash, f"prompt text collides with an existing split: {clash[:3]}"


def _surface_episode(task, k: int):
    """The P1 teacher episode rendered on surface k (same oracle decisions, read on the canonical probe)."""
    from actionproof.bench.prompts import TOOL_RESULT_FORMAT
    from actionproof.integrity.sim import agents
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.stage3 import scripted
    from actionproof.stage3.env import make_env_s3
    from actionproof.stage3.rollout import act_text, system_prompt
    from actionproof.stage4.surface import make_env_surface
    probe = make_env_s3(task, f"p4demo-probe-{task.task_id}")
    for tool, args in task.meta.get("start_prefix") or []:
        probe.call(tool, dict(args))
    n0 = len(probe.trace)
    (scripted.oracle if task.meta.get("start_prefix") else agents.oracle)(probe, task)
    acts = [(s["tool"], s["args"]) for s in probe.trace[n0:]]
    env = make_env_surface(task, f"p4demo-{task.task_id}", None, k)
    conv = [{"role": "system", "content": system_prompt(env, PROC_NEUTRAL)}, {"role": "user", "content": task.prompt}]
    for tool, args in task.meta.get("start_prefix") or []:
        t, a = env.to_surface(tool, dict(args))
        obs = env.call(t, dict(a))
        conv.append({"role": "assistant", "content": act_text(t, a)})
        conv.append({"role": "user", "content": TOOL_RESULT_FORMAT.format(name=t, content=json.dumps(obs))})
    first = len(conv)
    for tool, args in acts:
        t, a = env.to_surface(tool, dict(args))
        conv.append({"role": "assistant", "content": act_text(t, a)})
        obs = env.call(t, dict(a))
        conv.append({"role": "user", "content": TOOL_RESULT_FORMAT.format(name=t, content=json.dumps(obs))})
        if env.done:
            break
    if not env.done or len(env.trace) - n0 != len(acts):
        return None
    return conv, episode_record_proc(task, env), first


def _correct(task, rec) -> bool:
    s, status = rec["score"], (rec.get("report") or {}).get("status")
    return bool(s["goal_met"] or (task.meta.get("never_commit") and status == "failed") or
                (str(task.meta.get("fault_class", "")).startswith("s3:never_commit") and status == "failed"))


def _rows(task, conv, first, surface) -> list[dict]:
    return [{"prompt": conv[:j], "completion": [conv[j]], "task_id": task.task_id, "fault_class": task.meta.get("fault_class"),
             "workflow": task.workflow, "surface": surface} for j in range(first, len(conv)) if conv[j]["role"] == "assistant"]


def main() -> None:
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.p1 import build as P1
    from actionproof.stage4.surface import choose
    if OUT.exists():
        sys.exit(f"{OUT} exists; P4 data are built once")
    stats, rows = Counter(), []
    p1_ids = {json.loads(l)["task_id"] for l in (ROOT / "data/rft/p1_recovery_rows.jsonl").read_text().splitlines()}
    # 1. P1 recovery episodes: the same kept tasks as Dw; half re-rendered on a surface
    for split in ("p1_train_full", "p1_train_pp", "p1_train_rec"):
        for t in load_tasks(DATA / f"{split}.jsonl"):
            if t.task_id not in p1_ids:
                continue
            k = choose(t.task_id, None)
            ep = P1._episode(t) if k is None else _surface_episode(t, k)
            if ep is None or not _correct(t, ep[1]) or not P1._keys_ok(ep[1]):
                stats["recovery_dropped"] += 1
                continue
            rows += _rows(t, ep[0], ep[2], k)
            stats["recovery_surface" if k is not None else "recovery_canonical"] += 1
    # 2. new compositions
    comp = _composition_tasks()
    _check_disjoint(comp)
    for t in comp:
        k = choose(t.task_id, None)
        ep = P1._episode(t) if k is None else _surface_episode(t, k)
        if ep is None or not _correct(t, ep[1]) or not P1._keys_ok(ep[1]):
            stats["composition_dropped"] += 1
            continue
        rows += _rows(t, ep[0], ep[2], k)
        stats[f"composition_kept:{t.workflow}"] += 1
    # 3. the original warm-start demonstrations, all kept (as in Dw)
    demo = [json.loads(l) for l in (ROOT / "data/rft/proc_demo_rows.jsonl").read_text().splitlines()]
    rows = demo + rows
    stats.update(rows=len(rows), demo_rows=len(demo), dw_rows=DW_ROWS, extra_rows_vs_dw=len(rows) - DW_ROWS)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    OUT.with_suffix(".stats.json").write_text(json.dumps(dict(stats), indent=2, sort_keys=True))
    (ROOT / "data/bench/proc/p4_comp_train.jsonl").write_text("".join(json.dumps(t.to_json()) + "\n" for t in comp))
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
