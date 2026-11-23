"""P1 practical-agent study (PREREGISTRATION_P1.md): fresh splits and validated recovery demonstrations.

**Splits.** Every task is new: new generator seeds, with task and PO ids checked against every existing split.

| Split | Size | Contents |
|---|---|---|
| p1_train_full | 240 | full episodes, training fault mix (generator default), 80 per workflow |
| p1_train_pp | 90 | full episodes with a partial payment on attempt 0, ratios 0.25/0.5/0.75, 30 per workflow |
| p1_train_rec | 300 | recovery starts (`stage3.tasks.make_pool` mix: phantom, partial, lost ack, duplicate, complete, never-commit, done) |
| p1_dev_full | 90 | as p1_train_full; development only |
| p1_dev_rec | 90 | as p1_train_rec; development only |
| p1_test_full | 120 | frozen test: training fault mix, 40 per workflow |
| p1_test_pp | 90 | frozen test: partial payment, **unseen ratios 0.4 and 0.6**, 30 per workflow |
| p1_test_rec | 120 | frozen test: recovery starts |
| p1_test_wf4 | 60 | frozen test: **unseen workflow** order→receive→pay→post, one fault on pay or receive |
| p1_test_combo | 60 | frozen test: **two faults** in one episode (generator `proc_combo` rule), ids renamed |

**Demonstrations** (arms C and D):
- **Teacher.** The scripted `oracle` (`actionproof/integrity/sim/agents.py`, `stage3/scripted.py`) on p1_train_* only. It decides only from read results and the request; the hidden fault log is never read.
- **Format.** Episodes are replayed through the same conversation format as training and evaluation (`run_episode`; `start_conversation` + `act_text` for recovery starts). Only post-prefix assistant turns become SFT targets.
- **Validation.**
  - An episode is kept only if the verifier confirms the correct outcome: goal met, or an honest "failed" report on never-commit.
  - Every re-pay after a refund must use a new idempotency key (checked).
- **Curation.** CuratorKIT exact and MinHash deduplication over the action sequence plus prompt template, so near-identical episodes do not dominate.

    python -m actionproof.p1.build splits    # once; refuses to overwrite
    python -m actionproof.p1.build demos     # -> data/rft/p1_recovery_rows.jsonl (+ .stats.json)
"""
from __future__ import annotations

import copy
import dataclasses
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WF = ("order_prepay", "pay_post", "receive_pay")
SPLITS = ("p1_train_full", "p1_train_pp", "p1_train_rec", "p1_dev_full", "p1_dev_rec",
          "p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")


def _balanced(name: str, seed: int, per_wf: int, make=None) -> list:
    from actionproof.bench.generator_proc import make_task
    make = make or make_task
    rng, out, i, have = random.Random(seed), [], 0, Counter()
    while min(have[w] for w in WF) < per_wf:
        t = make(name, i, rng)
        i += 1
        if have[t.workflow] < per_wf:
            out.append(t)
            have[t.workflow] += 1
    return out


def _pp(name: str, seed: int, n: int, ratios: tuple[float, ...]) -> list:
    sys.path.insert(0, str(ROOT / "scripts"))
    import s3_make_rebal as RB
    fresh, out = RB.Fresh(name, seed), []
    for j in range(n):
        wf = WF[j % 3]
        out.append(RB.with_fault(fresh.take(wf), {"tool": RB.PAY, "k": 0, "kind": "partial",
                                                   "ratio": ratios[(j // 3) % len(ratios)]}, None, "partial"))
    return out


def _wf4(name: str, seed: int, n: int) -> list:
    from actionproof.integrity.sim.env import STEP_TOOL
    from actionproof.stage4.shift import workflow4_task
    rng, out = random.Random(seed), []
    kinds = ("phantom", "partial", "lost_ack", "precommit_timeout")
    for i in range(n):
        t = workflow4_task(name, i, rng)
        step, kind = rng.choice(("pay", "receive")), kinds[i % len(kinds)]
        f = {"tool": STEP_TOOL[step], "k": 0, "kind": kind}
        if kind == "partial":
            f["ratio"] = rng.choice((0.25, 0.5, 0.75))
        t.meta.update(faults=[f], fault_class=f"wf4:{kind}:{step}")
        out.append(t)
    return out


def _combo(name: str, seed: int, n: int) -> list:
    from actionproof.bench.generator_proc import make_task
    rng, out = random.Random(seed), []
    for i in range(n):
        t = make_task("proc_combo", i, rng)  # the generator's two-fault rule is keyed on this split name
        t = dataclasses.replace(t, task_id=f"{name}-{i:04d}", split=name) if dataclasses.is_dataclass(t) else t
        if not dataclasses.is_dataclass(t):
            t.task_id, t.split = f"{name}-{i:04d}", name
        out.append(t)
    return out


def make_splits() -> None:
    from actionproof.bench.generator_proc import DATA
    from actionproof.stage3.tasks import make_pool, seen_ids, write
    for s in SPLITS:
        if (DATA / f"{s}.jsonl").exists():
            raise SystemExit(f"{s} exists; P1 splits are built once")
    built = {
        "p1_train_full": _balanced("p1_train_full", 1401, 80),
        "p1_train_pp": _pp("p1_train_pp", 1402, 90, (0.25, 0.5, 0.75)),
        "p1_train_rec": make_pool("p1_train_rec", 300, 1403),
        "p1_dev_full": _balanced("p1_dev_full", 1404, 30),
        "p1_dev_rec": make_pool("p1_dev_rec", 90, 1405),
        "p1_test_full": _balanced("p1_test_full", 1406, 40),
        "p1_test_pp": _pp("p1_test_pp", 1407, 90, (0.4, 0.6)),
        "p1_test_rec": make_pool("p1_test_rec", 120, 1408),
        "p1_test_wf4": _wf4("p1_test_wf4", 1409, 60),
        "p1_test_combo": _combo("p1_test_combo", 1410, 60),
    }
    tids, pos = seen_ids()
    allt = [t for ts in built.values() for t in ts]
    new_pos = [p["po_id"] for t in allt for p in t.initial_state["pos"]]
    assert len(set(t.task_id for t in allt)) == len(allt) and not {t.task_id for t in allt} & tids, "task ids collide"
    assert len(new_pos) == len(set(new_pos)) and not set(new_pos) & pos, "PO ids collide"
    for name, ts in built.items():
        write(name, ts)
        print(name, len(ts), dict(Counter(t.workflow for t in ts)), dict(Counter(t.meta["fault_class"] for t in ts).most_common(4)))


def _act(tool: str, args: dict) -> str:
    return json.dumps([{"tool_name": tool, "parameters": args}])


def _episode(task) -> tuple[list[dict], dict, int] | None:
    """Teacher conversation, verifier record, and the index of the first teacher turn (after any scripted prefix)."""
    from actionproof.bench.prompts import TOOL_RESULT_FORMAT
    from actionproof.integrity.sim import agents
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.rollout.episode import _agenttune
    from actionproof.stage3 import scripted
    from actionproof.stage3.env import make_env_s3
    from actionproof.stage3.rollout import start_conversation
    _, _, parse = _agenttune()
    probe = make_env_s3(task, f"p1demo-probe-{task.task_id}")
    for tool, args in task.meta.get("start_prefix") or []:
        probe.call(tool, dict(args))
    n0 = len(probe.trace)
    (scripted.oracle if task.meta.get("start_prefix") else agents.oracle)(probe, task)
    acts = [(s["tool"], s["args"]) for s in probe.trace[n0:]]
    env = make_env_s3(task, f"p1demo-{task.task_id}")
    conv = start_conversation(env, task, PROC_NEUTRAL)
    first = len(conv)
    for tool, args in acts:
        text = _act(tool, args)
        conv.append({"role": "assistant", "content": text})
        res = []
        for c in parse(text) or []:
            obs = env.call(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
            res.append(TOOL_RESULT_FORMAT.format(name=c["function"]["name"], content=json.dumps(obs)))
        conv.append({"role": "user", "content": "\n\n".join(res)})
        if env.done:
            break
    if not env.done or len(env.trace) - n0 != len(acts):
        return None
    return conv, episode_record_proc(task, env), first


def _keys_ok(rec: dict) -> bool:
    """Every pay_supplier after a refund uses an idempotency key not used before."""
    used, refunded = set(), False
    for c in rec["trace"]:
        if c["tool"] == "refund_payment":
            refunded = True
        if c["tool"] == "pay_supplier":
            k = c["args"].get("idempotency_key")
            if refunded and k in used:
                return False
            used.add(k)
    return True


def demos() -> None:
    from actionproof.bench.generator_proc import DATA, load_tasks
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
    from curatorkit.schema import DataSample
    stats = Counter()
    kept = []
    for split in ("p1_train_full", "p1_train_pp", "p1_train_rec"):
        for t in load_tasks(DATA / f"{split}.jsonl"):
            ep = _episode(t)
            if ep is None:
                stats["dropped_replay_mismatch"] += 1
                continue
            conv, rec, first = ep
            s = rec["score"]
            status = (rec.get("report") or {}).get("status")
            correct = s["goal_met"] or (t.meta.get("never_commit") and status == "failed") or \
                (str(t.meta.get("fault_class", "")).startswith("s3:never_commit") and status == "failed")
            if not correct:
                stats["dropped_not_verified"] += 1
                continue
            if not _keys_ok(rec):
                stats["dropped_key_reuse"] += 1
                continue
            sig = " | ".join(m["content"] for m in conv[first:] if m["role"] == "assistant")
            # dedup key: workflow, fault class and the teacher's action sequence with ids masked to their shape
            import re
            masked = re.sub(r"(PO|PAY|RF|GR|JE|SO|ACCT)-[0-9A-F]+", r"\1-X", sig)
            masked = re.sub(r"\d{3,}", "N", masked)
            kept.append((t, conv, first, rec, f"{t.workflow} :: {t.meta.get('fault_class')}", masked))
            stats[f"kept:{split}"] += 1
    # CuratorKIT: exact and near-duplicate removal over the full (unmasked) request + teacher actions
    samples = [DataSample(id=str(i), source_uri=f"p1/{k[0].task_id}", instruction=k[0].prompt,
                          output=" | ".join(m["content"] for m in k[1][k[2]:] if m["role"] == "assistant"),
                          task_type="instruction_following") for i, k in enumerate(kept)]
    after_exact = ExactDeduplicator().run(samples)
    after_mh = MinHashDeduplicator(threshold=0.95).run(after_exact)
    survivors = sorted(int(s.id) for s in after_mh)
    # balance (ours, not CuratorKIT): at most CAP episodes per (workflow, fault class, masked action pattern)
    CAP, per = 12, Counter()
    keep_ids = set()
    for i in survivors:
        key = (kept[i][4], kept[i][5])
        if per[key] < CAP:
            per[key] += 1
            keep_ids.add(i)
    stats["balance_cap_removed"] = len(survivors) - len(keep_ids)
    stats["distinct_action_patterns"] = len(per)
    stats["curatorkit_exact_removed"] = len(samples) - len(after_exact)
    stats["curatorkit_minhash_removed"] = len(after_exact) - len(after_mh)
    out = ROOT / "data/rft/p1_recovery_rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out.open("w") as f:
        for i, (t, conv, first, rec, *_) in enumerate(kept):
            if i not in keep_ids:
                continue
            for j in range(first, len(conv)):
                if conv[j]["role"] == "assistant":
                    f.write(json.dumps({"prompt": conv[:j], "completion": [conv[j]], "task_id": t.task_id,
                                        "fault_class": t.meta.get("fault_class"), "workflow": t.workflow}) + "\n")
                    n_rows += 1
            stats[f"episodes_by_fault:{t.meta.get('fault_class')}"] += 1
    stats["episodes"] = len(keep_ids)
    stats["rows"] = n_rows
    out.with_suffix(".stats.json").write_text(json.dumps(dict(stats), indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("episodes_by_fault")}, indent=1))


if __name__ == "__main__":
    {"splits": make_splits, "demos": demos}[sys.argv[1]]()
