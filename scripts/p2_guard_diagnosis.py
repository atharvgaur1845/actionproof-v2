"""Why the guard lowers VSC (descriptive; written after the P2 results, changes nothing registered).

    python scripts/p2_guard_diagnosis.py      # writes results/quality/p2_guard_diagnosis.json

**Part A, the guarded episodes.** For each guarded arm and its unguarded counterpart, on the P2 frozen test:
- VSC; the share of episodes with at least one block, and blocks per episode by rule;
- how episodes end (report, max_turns, context, no_call);
- VSC with and without a block.

**Part B, the guarded oracle.** The scripted oracle (P2 reference O) replayed through the guard on the same test
tasks. Each oracle call is checked against the conversation the guard would see; the call is then executed anyway,
so the oracle's correct trajectory is kept. A would-block on a correct trajectory is a rule that blocks behaviour the
registered business semantics require.
"""
from __future__ import annotations

import glob
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
PAIRS = {"Ag": ("A", "results/p1/test"), "Cg": ("C", "results/p1/test"), "Dg": ("D", "results/p1/test"),
         "Eg": ("E", "results/p2/test")}


def rows(d: str, arm: str):
    for f in sorted(glob.glob(str(ROOT / d / f"{arm}_s[0-9]__*.jsonl"))):
        if Path(f).stem.split("__")[1] in SPLITS:
            yield from map(json.loads, Path(f).read_text().splitlines())


def part_a() -> dict:
    out = {}
    for g, (base, bdir) in PAIRS.items():
        for arm, d in ((g, "results/p2/test"), (base, bdir)):
            for mode in ("sampled", "greedy"):
                n, v, blk, vb, nb, vnb = 0, 0, 0, 0, 0, 0
                rules, ended, ended_blk, first_rule = Counter(), Counter(), Counter(), Counter()
                for r in rows(d, arm):
                    if (r["sample"] < 0) != (mode == "greedy"):
                        continue
                    n += 1
                    v += bool(r["vsc"])
                    ended[r["ended_by"]] += 1
                    b = r.get("guard_blocks") or []
                    for x in b:
                        rules[x["rule"]] += 1
                    if b:
                        blk += 1
                        vb += bool(r["vsc"])
                        ended_blk[r["ended_by"]] += 1
                        first_rule[b[0]["rule"]] += 1
                    else:
                        nb += 1
                        vnb += bool(r["vsc"])
                if n:
                    out[f"{arm}|{mode}"] = {"episodes": n, "vsc": v / n, "share_blocked": blk / n,
                                            "blocks_per_episode_by_rule": {k: c / n for k, c in rules.items()},
                                            "first_block_rule": dict(first_rule), "ended_by": dict(ended),
                                            "ended_by_if_blocked": dict(ended_blk),
                                            "vsc_if_blocked": vb / blk if blk else None,
                                            "vsc_if_not_blocked": vnb / nb if nb else None}
    return out


def part_b() -> dict:
    spec = importlib.util.spec_from_file_location("p2_eval", ROOT / "scripts/p2_eval.py")
    pe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pe)
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.bench.prompts import TOOL_RESULT_FORMAT
    from actionproof.integrity.sim import agents
    from actionproof.p2 import guard as G
    from actionproof.stage3 import scripted
    from actionproof.stage3.env import make_env_s3
    res = {}
    for sp in SPLITS:
        n, would, vsc, rules, examples = 0, 0, 0, Counter(), []
        by_tool = defaultdict(Counter)
        for t in load_tasks(DATA / f"{sp}.jsonl"):
            env = make_env_s3(t, f"p2-guard-oracle-{t.task_id}")
            conv = pe.start(env, t, False)
            cfg = G.config(t)
            orig = env.call
            blocks = []

            def call(name, args, _orig=orig, _conv=conv, _cfg=cfg, _blocks=blocks, _t=t):
                _conv.append({"role": "assistant", "content": json.dumps([{"tool_name": name, "parameters": args}])})
                b = G.check(name, dict(args or {}), _conv, _cfg)
                if b:
                    _blocks.append({"tool": name, "rule": b["error"], "step": len(_conv)})
                obs = _orig(name, args)
                _conv.append({"role": "user", "content": TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs))})
                return obs

            env.call = call
            (scripted.oracle if t.meta.get("start_prefix") else agents.oracle)(env, t)
            r = pe.P1.E1.S3E.episode_record_proc(t, env)
            n += 1
            vsc += bool(r["vsc"])
            if blocks:
                would += 1
                for b in blocks:
                    rules[b["rule"]] += 1
                    by_tool[b["rule"]][b["tool"]] += 1
                if len(examples) < 6:
                    examples.append({"task_id": t.task_id, "workflow": t.workflow, "blocks": blocks,
                                     "faults": [f"{f['kind']}:{f['tool']}" for f in t.meta["faults"]]})
        res[sp] = {"tasks": n, "oracle_vsc": vsc / n, "tasks_with_a_would_block": would,
                   "would_blocks_by_rule": dict(rules), "by_rule_tool": {k: dict(v) for k, v in by_tool.items()},
                   "examples": examples}
    return res


def main() -> None:
    out = {"part_a_guarded_episodes": part_a(), "part_b_guarded_oracle": part_b()}
    p = ROOT / "results/quality/p2_guard_diagnosis.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    for k, v in out["part_a_guarded_episodes"].items():
        if k.endswith("sampled"):
            print(k, round(v["vsc"], 3), "blocked", round(v["share_blocked"], 3), v["first_block_rule"], v["ended_by_if_blocked"])
    for k, v in out["part_b_guarded_oracle"].items():
        print(k, v["tasks"], round(v["oracle_vsc"], 3), "would-block tasks", v["tasks_with_a_would_block"], v["would_blocks_by_rule"])


if __name__ == "__main__":
    main()
