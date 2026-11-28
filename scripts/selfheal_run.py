"""Self-healing loop runner (docs/selfheal/DESIGN.md). CPU steps only; the agent evaluation itself is
scripts/p1_eval.py / scripts/txn_eval.py on a GPU.

    # 1. DIAGNOSE (descriptive): failure taxonomy of recorded episodes, grouped by agent (seed suffix _sN pooled)
    python scripts/selfheal_run.py diagnose --inputs R/txn/D_s0__txn_test.jsonl ... --out results/selfheal/diag_txn_test
    # 2-3. CORRECT: cut, B4 teacher, replay, verify (dev or loop-pool splits only; test splits are refused)
    python scripts/selfheal_run.py correct --inputs R/p1/dev/D_s0__p1_dev_full.jsonl ... --teacher b4 \
        --out results/selfheal/r1/corrections.jsonl --ledger results/selfheal/ledger.json --round r1
    # 4. CURATE: CuratorKIT dedup, class balance, hold-out by task id -> SFT rows, DPO pairs
    python scripts/selfheal_run.py curate --corrections results/selfheal/r1/corrections.jsonl --out-dir data/selfheal/r1 \
        --ledger results/selfheal/ledger.json --round r1
    # 6. GATE: candidate vs incumbent on fresh splits under frozen criteria -> promote/reject file
    python scripts/selfheal_run.py gate --criteria configs/selfheal/gate_v1.json --candidate ... --incumbent ... \
        --ledger results/selfheal/ledger.json --out results/selfheal/r1/gate_decision.json
    # fresh splits for the gate and loop pools (built once; --check verifies)
    python scripts/selfheal_run.py splits [--check]

The LLM teacher: `correct --teacher llm --api-base http://host:8000/v1 --api-model CohereLabs/aya-expanse-32b`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def tag_of(path: str | Path) -> str:
    """results/<...>/<tag>__<split>.jsonl -> tag."""
    return Path(path).name.split("__")[0]


def group_of(tag: str) -> str:
    """D_s0 -> D; Dwaug_s2 -> Dwaug; dev_D_s0_none -> dev_D_none (seeds pooled)."""
    return re.sub(r"_s\d+", "", tag)


# ---------------------------------------------------------------------- diagnose
def cmd_diagnose(a: argparse.Namespace) -> None:
    from actionproof.selfheal import taxonomy
    from actionproof.selfheal.envs import spec_for, task_from_record
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    per_file = {}
    for path in a.inputs:
        tag = tag_of(path)
        rows = []
        for rec in read_jsonl(path):
            task = task_from_record(rec)
            spec = spec_for(task)
            rec = spec.annotate(rec)
            ok = spec.success(rec)
            c = taxonomy.classify(rec, ok)
            m = rec["task"]["meta"]
            c.update(tag=tag, split=rec["task"]["split"], env=spec.name, fault_class=m.get("fault_class"),
                     workflow=rec["task"]["workflow"], group=(m.get("txn") or {}).get("group"),
                     regime=(m.get("txn") or {}).get("regime"), vtc=bool(rec["score"]["vsc"]),
                     greedy=(-1 if rec.get("sample") is None else rec["sample"]) < 0)
            rows.append(c)
        per_file[str(path)] = {"tag": tag, **taxonomy.tally(rows),
                               "vtc": sum(r["vtc"] for r in rows), "success": sum(r["success"] for r in rows)}
        rows_by_group[group_of(tag)] += rows
    summary: dict[str, Any] = {"label": "DESCRIPTIVE: failure taxonomy of existing recorded episodes; no hypothesis "
                                        "is tested and nothing here was used to choose any loop setting.",
                               "definitions": "actionproof/selfheal/taxonomy.py", "inputs": per_file, "groups": {}}
    for g, rows in sorted(rows_by_group.items()):
        fails = [r for r in rows if not r["success"]]
        by_fault: dict[str, Counter] = defaultdict(Counter)
        for r in fails:
            by_fault[str(r["fault_class"])][r["primary"]] += 1
        by_group: dict[str, Counter] = defaultdict(Counter)
        for r in fails:
            by_group[str(r["group"])][r["primary"]] += 1
        summary["groups"][g] = {
            **taxonomy.tally(rows),
            "vtc": {"num": sum(r["vtc"] for r in rows), "den": len(rows)},
            "success": {"num": sum(r["success"] for r in rows), "den": len(rows)},
            "greedy": taxonomy.tally([r for r in rows if r["greedy"]]),
            "sampled": taxonomy.tally([r for r in rows if not r["greedy"]]),
            "primary_by_fault_class": {k: dict(v) for k, v in sorted(by_fault.items())},
            "primary_by_task_group": {k: dict(v) for k, v in sorted(by_group.items())},
            "tasks_with_any_failure": len({r["task_id"] for r in fails}),
        }
        with (out / f"episodes__{g}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (out / "taxonomy.json").write_text(json.dumps(summary, indent=1))
    (out / "taxonomy.md").write_text(markdown(summary))
    print(markdown(summary))


def markdown(s: dict[str, Any]) -> str:
    from actionproof.selfheal.taxonomy import CLASSES
    groups = s["groups"]
    lines = [f"# Failure taxonomy ({s['label']})", "", "Definitions: `actionproof/selfheal/taxonomy.py`. Counts are "
             "episodes (greedy and sampled pooled; seeds pooled per agent). Success is the verified-correct outcome "
             "(P1: VSC, or an honest failed report on never-commit; TXN: STR).", "",
             "| | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    lines.append("| episodes | " + " | ".join(str(g["episodes"]) for g in groups.values()) + " |")
    lines.append("| verified completion (VSC/VTC) | " + " | ".join(
        f"{g['vtc']['num']}/{g['vtc']['den']} ({g['vtc']['num'] / g['vtc']['den']:.3f})" for g in groups.values()) + " |")
    lines.append("| failures (not success) | " + " | ".join(str(g["failures"]) for g in groups.values()) + " |")
    for c in CLASSES:
        if any(g["primary"].get(c) for g in groups.values()):
            lines.append(f"| {c} | " + " | ".join(str(g["primary"].get(c, 0)) for g in groups.values()) + " |")
    lines += ["", "Flags among failures (multi-label):", "", "| | " + " | ".join(groups) + " |",
              "|---|" + "---|" * len(groups)]
    for f in next(iter(groups.values()))["flags_among_failures"]:
        lines.append(f"| {f} | " + " | ".join(str(g["flags_among_failures"][f]) for g in groups.values()) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- correct
def make_teacher(a: argparse.Namespace):
    from actionproof.selfheal.teachers import B4Teacher, LLMTeacher, http_client
    if a.teacher == "b4":
        return B4Teacher()
    if a.teacher == "llm":
        import os
        if not a.api_base or not a.api_model:
            raise SystemExit("--teacher llm needs --api-base and --api-model")
        return LLMTeacher(http_client(a.api_base, a.api_model, os.environ.get("SELFHEAL_API_KEY"),
                                      temperature=a.temperature))
    raise SystemExit(a.teacher)


def cmd_correct(a: argparse.Namespace) -> None:
    from actionproof.selfheal import correct
    from actionproof.selfheal.ledger import Ledger

    def records():
        for path in a.inputs:
            for rec in read_jsonl(path):
                rec.setdefault("tag", tag_of(path))
                yield rec
    led = Ledger.load(a.ledger) if a.ledger else None
    exclude = led.gate_task_ids() if led else set()  # gate tasks never become correction data
    out = Path(a.out)
    t0 = time.time()
    stats = correct.run(records(), make_teacher(a), out, max_cuts=a.max_cuts, exclude_task_ids=exclude)
    used = sorted({json.loads(l)["task_id"] for l in out.read_text().splitlines() if l.strip()})
    stats.update(inputs=a.inputs, teacher=a.teacher, seconds=round(time.time() - t0, 1), out=str(out),
                 failed_tasks=len(used))
    if led:
        led.use(a.round, "correct", used, sources=a.inputs)
        led.save()
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------------- curate
def cmd_curate(a: argparse.Namespace) -> None:
    from actionproof.selfheal import curate
    from actionproof.selfheal.ledger import Ledger
    rows = [r for p in a.corrections for r in read_jsonl(p)]
    cfg = curate.CurateConfig(holdout_frac=a.holdout_frac, seed=a.seed, per_class_cap=a.per_class_cap,
                              per_task_cap=a.per_task_cap, minhash_threshold=a.minhash,
                              use_curatorkit=not a.no_curatorkit)
    stats = curate.build(rows, Path(a.out_dir), cfg)
    if a.ledger:
        led = Ledger.load(a.ledger)
        led.use(a.round, "train", stats["train_task_ids"], sources=a.corrections)
        led.use(a.round, "holdout", stats["holdout_task_ids"], sources=a.corrections)
        led.save()
    print(json.dumps({k: v for k, v in stats.items() if not k.endswith("_task_ids")}, indent=1))


def cmd_mix(a: argparse.Namespace) -> None:
    """Correction SFT rows plus a seeded sample of the incumbent's own warm-start rows (a replay buffer against
    forgetting): n_replay = round(ratio x n_corrections), drawn without replacement."""
    import random
    rows = [l for l in Path(a.rows).read_text().splitlines() if l.strip()]
    pool = [l for l in Path(a.replay).read_text().splitlines() if l.strip()]
    k = min(len(pool), round(a.ratio * len(rows)))
    rng = random.Random(a.seed)
    picked = rng.sample(pool, k)
    mixed = rows + picked
    rng.shuffle(mixed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(mixed) + "\n")
    st = {"corrections_rows": len(rows), "replay_rows": k, "replay_source": a.replay, "ratio": a.ratio, "seed": a.seed}
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st))


# ---------------------------------------------------------------------- gate
def cmd_gate(a: argparse.Namespace) -> None:
    from actionproof.selfheal import gate
    from actionproof.selfheal.ledger import Ledger
    led = Ledger.load(a.ledger) if a.ledger else None
    d = gate.decide(Path(a.criteria), a.candidate, a.incumbent, led)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=1))
    print(json.dumps({"decision": d["decision"], "criteria": {k: v["pass"] for k, v in d["criteria"].items()}},
                     indent=1))


# ---------------------------------------------------------------------- splits
def cmd_splits(a: argparse.Namespace) -> None:
    from actionproof.selfheal import splits
    if a.check:
        sys.exit(splits.check())
    print(json.dumps(splits.write(), indent=1))


def cmd_register_gate(a: argparse.Namespace) -> None:
    """Record the gate splits in the ledger, so `correct` skips their tasks and `use` refuses them as loop data."""
    from actionproof.bench.generator_proc import DATA
    from actionproof.selfheal import splits
    from actionproof.selfheal.ledger import Ledger
    led = Ledger.load(a.ledger)
    for n in splits.GATE_PRIMARY + splits.GATE_RETENTION:
        led.register_gate_split(n, DATA / f"{n}.jsonl")
    led.save()
    print(json.dumps({n: len(g["task_ids"]) for n, g in led.gate_splits.items()}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--inputs", nargs="+", required=True)
    d.add_argument("--out", required=True)
    c = sub.add_parser("correct")
    c.add_argument("--inputs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--teacher", choices=("b4", "llm"), default="b4")
    c.add_argument("--api-base")
    c.add_argument("--api-model")
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--max-cuts", type=int, default=None)
    c.add_argument("--ledger")
    c.add_argument("--round", default="r1")
    u = sub.add_parser("curate")
    u.add_argument("--corrections", nargs="+", required=True)
    u.add_argument("--out-dir", required=True)
    u.add_argument("--holdout-frac", type=float, default=0.2)
    u.add_argument("--seed", type=int, default=20261129)
    u.add_argument("--per-class-cap", type=int, default=150)
    u.add_argument("--per-task-cap", type=int, default=2)
    u.add_argument("--minhash", type=float, default=0.95)
    u.add_argument("--no-curatorkit", action="store_true")
    u.add_argument("--ledger")
    u.add_argument("--round", default="r1")
    x = sub.add_parser("mix")
    x.add_argument("--rows", required=True)
    x.add_argument("--replay", required=True)
    x.add_argument("--ratio", type=float, default=1.0)
    x.add_argument("--seed", type=int, default=20261129)
    x.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--criteria", required=True)
    g.add_argument("--candidate", nargs="+", required=True)
    g.add_argument("--incumbent", nargs="+", required=True)
    g.add_argument("--ledger")
    g.add_argument("--out", required=True)
    s = sub.add_parser("splits")
    s.add_argument("--check", action="store_true")
    rg = sub.add_parser("register-gate")
    rg.add_argument("--ledger", required=True)
    a = ap.parse_args(argv)
    {"diagnose": cmd_diagnose, "correct": cmd_correct, "curate": cmd_curate, "gate": cmd_gate, "mix": cmd_mix, "register-gate": cmd_register_gate,
     "splits": cmd_splits}[a.cmd](a)


if __name__ == "__main__":
    main()
