"""Run a registered testbed experiment family (docs/research/algorithm_discovery_v2/V2TB_PROTOCOL.md).

    python scripts/ad2/tab_sweep.py TB1 [--workers 14] [--limit N]

Writes results/ad2/tabular/<FAM>.jsonl, one line per run, appending; runs already on file (same run id) are skipped,
so an interrupted family resumes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.ad2.tabular import envs as E  # noqa: E402

OUT = ROOT / "results/ad2/tabular"

TB1_EST = ["grpo", "grpo_B", "nostd", "nostd_B", "rloo", "batchstd", "gdpo", "awrb", "awrbn", "critic_v", "critic_q",
           "igc", "lagr_grpo", "ema", "ctxema", "fork_act", "fork_bel", "cvc"]
BASE = ["grpo", "nostd", "rloo", "batchstd", "gdpo", "critic_v", "critic_q", "ema", "ctxema", "lagr", "awrb", "awrbn",
        "igc"]
FORK = ["fork_act", "fork_bel", "fork_bel_std", "cvc"]


def _est(name: str) -> tuple[str, str]:
    return (name[:-2], "B") if name.endswith("_B") else (name, "A")


def family(fam: str) -> list[dict]:
    runs = []
    if fam == "TB1":
        for (p, c, s, L) in E.tb1_grid():
            for name in TB1_EST:
                for seed in range(3):
                    est, cond = _est(name)
                    runs.append({"id": f"TB1|{p}|{c}|{s}|{L}|{name}|{seed}", "env": ["core", p, c, s, L], "est": est,
                                 "cond": cond, "seed": seed, "steps": 200, "fork_k": 1, "n_mc": 300,
                                 "budget": 0.5 * c})
    elif fam == "TB2":
        for mode in ("single", "multi"):
            for c in (0.02, 0.05, 0.1):
                for L in (0.0, 1.0):
                    for seed in range(5):
                        for name in BASE:
                            for mult in (1, 4, 8):
                                if mult > 1 and name in ("grpo", "batchstd", "gdpo", "awrb", "lagr"):
                                    continue
                                runs.append({"id": f"TB2|{mode}|{c}|{L}|{name}|x{mult}|{seed}", "env": ["rich", mode, c, L],
                                             "est": name, "cond": "A", "seed": seed, "steps": 200 * mult, "n_mc": 1000,
                                             "budget": 0.05})
                        for name in FORK:
                            for k in (1, 4):
                                runs.append({"id": f"TB2|{mode}|{c}|{L}|{name}|k{k}|{seed}", "env": ["rich", mode, c, L],
                                             "est": name, "cond": "A", "seed": seed, "steps": 200, "fork_k": k,
                                             "n_mc": 1000})
    elif fam == "TB3":
        for name in ["grpo", "nostd", "rloo", "critic_q", "igc", "ctxema", "awrbn", "fork_act", "fork_bel", "cvc"]:
            for seed in range(5):
                for mult in ((1, 4) if name in ("nostd", "rloo", "critic_q", "igc", "ctxema") else (1,)):
                    runs.append({"id": f"TB3|{name}|x{mult}|{seed}", "env": ["gen"], "est": name, "cond": "A",
                                 "seed": seed, "steps": 400 * mult, "fork_k": 1, "n_mc": 200, "mode": "feat"})
    elif fam == "TB4":
        for name in ["awrb", "awrbn", "fork_bel", "cvc", "igc"]:
            for prior in ("true", "est", "mis:0.25", "mis:4", "flat:0.3"):
                for seed in range(5):
                    runs.append({"id": f"TB4|{name}|{prior}|{seed}", "env": ["rich", "multi", 0.05, 1.0], "est": name,
                                 "cond": "A", "seed": seed, "steps": 200, "fork_k": 1, "n_mc": 1000, "prior": prior})
        for name in ["nostd", "fork_act"]:
            for seed in range(5):
                runs.append({"id": f"TB4|{name}|none|{seed}", "env": ["rich", "multi", 0.05, 1.0], "est": name,
                             "cond": "A", "seed": seed, "steps": 200, "fork_k": 1, "n_mc": 1000})
    elif fam == "TB6":
        for p in (0.2, 0.4, 0.6):
            for c in (0.01, 0.05, 0.2):
                for est in ("grpo", "nostd"):
                    for rho0 in (0.05, 0.3, 0.6):
                        for G in (4, 8, 16):
                            for opt in ("adam", "sgd"):
                                for cond in ("A", "B"):
                                    for seed in range(3):
                                        runs.append({"id": f"TB6|{p}|{c}|{est}|{rho0}|{G}|{opt}|{cond}|{seed}",
                                                     "env": ["core", p, c, 1.0, 0.0], "est": est, "cond": cond,
                                                     "seed": seed, "steps": 200, "rho0": rho0, "G": G, "opt": opt,
                                                     "lr": 0.05 if opt == "adam" else 2.0, "n_mc": 100})
    else:
        raise SystemExit(f"unknown family {fam}")
    return runs


def build(env: list) -> tuple[list, list]:
    if env[0] == "core":
        t = [E.core(*env[1:])]
        return t, t
    if env[0] == "rich":
        t = E.rich(env[1], env[2], env[3])
        return t, t
    if env[0] == "gen":
        tr = E.gen_tasks("train")
        return tr, tr[:6] + E.gen_tasks("test_iid") + E.gen_tasks("test_shift") + E.gen_tasks("test_long")
    raise ValueError(env)


def run_one(r: dict) -> dict:
    from actionproof.ad2.tabular.train import train
    tasks, ev = build(r["env"])
    spec = {k: v for k, v in r.items() if k not in ("id", "env")}
    spec.update(tasks=tasks, eval_tasks=ev)
    t = time.time()
    out = train(spec)
    out.pop("theta", None) if r["env"][0] == "core" else None
    return {"id": r["id"], "env": r["env"], **out, "seconds": time.time() - t}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry", type=int, default=0, help="plumbing check: run N spread-out runs, write nothing")
    a = ap.parse_args()
    if a.dry:
        fam = family(a.family)
        step = max(1, len(fam) // a.dry)
        for r in fam[::step][: a.dry]:
            res = run_one(r)
            e = next(iter(res["eval"].values()))
            print(r["id"], f"{res['seconds']:.1f}s eps={res['episodes_used']:.0f} J={e['J']:.3f} regret={e['regret']:.3f}",
                  flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.family}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}
    todo = [r for r in family(a.family) if r["id"] not in done][: a.limit]
    print(f"{a.family}: {len(todo)} runs to do ({len(done)} on file)", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, path.open("a") as f:
        for i, res in enumerate(pool.imap_unordered(run_one, todo, chunksize=1)):
            f.write(json.dumps(res, default=float) + "\n")
            f.flush()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"{a.family} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
