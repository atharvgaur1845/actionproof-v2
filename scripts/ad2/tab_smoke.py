"""Smoke run of tabular estimators on the core two-op workflow (risky op + safe op). Not an experiment.

    python scripts/ad2/tab_smoke.py <p> <s> <c> <est,est,...> [steps] [cond] [rho0]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from actionproof.ad2.tabular.env import Config, OpType  # noqa: E402
from actionproof.ad2.tabular.train import train  # noqa: E402

p, s, c = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200
cond = sys.argv[6] if len(sys.argv) > 6 else "A"
rho0 = float(sys.argv[7]) if len(sys.argv) > 7 else 0.3
cfg = Config([OpType("pay", p, s, tid=0), OpType("post", 0.0, 1.0, tid=1)], c=c, tag="core")
for est in sys.argv[4].split(","):
    t = time.time()
    r = train({"tasks": [cfg], "est": est, "steps": steps, "seed": 0, "rho0": rho0, "n_mc": 500, "cond": cond})
    e = r["eval"]["core"]
    print(f"{est:12s} J={e['J']:.3f} J*={e['Jstar']:.3f} vprob={[round(x, 3) for x in e['verify_prob']]} "
          f"voi={[round(x, 3) for x in e['voi']]} zero={r['curve'][-1]['zero_std_frac']:.2f} "
          f"eps={r['episodes_used']:.0f} {time.time() - t:.1f}s", flush=True)
