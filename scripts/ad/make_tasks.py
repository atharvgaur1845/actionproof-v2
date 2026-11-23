"""Write the AD-D1 Slurm task files (slurm/tasks/12xx_ad_*.sh). Run once; the files are then hash-checked by name in
docs/research/algorithm_discovery/EXPERIMENT_REGISTRY.csv."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAD = """# minutes: {minutes}
{after}# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): {name}.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT={port}
"""

SMOKE = """.venv/bin/python scripts/ad/freeze.py --check || exit 1
for est in awrb nostd; do
  OUT=runs/ad_smoke_$est; rm -rf $OUT
  AP_ROLLOUT_LOG=logs/ad_smoke_$est.rollouts.jsonl .venv/bin/python -m actionproof.ad.train --estimator $est \\
    --pools data/bench/proc/cw_train.jsonl --seed 0 --steps 2 --init-adapter runs/rft_proc/final --out $OUT \\
    --use-vllm > logs/ad_smoke_$est.log 2>&1 || { tail -40 logs/ad_smoke_$est.log; exit 1; }
  .venv/bin/python - "$est" <<'PY' || exit 1
import json, sys
est = sys.argv[1]
rows = [json.loads(l) for l in open(f"logs/ad_smoke_{est}.rollouts.jsonl")]
bad = [r["task_id"] for r in rows if r["ad"]["r_bar"] is None]
print(est, "episodes", len(rows), "no r_bar", len(bad), "mean reads", sum(r["n_verify"] for r in rows) / len(rows),
      "mean r_v", sum(r["score"]["r_v"] for r in rows) / len(rows), "mean r_bar",
      sum(r["ad"]["r_bar"] for r in rows) / len(rows))
assert not bad and len(rows) >= 64
cfg = json.load(open(f"runs/ad_smoke_{est}/run_config.json"))
assert cfg["scale_rewards"] == {"awrb": "group", "nostd": "none"}[est]
PY
  rm -rf $OUT
done
.venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged --tag ad_smoke_warm --split cw_dev --n-tasks 10 \\
  --out-dir logs/ad_smoke_eval > logs/ad_smoke_eval.log 2>&1 || { tail -30 logs/ad_smoke_eval.log; exit 1; }
tail -1 logs/ad_smoke_eval.log
echo SMOKE_OK
"""

BASELINES = """.venv/bin/python scripts/ad/freeze.py --check || exit 1
mkdir -p results/ad/dev
ev() {  # tag adapter split samples
  s=results/ad/dev/$1__$3.summary.json
  [ -f $s ] || .venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged ${2:+--adapter $2} --tag $1 \\
    --split $3 --n-tasks 150 --samples $4 --out-dir results/ad/dev > logs/ad_eval_$1_$3.log 2>&1 \\
    || { tail -20 logs/ad_eval_$1_$3.log; exit 1; }
  grep -v '^INFO\\|^WARNING' logs/ad_eval_$1_$3.log | tail -5
}
ev ad_base_warm "" cw_dev 4
ev ad_base_a12rv_s0 runs/a12_proc_rv_s0 cw_dev 4
ev ad_base_rsc_s0 runs/s3_rv_mix_s0/final cw_dev 4
"""


def write(tid: int, name: str, minutes: int, after: str | None, body: str) -> None:
    p = ROOT / "slurm/tasks" / f"{tid}_{name}.sh"
    p.write_text(HEAD.format(minutes=minutes, after=f"# after: {after}\n" if after else "", name=name,
                             port=29000 + tid % 1000) + body)
    print(p.relative_to(ROOT))


def main() -> None:
    write(1200, "ad_smoke", 60, None, SMOKE)
    arms = [(1201, "awrb", 0, "1200_ad_smoke"), (1202, "grpo", 0, "1210_ad_baselines"),
            (1203, "nostd", 0, "1201_ad_awrb_s0"), (1204, "awrb", 1, "1202_ad_grpo_s0"),
            (1205, "grpo", 1, "1203_ad_nostd_s0"), (1206, "nostd", 1, "1204_ad_awrb_s1")]
    for tid, est, seed, after in arms:
        write(tid, f"ad_{est}_s{seed}", 300, after, f"source slurm/ad_arm.sh {est} {seed} 200\n")
    write(1210, "ad_baselines", 90, "1200_ad_smoke", BASELINES)


if __name__ == "__main__":
    main()
