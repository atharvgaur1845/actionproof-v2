"""Write the AD-C1 evaluation tasks (PREREGISTRATION_AD_C1.md §2, §4), run after every C1 training task.

Task A: cw_conf, once per model (greedy + 4 samples), for the first half of the models; then the shift splits
        (s4_api, s4_lang_hi, s4_wf4) for seed 0 of every arm through scripts/s4_eval.py.
Task B: cw_conf for the second half; the context baselines (warm start, A12 R_V s0, RSC s0) on cw_conf; then the R1
        retention generations (actionproof/retention/gen.py via slurm/r1_evalset.sh) for seed 0 of every arm.

    python scripts/ad/c1_eval_tasks.py --arms grpo nostd ... --after-a <task> --after-b <task> --ids 1240 1241
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAD = """# minutes: {minutes}
# after: {after}
# AD-C1 evaluation (PREREGISTRATION_AD_C1.md): {what}.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/ad/freeze_c1.py --check || exit 1
mkdir -p results/ad/conf results/ad/shift logs
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \\
  CohereLabs/tiny-aya-global > logs/ad_merge.log 2>&1 || exit 1
conf() {{  # tag adapter-or-empty
  s=results/ad/conf/$1__cw_conf.summary.json
  [ -f $s ] || .venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged ${{2:+--adapter $2}} --tag $1 \\
    --split cw_conf --samples 4 --cost 0.05 --out-dir results/ad/conf > logs/$1_conf.log 2>&1 || {{ tail -20 logs/$1_conf.log; exit 1; }}
  grep -v '^INFO\\|^WARNING' logs/$1_conf.log | head -1
}}
shift_eval() {{  # tag adapter split
  s=results/ad/shift/$1__$3.summary.json
  [ -f $s ] || .venv/bin/python scripts/s4_eval.py --model runs/rft_proc/merged --adapter $2 --tag $1 --split $3 \\
    --out-dir results/ad/shift > logs/$1_$3.log 2>&1 || {{ tail -20 logs/$1_$3.log; exit 1; }}
  tail -1 logs/$1_$3.log
}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--after-a", required=True)
    ap.add_argument("--after-b", required=True)
    ap.add_argument("--ids", nargs=2, type=int, required=True)
    a = ap.parse_args()
    models = [f"ad_{e}_s{s}" for e in a.arms for s in range(5)]
    half = (len(models) + 1) // 2
    A, B = models[:half], models[half:]
    body_a = "".join(f"conf {m} runs/{m}/final\n" for m in A)
    body_a += "".join(f"shift_eval ad_{e}_s0 runs/ad_{e}_s0/final {sp}\n" for e in a.arms
                      for sp in ("s4_api", "s4_lang_hi", "s4_wf4"))
    body_b = "".join(f"conf {m} runs/{m}/final\n" for m in B)
    body_b += "conf ad_base_warm \"\"\nconf ad_base_a12rv_s0 runs/a12_proc_rv_s0\nconf ad_base_rsc_s0 runs/s3_rv_mix_s0/final\n"
    body_b += 'source slurm/r1_evalset.sh "' + " ".join(f"ad_{e}_s0=runs/ad_{e}_s0" for e in a.arms) + '"\n'
    for tid, after, body, what in ((a.ids[0], a.after_a, body_a, "cw_conf part A + shifts"),
                                   (a.ids[1], a.after_b, body_b, "cw_conf part B + context baselines + R1 retention")):
        minutes = 30 + 12 * body.count("\n")
        p = ROOT / "slurm/tasks" / f"{tid}_ad_c1_eval_{'ab'[tid - a.ids[0]]}.sh"
        p.write_text(HEAD.format(minutes=minutes, after=after, what=what) + body + "echo C1_EVAL_OK\n")
        print(p.relative_to(ROOT), "models:", body.count("conf "), "after", after)


if __name__ == "__main__":
    main()
