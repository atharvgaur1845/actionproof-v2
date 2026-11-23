"""Write the P2 cluster tasks (slurm/tasks/118x-119x_p2_*.sh). Run once before registration; the files are frozen."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
DEV = "p1_dev_full p1_dev_rec"
SHIFT = "s4_lang_hi s4_lang_hinglish s4_api"
HEAD = """set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
"""


def M(arm: str, k: int | None, flags: str = "-") -> str:
    base, ad = {"A": ("runs/rft_proc/merged", f"runs/s3_rv_mix_s{k}/final"),
                "C": (f"runs/s3_rv_mix_s{k}/merged", f"runs/p1_c_s{k}/final"),
                "D": ("runs/p1_dwarm/merged", f"runs/p1_d_s{k}/final"),
                "E": ("runs/p2_ewarm/merged", f"runs/p2_e_s{k}/final"),
                "M": ("runs/rft_proc/merged", f"runs/s3_matched_s{k}/final"),
                "W": ("runs/rft_proc/merged", "-")}[arm]
    return base, ad


def ev(phase: str, tag: str, arm: str, k, flags: str, splits: str) -> str:
    base, ad = M(arm, k)
    return f'source slurm/p2_evalset.sh {phase} "{tag}|{base}|{ad}|{flags}" "{splits}"\n'


def task(name: str, minutes: int, desc: str, body: str, after: str = "") -> None:
    txt = f"# minutes: {minutes}\n# P2 (PREREGISTRATION_P2.md): {desc}\n" + (f"# after: {after}\n" if after else "") + HEAD + body
    (ROOT / f"slurm/tasks/{name}.sh").write_text(txt)


def merge_lines(k: int) -> str:
    return (f"[ -f runs/s3_rv_mix_s{k}/merged/config.json ] || .venv/bin/python -m actionproof.train.merge "
            f"runs/s3_rv_mix_s{k}/final runs/rft_proc/merged > logs/p2_merge_rsc{k}.log 2>&1 || exit 1\n")


def main() -> None:
    task("1180_p2_smoke", 60, "smoke (not analysed): tests, ledger and guard on 8 dev tasks, harness equivalence with P1",
         ".venv/bin/python -m pytest -q tests/p2 tests/p1 || exit 1\n"
         ".venv/bin/python scripts/p2_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag B2_smoke "
         "--split p1_dev_rec --phase smoke --samples 1 --ledger --n-tasks 8 > logs/p2_smoke_b2.log 2>&1 || { tail -30 logs/p2_smoke_b2.log; exit 1; }\n"
         ".venv/bin/python scripts/p2_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag Ag_smoke "
         "--split p1_dev_rec --phase smoke --samples 1 --guard --n-tasks 8 > logs/p2_smoke_ag.log 2>&1 || { tail -30 logs/p2_smoke_ag.log; exit 1; }\n"
         ".venv/bin/python scripts/p2_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag A_equiv "
         "--split p1_dev_rec --phase smoke --samples 4 > logs/p2_smoke_eq.log 2>&1 || { tail -30 logs/p2_smoke_eq.log; exit 1; }\n"
         ".venv/bin/python scripts/p2_equivalence.py > logs/p2_equivalence.log 2>&1; cat logs/p2_equivalence.log\n")
    task("1181_p2_oracle", 30, "scripted reference policy on test and language-shift splits (CPU)",
         ".venv/bin/python scripts/p2_oracle.py\n")
    task("1182_p2_ewarm", 90, "arm E warm start: arm D's rows with the ledger, D's exact SFT settings",
         "[ -f data/rft/p2_ewarm_rows.stats.json ] || .venv/bin/python -m actionproof.p2.build || exit 1\n"
         "[ -f runs/p2_ewarm/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --data data/rft/p2_ewarm_rows.jsonl \\\n"
         "  --out runs/p2_ewarm --epochs 2 --seed 0 > logs/p2_ewarm.full.log 2>&1 || { tail -30 logs/p2_ewarm.full.log; exit 1; }\n"
         "rm -rf runs/p2_ewarm/checkpoint-*/optimizer.pt\n"
         "[ -f runs/p2_ewarm/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/p2_ewarm/final CohereLabs/tiny-aya-global > logs/p2_merge_ewarm.log 2>&1 || exit 1\n")
    for k in (0, 1, 2):
        task(f"118{3 + k}_p2_e_s{k}", 210, f"arm E seed {k}: the RSC recipe unchanged (R_V, s3_trainb_half + s3_rec_train_half, 200 steps) with ledger rollouts",
             f"export MASTER_PORT={29960 + k} AP_ROLLOUT_LOG=results/p2/rollouts/E_s{k}.jsonl\nmkdir -p results/p2/rollouts\n"
             f"[ -f runs/p2_e_s{k}/final/adapter_config.json ] || .venv/bin/python -m actionproof.p2.train --reward rv \\\n"
             f"  --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl --seed {k} --steps 200 \\\n"
             f"  --init-adapter runs/p2_ewarm/final --out runs/p2_e_s{k} --use-vllm > logs/p2_e_s{k}.full.log 2>&1 || {{ tail -40 logs/p2_e_s{k}.full.log; exit 1; }}\n"
             f"gzip -f $AP_ROLLOUT_LOG; rm -rf runs/p2_e_s{k}/checkpoint-*/optimizer.pt\n" + ev("dev", f"E_s{k}", "E", k, "L", DEV),
             after="1182_p2_ewarm")
    task("1186_p2_dev_b2g", 90, "development (descriptive): B2 and Ag, three seeds",
         "".join(merge_lines(k) for k in (0, 1, 2)) +
         "".join(ev("dev", f"B2_s{k}", "A", k, "L", DEV) + ev("dev", f"Ag_s{k}", "A", k, "G", DEV) for k in (0, 1, 2)))
    for n, (tag, arm, fl, after) in enumerate([("B2", "A", "L", ""), ("Ag", "A", "G", ""), ("Cg", "C", "G", ""),
                                               ("Dg", "D", "G", "1154_p1_d_s0 1155_p1_d_s1 1156_p1_d_s2")]):
        task(f"118{7 + n}_p2_test_{tag}" if 7 + n < 10 else f"119{7 + n - 10}_p2_test_{tag}", 120,
             f"frozen test once: {tag}, three seeds",
             "".join(merge_lines(k) for k in (0, 1, 2)) + "".join(ev("test", f"{tag}_s{k}", arm, k, fl, TEST) for k in (0, 1, 2)),
             after=after)
    task("1191_p2_test_E", 150, "frozen test once: E and Eg, three seeds",
         "".join(ev("test", f"E_s{k}", "E", k, "L", TEST) + ev("test", f"Eg_s{k}", "E", k, "LG", TEST) for k in (0, 1, 2)),
         after="1183_p2_e_s0 1184_p2_e_s1 1185_p2_e_s2")
    task("1192_p2_ref", 90, "references on the test union: warm start W, class-matched control M (three seeds)",
         ev("ref", "W", "W", None, "-", TEST).replace('"W|', '"W|') + "".join(ev("ref", f"M_s{k}", "M", k, "-", TEST) for k in (0, 1, 2)))
    for n, (arm, fl, after) in enumerate([("A", "-", ""), ("C", "-", ""), ("D", "-", "1154_p1_d_s0 1155_p1_d_s1 1156_p1_d_s2"),
                                          ("E", "L", "1183_p2_e_s0 1184_p2_e_s1 1185_p2_e_s2")]):
        task(f"119{3 + n}_p2_shift_{arm}", 120, f"shift splits once (Hindi, Hinglish, renamed API): {arm}, three seeds",
             ("".join(merge_lines(k) for k in (0, 1, 2)) if arm == "C" else "") +
             "".join(ev("shift", f"{arm}_s{k}", arm, k, fl, SHIFT) for k in (0, 1, 2)), after=after)


if __name__ == "__main__":
    main()
