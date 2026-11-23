#!/bin/bash
#SBATCH -p compute
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -J p2_committee
#SBATCH -o /scratch/pabitra/actionproof/logs/p2_committee.%j.log
# P2 secondary (PREREGISTRATION_P2.md §6): the J1 four-family committee on seed-0 greedy test episodes of A, C, D, E, Eg.
# Waits (polling every 10 min) until those test outputs exist, then build -> run (spend stops at the $15 cap) -> analyze.
cd /scratch/pabitra/actionproof
need() { for a in A C D; do for s in p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo; do [ -f results/p1/test/${a}_s0__$s.summary.json ] || return 1; done; done
         for a in E Eg; do for s in p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo; do [ -f results/p2/test/${a}_s0__$s.summary.json ] || return 1; done; done; }
until need; do sleep 600; done
.venv/bin/python scripts/freeze_p2.py --check || exit 1
.venv/bin/python evaluation/llm_judge_committee/committee_p2.py build && \
.venv/bin/python evaluation/llm_judge_committee/committee_p2.py run && \
.venv/bin/python evaluation/llm_judge_committee/committee_p2.py analyze
