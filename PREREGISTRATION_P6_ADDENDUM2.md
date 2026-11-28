# P6 addendum 2: seed 2 runs on a cluster A100

Written 2026-11-29, after addendum 1 (commit `0057582`) and before any P6 training or evaluation has started.

Addendum 1 placed seed 2 of both arms on one sharanga H200. That submission was refused: the H200 QOS allows one
submitted job per user (`qos_gpu_h200`, MaxSubmitPU = 1), and the running worker `ap_w_h200` holds it. The H100 queue
has had jobs pending since 2026-11-25. Seed 2 therefore runs on one **A100 80 GB** (`gpu_a100_8`), both arms in
sequence on the same GPU, with the same job file:

    sbatch -p gpu_a100_8 --gres=gpu:1 -c 4 --mem=140G --time=10:00:00 -J p6_seed2 slurm/p6_cluster_seed2.sbatch

Both arms of each seed still share one platform and GPU type (seeds 0 and 1 on HF H200, seed 2 on the cluster A100).
Nothing else changes.

The A100 queue is long (29 jobs pending at 01:35 IST), and P1's recipe took about 3 h 35 min per arm on this A100, so
seed 2 is not expected to finish before the 2026-11-29 17:29 IST submission cutoff. Under the registered cutoff rule,
P6 is then not reported in the submitted papers, and no seed-0/1 result is quoted before the registered analysis has
run on all three seeds.
