#!/bin/bash
# Pull result files of research/latent-state-theory jobs back from the cluster working copy.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/theory results/research
rsync -a sharanga:/scratch/pabitra/lab_theory/results/theory/ results/theory/
rsync -a sharanga:/scratch/pabitra/lab_theory/results/research/ results/research/
