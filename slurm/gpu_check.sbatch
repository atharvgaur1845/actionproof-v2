#!/bin/bash
#SBATCH --job-name=ap_gpucheck
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/pabitra/actionproof/logs/%x-%j.out
# G0 prerequisite: driver must support the cu130 torch wheels; check torch sees the GPU.
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
/scratch/pabitra/actionproof/.venv/bin/python -c "import torch;print('cuda ok', torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
