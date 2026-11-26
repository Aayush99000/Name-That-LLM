#!/bin/bash
# Quick sanity check that the project env sees a GPU.
#   sbatch slurm/gpu_check.sbatch
#SBATCH --job-name=ntl-gpucheck
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_check_%j.out

module purge
module load python/3.13.5
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
/scratch/$USER/envs/name-that-llm/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok:", torch.isfinite(x @ x).all().item())
EOF
