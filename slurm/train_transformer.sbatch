#!/bin/bash
# 5-fold transformer fine-tune. Extra args go straight to train_transformer.py:
#   sbatch slurm/train_transformer.sbatch --model microsoft/deberta-v3-base
#   sbatch --gres=gpu:a100:1 slurm/train_transformer.sbatch --model microsoft/deberta-v3-large --lr 1e-5
#SBATCH --job-name=ntl-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out

set -euo pipefail
module purge
module load python/3.13.5

export HF_HOME="/scratch/$USER/hf_cache"
export HF_HUB_OFFLINE=1           # weights come from download_models.sh
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/Name-That-LLM"
/scratch/$USER/envs/name-that-llm/bin/python train_transformer.py "$@"
