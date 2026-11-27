#!/bin/bash
# Run a Kaggle CLI command on a compute node (login nodes kill long uploads;
# compute nodes reach the internet through the cluster proxy). Args go to `kaggle`:
#   sbatch slurm/kaggle_cli.sbatch datasets create -p /scratch/$USER/name-that-llm/kaggle_upload --dir-mode skip
#   sbatch slurm/kaggle_cli.sbatch kernels push -p /scratch/$USER/name-that-llm/kaggle_kernel
#SBATCH --job-name=ntl-kaggle
#SBATCH --partition=sharing
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/kaggle_%j.out

set -euo pipefail
module purge
module load python/3.13.5
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR"
# Drop tqdm progress frames from the log; keep the CLI's own exit status.
set +e
/scratch/$USER/envs/name-that-llm/bin/kaggle "$@" 2>&1 | tr '\r' '\n' | grep -v '^ *[0-9]*%|'
status=${PIPESTATUS[0]}
exit "$status"
