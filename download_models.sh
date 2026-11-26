#!/bin/bash
# Pre-download pretrained weights into the shared HF cache on /scratch.
# Run on the LOGIN node (compute nodes have no direct internet):
#   bash download_models.sh [model_id ...]
set -euo pipefail

export HF_HOME="/scratch/$USER/hf_cache"
PY="/scratch/$USER/envs/name-that-llm/bin/python"
module purge
module load python/3.13.5

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(microsoft/deberta-v3-base microsoft/deberta-v3-large)
fi

for m in "${MODELS[@]}"; do
    echo "== $m"
    # Weights: safetensors when the repo has them, else pytorch_model.bin
    # (e.g. DeBERTa-v3). Skips TF/Rust/generator weights.
    "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$m', allow_patterns=['*.json','*.model','*.txt','*.safetensors','*.tiktoken','*.py','pytorch_model.bin','pytorch_model-*.bin']))"
done
