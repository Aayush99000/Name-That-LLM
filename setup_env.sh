#!/bin/bash
# setup_env.sh — build the project venv on the Explorer cluster.
# Run on the LOGIN node (it has internet): bash setup_env.sh
#
# The venv lives on /scratch, not in $HOME: the home quota is too small
# for torch + CUDA libraries. Activate with:
#   source /scratch/katoch.aa/envs/name-that-llm/bin/activate
#
# torch uses the cu126 wheel: it covers every GPU on the cluster
# (V100 sm70, T4 sm75, A100 sm80, H200 sm90); cu128+ wheels drop Volta.

set -euo pipefail

PROJECT="$HOME/Name-That-LLM"
ENV_DIR="/scratch/$USER/envs/name-that-llm"

# Keep pip's cache and temp files off $HOME as well.
export PIP_CACHE_DIR="/scratch/$USER/pip_cache"
export TMPDIR="/scratch/$USER/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$ENV_DIR")"

module purge
module load python/3.13.5

if [ ! -x "$ENV_DIR/bin/python" ]; then
    python3 -m venv "$ENV_DIR"
fi
PIP="$ENV_DIR/bin/pip"

"$PIP" install --upgrade pip wheel setuptools
"$PIP" install torch --index-url https://download.pytorch.org/whl/cu126
"$PIP" install -r "$PROJECT/requirements.txt"

"$ENV_DIR/bin/python" - <<'EOF'
import torch, transformers, sklearn, pandas
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "sklearn", sklearn.__version__, "pandas", pandas.__version__)
EOF
echo "ENV READY: $ENV_DIR"
