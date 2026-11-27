#!/bin/bash
# Dry-run the Kaggle notebook source against the staged dataset on a GPU,
# and check the tokenizer fallback matches AutoTokenizer.
#   sbatch slurm/test_submission.sbatch
#SBATCH --job-name=ntl-subtest
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/subtest_%j.out

set -euo pipefail
module purge
module load python/3.13.5
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/scratch/$USER/envs/name-that-llm/bin/python
DS=/scratch/$USER/name-that-llm/kaggle_dataset
WORK=/scratch/$USER/name-that-llm/subtest
mkdir -p "$WORK"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$WORK"   # run outside the repo so code is imported from the staged dataset
NTL_DATA_DIR=$HOME/Name-That-LLM/data NTL_MODELS_DIR=$DS NTL_OUT=$WORK $PY "$HOME/Name-That-LLM/kaggle/submission.py"

$PY - "$DS" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "code"))
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast
fold = next(Path(sys.argv[1]).glob("*/fold0"))
texts = pd.read_csv(Path.home() / "Name-That-LLM/data/train.csv").text[:300].tolist()
auto = AutoTokenizer.from_pretrained(fold)
raw = PreTrainedTokenizerFast(tokenizer_file=str(fold / "tokenizer.json"), pad_token="[PAD]")
a = auto(texts, truncation=True, max_length=512)["input_ids"]
b = raw(texts, truncation=True, max_length=512)["input_ids"]
print("tokenizer fallback identical ids:", a == b, "| pad id", auto.pad_token_id, raw.pad_token_id)
EOF
