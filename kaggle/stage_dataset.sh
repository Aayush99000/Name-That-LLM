#!/bin/bash
# Assemble the Kaggle dataset + notebook folders on /scratch. Uploads nothing.
# Run on an Explorer login node (pip download needs internet):
#   bash kaggle/stage_dataset.sh BLEND_JSON RUN_DIR [RUN_DIR ...]
# e.g.
#   bash kaggle/stage_dataset.sh /scratch/$USER/name-that-llm/blend.json \
#       /scratch/$USER/name-that-llm/runs/deberta-v3-large
# Then upload (both private) with the commands printed at the end.
set -euo pipefail

BLEND=$1; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/scratch/$USER/name-that-llm
DS=$ROOT/kaggle_dataset
KN=$ROOT/kaggle_kernel
PY=/scratch/$USER/envs/name-that-llm/bin/python
KUSER=$($PY -c "import json, os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")

rm -rf "$DS" "$KN"
mkdir -p "$DS/code" "$DS/wheels" "$KN"

cp "$REPO"/baseline.py "$REPO"/ensemble.py "$REPO"/train_transformer.py "$DS/code/"
cp "$BLEND" "$DS/blend.json"

for run in "$@"; do
    name=$(basename "$run")
    ls "$run"/fold*/model.safetensors >/dev/null   # fails loudly if the run has no saved models
    mkdir "$DS/$name"
    cp "$run/classes.json" "$run/config.json" "$DS/$name/"
    cp -r "$run"/fold* "$DS/$name/"
done

# Every transformer component named in blend.json must have been staged.
$PY - "$DS" <<'EOF'
import json, sys
from pathlib import Path
ds = Path(sys.argv[1])
runs = json.loads((ds / "blend.json").read_text())["runs"]
missing = [r for r in runs if r != "tfidf" and not (ds / r / "fold0").exists()]
if missing:
    sys.exit(f"blend.json names runs that were not staged: {missing}")
print("blend.json components:", runs)
EOF

# Pinned transformers for Kaggle images that ship v4 (installed only if needed).
for pyver in 3.11 3.12; do
    $PY -m pip download -q -d "$DS/wheels" --only-binary=:all: --platform manylinux2014_x86_64 \
        --implementation cp --python-version "$pyver" transformers==5.24.1
done

cat > "$DS/dataset-metadata.json" <<EOF
{"title": "name-that-llm-models", "id": "$KUSER/name-that-llm-models",
 "licenses": [{"name": "CC0-1.0"}]}
EOF

cp "$REPO/kaggle/name-that-llm-submission.ipynb" "$KN/"
cat > "$KN/kernel-metadata.json" <<EOF
{"id": "$KUSER/name-that-llm-submission", "title": "name-that-llm-submission",
 "code_file": "name-that-llm-submission.ipynb", "language": "python", "kernel_type": "notebook",
 "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": ["$KUSER/name-that-llm-models"], "competition_sources": ["name-that-llm"],
 "kernel_sources": []}
EOF

du -sh "$DS" "$DS"/*
cat <<EOF

Staged. To upload (private):
  kaggle datasets create -p $DS --dir-mode zip            # first time
  kaggle datasets version -p $DS --dir-mode zip -m "msg"  # later updates
  kaggle kernels push -p $KN
EOF
