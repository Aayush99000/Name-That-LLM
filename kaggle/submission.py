# %% [markdown]
# # Name That LLM: TF-IDF + DeBERTa blend (offline inference)
#
# Needs two inputs attached:
# * the `name-that-llm` competition data;
# * the `name-that-llm-models` dataset made by `kaggle/stage_dataset.sh`. It holds
#   `blend.json` (components and weights), `code/` (baseline.py, ensemble.py,
#   train_transformer.py), one folder per transformer run with `fold*/` models, and
#   `wheels/` (a pinned transformers, used only if Kaggle's copy is older than v5).
#
# The TF-IDF model is refit on the full train set here; transformer folds are only
# run for inference. Weights were chosen on prompt-grouped out-of-fold predictions.

# %%
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_models_dir():
    if os.environ.get("NTL_MODELS_DIR"):
        return Path(os.environ["NTL_MODELS_DIR"])
    hits = sorted(Path("/kaggle/input").glob("**/blend.json"))
    if not hits:
        raise FileNotFoundError("blend.json not found: attach the name-that-llm-models dataset")
    return hits[0].parent


def find_one(pattern):
    """Kaggle unpacks uploaded sub-folders itself, sometimes one level deeper; search, don't assume."""
    hits = sorted(MODELS.glob(f"**/{pattern}"), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {MODELS}")
    return hits[0]


MODELS = find_models_dir()
OUT = Path(os.environ.get("NTL_OUT", "/kaggle/working" if Path("/kaggle/working").exists() else "."))
print("models dataset:", MODELS)

# Fold models were saved by transformers v5; install the pinned wheel offline only if needed.
tf_version = importlib.metadata.version("transformers")
print("transformers", tf_version)
if int(tf_version.split(".")[0]) < 5:
    wheels = find_one("transformers-*.whl").parent
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                    "--find-links", str(wheels), "transformers==5.24.1"], check=True)
    print("installed transformers", importlib.metadata.version("transformers"))

# %%
import numpy as np
import pandas as pd

sys.path.insert(0, str(find_one("ensemble.py").parent))
import baseline as B  # noqa: E402
import ensemble as E  # noqa: E402

cfg = json.loads((MODELS / "blend.json").read_text())
print("blend:", dict(zip(cfg["runs"], cfg["weights"])), "| OOF balanced accuracy", cfg.get("oof_bal_acc"))

train = pd.read_csv(B.DATA_DIR / "train.csv")
test = pd.read_csv(B.DATA_DIR / "test.csv")
classes = np.array(sorted(train.model.unique()))
print(f"train {len(train)}  test {len(test)}  classes {len(classes)}")

# %%
probs = {}
t0 = time.time()
Xtr, Xte = B.build_features("char+word", train.text, test.text)
lr = B.make_model("logreg").fit(Xtr, train.model)
assert list(lr.classes_) == list(classes)
probs["tfidf"] = lr.predict_proba(Xte)
print(f"tfidf done ({time.time() - t0:.0f}s)")

# %%
for name in cfg["runs"]:
    if name == "tfidf":
        continue
    run = find_one(f"{name}/classes.json").parent
    if not any(run.glob("fold*/config.json")):
        raise FileNotFoundError(f"{run}: no fold*/ models")
    if json.loads((run / "classes.json").read_text()) != list(classes):
        raise ValueError(f"{name}: class order differs from train")
    max_len = json.loads((run / "config.json").read_text()).get("max_len", 512)
    t0 = time.time()
    probs[name] = E.predict_with_fold_models(run, test.text, max_len)
    print(f"{name} done ({time.time() - t0:.0f}s)")

# %%
blended = E.blend([probs[n] for n in cfg["runs"]], np.array(cfg["weights"]))
sub = pd.DataFrame({"id": test.id, "model": classes[blended.argmax(1)]})
E.validate_submission(sub, test, classes)
sub.to_parquet(OUT / "submission.parquet", index=False)
print(f"wrote {OUT / 'submission.parquet'} ({len(sub)} rows)")
print(sub.model.value_counts().to_string())
