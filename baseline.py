"""TF-IDF + linear baseline for Name That LLM.

Validation holds out whole prompts (GroupKFold on prompt_id) because test
prompts never appear in train. Metric is balanced accuracy.

Usage:
    python baseline.py                # CV report for all feature sets
    python baseline.py --submit       # fit best config on all train, write submission.parquet
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

KAGGLE_DIR = Path("/kaggle/input/competitions/name-that-llm")
DATA_DIR = KAGGLE_DIR if KAGGLE_DIR.exists() else Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def lab_of(model: str) -> str:
    for prefix, lab in [("gpt", "openai"), ("gemini", "google"), ("claude", "anthropic"),
                        ("qwen", "qwen"), ("glm", "zai"), ("kimi", "moonshot")]:
        if model.startswith(prefix):
            return lab
    return "other"


# ---------------------------------------------------------------- features

STYLE_PATTERNS = {
    "emdash": "—", "endash": "–", "curly_apos": "’", "straight_apos": "'",
    "curly_dq": "[“”]", "straight_dq": '"', "semicolon": ";", "colon": ":",
    "comma": ",", "paren": r"\(", "question": r"\?", "exclaim": "!",
    "digit": r"\d", "hyphen_word": r"\w-\w", "non_ascii": r"[^\x00-\x7F]",
}


def style_features(texts: pd.Series) -> np.ndarray:
    """Dense hand-crafted stylometry, length-normalised where it makes sense."""
    n_chars = texts.str.len().clip(lower=1)
    words = texts.str.split()
    n_words = words.str.len().clip(lower=1)
    sents = texts.apply(lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t) if s])
    cols = {
        "log_chars": np.log(n_chars),
        "log_words": np.log(n_words),
        "mean_word_len": n_chars / n_words,
        "n_sents": sents.str.len(),
        "mean_sent_words": n_words / sents.str.len().clip(lower=1),
        "type_token": words.apply(lambda w: len({x.lower() for x in w}) / max(len(w), 1)),
    }
    for name, pat in STYLE_PATTERNS.items():
        cnt = texts.str.count(pat)
        cols[f"{name}_per100w"] = 100 * cnt / n_words
        cols[f"{name}_any"] = (cnt > 0).astype(float)
    return pd.DataFrame(cols).to_numpy(dtype=np.float64)


def make_vectorizers(kind: str):
    vecs = []
    if kind in ("char", "char+word", "all"):
        vecs.append(TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                                    sublinear_tf=True, min_df=2, max_features=400_000))
    if kind in ("word", "char+word", "all"):
        vecs.append(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                                    token_pattern=r"(?u)\b\w+\b|[^\w\s]",
                                    sublinear_tf=True, min_df=2, max_features=200_000))
    return vecs


def build_features(kind, tr_text, va_text):
    blocks_tr, blocks_va = [], []
    for v in make_vectorizers(kind):
        blocks_tr.append(v.fit_transform(tr_text))
        blocks_va.append(v.transform(va_text))
    if kind in ("style", "all"):
        sc = StandardScaler()
        weight = 1.0 if kind == "style" else 0.3  # keep dense block from dominating tf-idf
        blocks_tr.append(sparse.csr_matrix(weight * sc.fit_transform(style_features(tr_text))))
        blocks_va.append(sparse.csr_matrix(weight * sc.transform(style_features(va_text))))
    return sparse.hstack(blocks_tr).tocsr(), sparse.hstack(blocks_va).tocsr()


def make_model(name):
    if name == "logreg":
        return LogisticRegression(C=20, max_iter=3000, class_weight="balanced")
    if name == "svm":
        return LinearSVC(C=0.5, class_weight="balanced")
    raise ValueError(name)


def scores_of(model, X):
    return model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X)


# ---------------------------------------------------------------- CV

def prompt_assignment(scores, prompt_ids, classes):
    """Oracle-grouping upper bound: within each prompt, each model is used once.

    Test has no prompt_id, so this only measures how much a recovered grouping
    could be worth; it is not usable as-is at inference time.
    """
    pred = np.empty(len(scores), dtype=object)
    for pid in np.unique(prompt_ids):
        idx = np.where(prompt_ids == pid)[0]
        s = scores[idx]
        if s.min() >= 0:  # probabilities -> log space
            s = np.log(s + 1e-12)
        rows, cols = linear_sum_assignment(-s)
        pred[idx[rows]] = classes[cols]
    return pred


def run_cv(df, kind, model_name, n_splits=5):
    y = df.model.to_numpy()
    groups = df.prompt_id.to_numpy()
    classes = np.array(sorted(df.model.unique()))
    oof = np.zeros((len(df), len(classes)))
    t0 = time.time()
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(df, y, groups):
        Xtr, Xva = build_features(kind, df.text.iloc[tr_idx], df.text.iloc[va_idx])
        m = make_model(model_name).fit(Xtr, y[tr_idx])
        assert list(m.classes_) == list(classes)
        oof[va_idx] = scores_of(m, Xva)
    pred = classes[oof.argmax(1)]
    labs = np.array([lab_of(c) for c in classes])
    res = {
        "features": kind, "model": model_name,
        "bal_acc": balanced_accuracy_score(y, pred),
        "lab_acc": np.mean([lab_of(a) == lab_of(b) for a, b in zip(y, pred)]),
        "bal_acc_oracle_groups": balanced_accuracy_score(y, prompt_assignment(oof, groups, classes)),
        "secs": time.time() - t0,
    }
    return res, oof, classes, labs


def per_class_report(df, oof, classes):
    y = df.model.to_numpy()
    pred = classes[oof.argmax(1)]
    rows = []
    for c in classes:
        mask = y == c
        wrong = pd.Series(pred[mask & (pred != c)]).value_counts()
        rows.append({"model": c, "recall": (pred[mask] == c).mean(),
                     "top_confusion": f"{wrong.index[0]} ({wrong.iloc[0]})" if len(wrong) else "-"})
    return pd.DataFrame(rows).sort_values("recall")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--features", default="all")
    ap.add_argument("--model", default="logreg")
    args = ap.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv")

    if args.submit:
        test = pd.read_csv(DATA_DIR / "test.csv")
        Xtr, Xte = build_features(args.features, train.text, test.text)
        m = make_model(args.model).fit(Xtr, train.model)
        sub = pd.DataFrame({"id": test.id, "model": m.predict(Xte)})
        sub.to_parquet("submission.parquet", index=False)
        print(sub.head(), f"\nwrote submission.parquet ({len(sub)} rows)")
        return

    OUT_DIR.mkdir(exist_ok=True)
    results = []
    best = None
    for kind in ["style", "word", "char", "char+word", "all"]:
        for model_name in ["logreg", "svm"]:
            res, oof, classes, _ = run_cv(train, kind, model_name)
            results.append(res)
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
            if best is None or res["bal_acc"] > best[0]["bal_acc"]:
                best = (res, oof, classes)

    table = pd.DataFrame(results).sort_values("bal_acc", ascending=False)
    table.to_csv(OUT_DIR / "baseline_cv.csv", index=False)
    print("\n" + table.round(4).to_string(index=False))

    res, oof, classes = best
    np.save(OUT_DIR / "oof_best.npy", oof)
    print(f"\nPer-class recall, best = {res['features']} + {res['model']}:")
    print(per_class_report(train, oof, classes).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
