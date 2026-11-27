"""Blend TF-IDF and transformer runs and write submission.parquet.

Components are combined by a weighted average of log-probabilities. Weights
are searched on out-of-fold (OOF) predictions, which all share the same
prompt-grouped folds (GroupKFold on prompt_id), or fixed with --weights.

Each transformer run dir (from train_transformer.py) needs oof.npy and
classes.json, plus one of:
  fold*/       saved fold models (train with --save_models): predicts test.csv here,
               which is what a Kaggle submission needs (hidden test set);
  test_probs.npy  test predictions made at training time (only valid for the
               same test.csv, e.g. the 3 example rows locally).

Usage:
    python ensemble.py --runs outputs/deberta-v3-base
    python ensemble.py --runs RUN_A RUN_B --weights 0.6 0.2 0.2   # tfidf first
    python ensemble.py --no_tfidf --runs RUN_A RUN_B
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import baseline as B

EPS = 1e-9


# ---------------------------------------------------------------- components

def tfidf_component(train, test, kind="char+word", model_name="logreg"):
    """OOF probs via the same grouped CV as everything else, test probs from a full-train fit."""
    res, oof, classes, _ = B.run_cv(train, kind, model_name)
    Xtr, Xte = B.build_features(kind, train.text, test.text)
    m = B.make_model(model_name).fit(Xtr, train.model)
    assert list(m.classes_) == list(classes)
    print(f"[tfidf {kind}/{model_name}] OOF balanced accuracy {res['bal_acc']:.4f}")
    return oof, m.predict_proba(Xte)


def load_tokenizer(path):
    """AutoTokenizer, falling back to the raw tokenizer.json for older transformers
    versions that can't read a tokenizer_config.json written by transformers v5."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:  # noqa: BLE001 - any load failure gets the fallback
        print(f"  AutoTokenizer failed ({type(e).__name__}: {e}); loading tokenizer.json directly")
        cfg = json.loads((Path(path) / "tokenizer_config.json").read_text())
        special = {k: cfg[k] for k in ("bos_token", "eos_token", "unk_token", "sep_token",
                                       "pad_token", "cls_token", "mask_token")
                   if isinstance(cfg.get(k), str)}
        tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(path) / "tokenizer.json"), **special)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def predict_with_fold_models(run_dir, texts, max_len):
    """Average softmax over saved fold models; imports torch only when needed."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

    from train_transformer import TextDS, predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = None
    if device.type == "cuda":
        amp = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    probs = []
    for fold_dir in sorted(Path(run_dir).glob("fold*")):
        tok = load_tokenizer(fold_dir)
        # Fold weights are saved in fp16; CPU needs fp32 to run them.
        dtype = torch.float32 if device.type == "cpu" else "auto"
        model = AutoModelForSequenceClassification.from_pretrained(fold_dir, dtype=dtype).to(device)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.pad_token_id
        dl = DataLoader(TextDS(texts, None, tok, max_len), batch_size=32,
                        collate_fn=DataCollatorWithPadding(tok, pad_to_multiple_of=8))
        probs.append(predict(model, dl, device, amp))
        print(f"  predicted with {fold_dir.name}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.mean(probs, axis=0)


def run_component(run_dir, classes, test):
    run_dir = Path(run_dir)
    run_classes = json.loads((run_dir / "classes.json").read_text())
    if run_classes != list(classes):
        raise ValueError(f"{run_dir}: class order differs from train")
    filled_path = run_dir / "oof_filled.npy"
    if filled_path.exists() and not np.load(filled_path).all():
        raise ValueError(f"{run_dir}: OOF incomplete (single-fold or smoke run?)")
    oof = np.load(run_dir / "oof.npy")

    if any(run_dir.glob("fold*")):
        max_len = json.loads((run_dir / "config.json").read_text()).get("max_len", 512)
        print(f"[{run_dir.name}] predicting test with saved fold models")
        te = predict_with_fold_models(run_dir, test.text, max_len)
    elif (run_dir / "test_probs.npy").exists():
        te = np.load(run_dir / "test_probs.npy")
        if len(te) != len(test):
            raise ValueError(f"{run_dir}: test_probs.npy has {len(te)} rows but test.csv has {len(test)}; "
                             "retrain with --save_models to predict a new test set")
    else:
        raise FileNotFoundError(f"{run_dir}: no fold*/ models and no test_probs.npy")
    return oof, te


# ---------------------------------------------------------------- blending

def blend(probs_list, weights):
    return sum(w * np.log(p + EPS) for w, p in zip(weights, probs_list))


def simplex_grid(k, step):
    n = round(1 / step)
    for combo in itertools.product(range(n + 1), repeat=k - 1):
        if sum(combo) <= n:
            yield np.array([*combo, n - sum(combo)]) / n


def search_weights(oofs, y, step):
    if len(oofs) == 1:
        return np.array([1.0]), balanced_accuracy_score(y, oofs[0].argmax(1))
    best_w, best_s = None, -1.0
    for w in simplex_grid(len(oofs), step):
        s = balanced_accuracy_score(y, blend(oofs, w).argmax(1))
        if s > best_s + 1e-9:
            best_w, best_s = w, s
    return best_w, best_s


def validate_submission(sub, test, classes):
    assert list(sub.columns) == ["id", "model"]
    assert sub.id.is_unique and len(sub) == len(test) and set(sub.id) == set(test.id)
    assert sub.model.notna().all() and set(sub.model) <= set(classes)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="transformer run directories")
    ap.add_argument("--no_tfidf", action="store_true")
    ap.add_argument("--weights", nargs="*", type=float, help="fixed weights (tfidf first, then runs)")
    ap.add_argument("--step", type=float, default=0.05, help="weight grid step")
    ap.add_argument("--out", default="submission.parquet")
    ap.add_argument("--save_blend", help="write the chosen components and weights to this JSON "
                                         "(read by the Kaggle notebook as blend.json)")
    args = ap.parse_args()

    train = pd.read_csv(B.DATA_DIR / "train.csv")
    test = pd.read_csv(B.DATA_DIR / "test.csv")
    classes = np.array(sorted(train.model.unique()))
    y = train.model.map({c: i for i, c in enumerate(classes)}).to_numpy()

    names, oofs, tests = [], [], []
    if not args.no_tfidf:
        o, t = tfidf_component(train, test)
        names.append("tfidf"); oofs.append(o); tests.append(t)
    for r in args.runs:
        o, t = run_component(r, classes, test)
        print(f"[{Path(r).name}] OOF balanced accuracy {balanced_accuracy_score(y, o.argmax(1)):.4f}")
        names.append(Path(r).name); oofs.append(o); tests.append(t)
    if not names:
        ap.error("nothing to blend")

    if args.weights:
        if len(args.weights) != len(names):
            ap.error(f"--weights needs {len(names)} values for {names}")
        w = np.array(args.weights) / sum(args.weights)
        score = balanced_accuracy_score(y, blend(oofs, w).argmax(1))
    else:
        w, score = search_weights(oofs, y, args.step)
    print("\nweights: " + ", ".join(f"{n}={x:.2f}" for n, x in zip(names, w)))
    print(f"blended OOF balanced accuracy {score:.4f}"
          + ("" if args.weights else "  (weights tuned on OOF: slightly optimistic)"))
    if args.save_blend:
        Path(args.save_blend).write_text(json.dumps(
            {"runs": names, "weights": [round(float(x), 4) for x in w], "oof_bal_acc": round(score, 4)},
            indent=2))
        print(f"wrote {args.save_blend}")

    pred = classes[blend(tests, w).argmax(1)]
    sub = pd.DataFrame({"id": test.id, "model": pred})
    validate_submission(sub, test, classes)
    sub.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} ({len(sub)} rows)")
    print(sub.model.value_counts().to_string())


if __name__ == "__main__":
    main()
