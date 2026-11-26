"""Fine-tune a transformer classifier for Name That LLM with prompt-grouped CV.

Same folds as baseline.py (GroupKFold on prompt_id), so out-of-fold
probabilities can be ensembled with the TF-IDF models.

Examples:
    python train_transformer.py --model microsoft/deberta-v3-base
    python train_transformer.py --model microsoft/deberta-v3-large --lr 1e-5 --bs 8 --grad_accum 2
    python train_transformer.py --model Qwen/Qwen3-0.6B --lr 3e-5 --lora      # decoder + LoRA
    python train_transformer.py --smoke                                         # tiny CPU/GPU check

Outputs (in --out_root/<run_name>/): oof.npy, test_probs.npy, metrics.json,
classes.json, config.json, and fold*/ model weights if --save_models.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_cosine_schedule_with_warmup)

KAGGLE_DIR = Path("/kaggle/input/competitions/name-that-llm")
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = KAGGLE_DIR if KAGGLE_DIR.exists() else REPO_DIR / "data"
DEFAULT_OUT = Path(f"/scratch/{os.environ.get('USER', 'user')}/name-that-llm/runs")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=None, help="LR for the classifier head (default 10x lr)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_fold", type=int, default=-1, help="run a single fold (for quick experiments)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", action="store_true", help="LoRA adapters (for decoder LLMs)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 fold, 1 epoch, short seqs")
    return ap.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def load_model(args, n_classes, tok):
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=n_classes)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=2 * args.lora_r,
                         lora_dropout=0.05, target_modules="all-linear")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return model


def param_groups(model, args):
    head_lr = args.head_lr or args.lr * 10
    head_keys = ("classifier", "score", "pooler")
    groups = {"body_decay": [], "body_nodecay": [], "head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in head_keys):
            groups["head"].append(p)
        elif p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias"):
            groups["body_nodecay"].append(p)
        else:
            groups["body_decay"].append(p)
    return [
        {"params": groups["body_decay"], "lr": args.lr, "weight_decay": args.wd},
        {"params": groups["body_nodecay"], "lr": args.lr, "weight_decay": 0.0},
        {"params": groups["head"], "lr": head_lr, "weight_decay": 0.0},
    ]


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(**batch).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def train_fold(args, fold, tr_df, va_df, te_df, tok, collate, n_classes, device, amp_dtype, out_dir):
    seed_everything(args.seed + fold)
    model = load_model(args, n_classes, tok).to(device)

    tr_ds = TextDS(tr_df.text, tr_df.label.to_numpy(), tok, args.max_len)
    va_ds = TextDS(va_df.text, va_df.label.to_numpy(), tok, args.max_len)
    te_ds = TextDS(te_df.text, None, tok, args.max_len)
    dl_kw = dict(collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(va_ds, batch_size=args.bs * 2, **dl_kw)
    te_dl = DataLoader(te_ds, batch_size=args.bs * 2, **dl_kw)

    opt = torch.optim.AdamW(param_groups(model, args), lr=args.lr)
    steps = math.ceil(len(tr_dl) / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history = []
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tr_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            running += loss.item() * args.grad_accum
        va_probs = predict(model, va_dl, device, amp_dtype)
        bal = balanced_accuracy_score(va_df.label, va_probs.argmax(1))
        nll = float(-np.log(va_probs[np.arange(len(va_df)), va_df.label] + 1e-12).mean())
        history.append({"epoch": epoch + 1, "train_loss": running / len(tr_dl), "val_bal_acc": bal,
                        "val_nll": nll, "secs": time.time() - t0})
        print(f"fold {fold} epoch {epoch + 1}: loss {history[-1]['train_loss']:.4f} "
              f"val_bal_acc {bal:.4f} val_nll {nll:.4f} ({history[-1]['secs']:.0f}s)", flush=True)

    # Fixed epoch count (no best-epoch picking) keeps the OOF score honest.
    te_probs = predict(model, te_dl, device, amp_dtype)
    if args.save_models:
        model.save_pretrained(out_dir / f"fold{fold}")
        tok.save_pretrained(out_dir / f"fold{fold}")
    del model, opt
    torch.cuda.empty_cache()
    return va_probs, te_probs, history


def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.max_len, args.only_fold = 1, 128, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None
    print(f"device {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}), amp {amp_dtype}")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    classes = sorted(train.model.unique())
    train["label"] = train.model.map({c: i for i, c in enumerate(classes)})

    run_name = args.run_name or (args.model.split("/")[-1] + ("-smoke" if args.smoke else ""))
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out_dir / "classes.json").write_text(json.dumps(classes))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    oof = np.full((len(train), len(classes)), np.nan)
    te_probs, histories = [], {}
    folds = list(GroupKFold(n_splits=args.folds).split(train, train.label, train.prompt_id))
    for fold, (tr_idx, va_idx) in enumerate(folds):
        if args.only_fold >= 0 and fold != args.only_fold:
            continue
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
        if args.smoke:
            tr_df, va_df = tr_df.sample(64, random_state=0), va_df.sample(36, random_state=0)
            va_idx = va_df.index.to_numpy()
        va_p, te_p, hist = train_fold(args, fold, tr_df, va_df, test, tok, collate,
                                      len(classes), device, amp_dtype, out_dir)
        oof[va_idx] = va_p
        te_probs.append(te_p)
        histories[fold] = hist
        np.save(out_dir / "oof.npy", oof)  # checkpoint after every fold

    done = ~np.isnan(oof[:, 0])
    y = train.label.to_numpy()[done]
    pred = oof[done].argmax(1)
    metrics = {
        "run": run_name, "n_oof": int(done.sum()),
        "bal_acc": balanced_accuracy_score(y, pred),
        "nll": float(-np.log(oof[done][np.arange(done.sum()), y] + 1e-12).mean()),
        "per_class_recall": {c: float((pred[y == i] == i).mean()) for i, c in enumerate(classes) if (y == i).any()},
        "history": histories,
    }
    np.save(out_dir / "test_probs.npy", np.mean(te_probs, axis=0))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n{run_name}: OOF balanced accuracy {metrics['bal_acc']:.4f}  nll {metrics['nll']:.4f}  "
          f"(n={metrics['n_oof']})  -> {out_dir}")


if __name__ == "__main__":
    main()
