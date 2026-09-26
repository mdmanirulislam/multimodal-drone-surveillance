"""Train the AudioCNN on the official folds; tune the threshold on validation; score the test folds.

Usage: python train_audio.py --seed 0 [--epochs 40]
Writes outputs/seed<k>/{model.pt, test_scores.csv, val_scores.csv, history.csv, run.json}.
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import BAD_WEIGHT, DATA, OUT, AudioCNN


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def spec_augment(x, n_freq=2, f_w=8, n_time=2, t_w=20, noise=0.05):
    """SpecAugment frequency/time masking plus additive Gaussian noise (on normalized log-Mel)."""
    b, _, F, T = x.shape
    x = x.clone()
    for i in range(b):
        for _ in range(n_freq):
            w = random.randint(0, f_w); f0 = random.randint(0, F - w)
            x[i, :, f0:f0 + w, :] = 0
        for _ in range(n_time):
            w = random.randint(0, t_w); t0 = random.randint(0, T - w)
            x[i, :, :, t0:t0 + w] = 0
    return x + noise * torch.randn_like(x)


def predict(model, X, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(X[i:i + bs]), 1)[:, 1])
    return torch.cat(out).numpy()


def best_threshold(y, p):
    grid = np.linspace(0.01, 0.99, 197)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))]), float(np.max(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    set_seed(a.seed)
    torch.set_num_threads(12)

    idx = pd.read_csv(OUT / "clip_index.csv")
    feats = torch.from_numpy(np.load(DATA / "logmel_64x173.npy").astype(np.float32)).unsqueeze(1)
    tr, va, te = (idx["split"] == s for s in ("train", "val", "test"))
    mu, sd = feats[tr.values].mean(), feats[tr.values].std()     # train-fold statistics only
    feats = (feats - mu) / sd
    Xtr, ytr = feats[tr.values], torch.tensor(idx.loc[tr, "label"].values)
    Xva, yva = feats[va.values], idx.loc[va, "label"].values
    Xte, yte = feats[te.values], idx.loc[te, "label"].values

    model = AudioCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, BAD_WEIGHT]))

    run_dir = OUT / f"seed{a.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    hist, best_ap, t0 = [], -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            loss = loss_fn(model(spec_augment(Xtr[b])), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        sched.step()
        pva = predict(model, Xva)
        val_ap, val_auc = average_precision_score(yva, pva), roc_auc_score(yva, pva)
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "val_ap": val_ap, "val_auc": val_auc})
        print(f"seed {a.seed} ep {ep:02d} loss {tot/len(perm):.4f} val AP {val_ap:.4f} AUC {val_auc:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if val_ap > best_ap:                                   # model selection on validation folds only
            best_ap = val_ap
            torch.save({"state": model.state_dict(), "mu": float(mu), "sd": float(sd), "epoch": ep},
                       run_dir / "model.pt")

    ck = torch.load(run_dir / "model.pt")
    model.load_state_dict(ck["state"])
    pva, pte = predict(model, Xva), predict(model, Xte)
    thr, val_f1 = best_threshold(yva, pva)                    # threshold tuned on validation folds only
    pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
    for name, mask, p in (("val", va, pva), ("test", te, pte)):
        d = idx.loc[mask, ["corpus", "file", "source_class", "fold", "label"]].copy()
        d["p_bad"] = p
        d.to_csv(run_dir / f"{name}_scores.csv", index=False)
    json.dump({"seed": a.seed, "epochs": a.epochs, "batch_size": a.bs, "lr": a.lr,
               "best_epoch": ck["epoch"], "best_val_ap": best_ap, "threshold": thr,
               "val_f1_at_threshold": val_f1, "train_seconds": time.time() - t0},
              open(run_dir / "run.json", "w"), indent=2)
    print("done", run_dir, "threshold", thr)


if __name__ == "__main__":
    main()
