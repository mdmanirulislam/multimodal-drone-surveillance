"""Regenerate Table IV, Figs. 5-6, per-class recall/counts, siren share, Table V, and the seed summary.

Primary run = seed 0 (declared before training). Other seeds are reported as mean +/- std.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from common import OUT
from train_audio import best_threshold

PRIMARY = 0


def metrics(d, thr):
    y, p = d["label"].values, d["p_bad"].values
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    return {"n_test": int(len(y)), "accuracy": (tp + tn) / len(y), "precision_bad": prec, "recall_bad": rec,
            "f1_bad": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "TN_good_correct": int(tn), "FP_good_to_bad": int(fp), "FN_bad_to_good": int(fn),
            "TP_bad_correct": int(tp), "threshold": thr}


def main():
    seeds = sorted(int(p.name[4:]) for p in OUT.glob("seed*") if (p / "run.json").exists())
    rows = []
    for s in seeds:
        run = json.load(open(OUT / f"seed{s}" / "run.json"))
        v = pd.read_csv(OUT / f"seed{s}" / "val_scores.csv")
        run["threshold"] = best_threshold(v["label"].values, v["p_bad"].values)[0]
        m = metrics(pd.read_csv(OUT / f"seed{s}" / "test_scores.csv"), run["threshold"])
        rows.append({"seed": s, "best_epoch": run["best_epoch"], **m})
    allm = pd.DataFrame(rows); allm.to_csv(OUT / "audio_metrics_all_seeds.csv", index=False)
    keys = ["accuracy", "precision_bad", "recall_bad", "f1_bad", "roc_auc", "average_precision"]
    summ = allm[keys].agg(["mean", "std"]).T; summ.to_csv(OUT / "audio_metrics_seed_summary.csv")

    run = json.load(open(OUT / f"seed{PRIMARY}" / "run.json"))
    d = pd.read_csv(OUT / f"seed{PRIMARY}" / "test_scores.csv")
    v = pd.read_csv(OUT / f"seed{PRIMARY}" / "val_scores.csv")
    thr = best_threshold(v["label"].values, v["p_bad"].values)[0]; m = metrics(d, thr)
    json.dump(m, open(OUT / "table_IV_primary_seed0.json", "w"), indent=2)
    with open(OUT / "table_IV_primary_seed0.md", "w") as f:
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Accuracy | {100*m['accuracy']:.2f}% |\n| Precision (Bad) | {100*m['precision_bad']:.2f}% |\n"
                f"| Recall (Bad) | {100*m['recall_bad']:.2f}% |\n| F1-Score (Bad) | {100*m['f1_bad']:.2f}% |\n"
                f"| ROC-AUC | {m['roc_auc']:.3f} |\n| Average Precision | {m['average_precision']:.3f} |\n")

    # Per-source-class recall (share of clips assigned their own Good/Bad label) with test-clip counts.
    d["pred"] = (d["p_bad"] >= thr).astype(int)
    pc = (d.assign(correct=d["pred"] == d["label"])
           .groupby(["corpus", "source_class", "label"])
           .agg(n_test=("correct", "size"), n_correct=("correct", "sum")).reset_index())
    pc["recall"] = pc["n_correct"] / pc["n_test"]
    pc["label"] = pc["label"].map({0: "Good", 1: "Bad"})
    pc.sort_values(["label", "recall"], ascending=[True, False]).to_csv(OUT / "per_class_recall_test_seed0.csv", index=False)

    # Siren share of Bad test clips.
    bad = d[d["label"] == 1]
    siren = bad["source_class"] == "siren"
    share = {"bad_test_clips": int(len(bad)), "siren_bad_test_clips": int(siren.sum()),
             "siren_share_pct": 100 * siren.mean(),
             "siren_by_corpus": bad[siren].groupby("corpus").size().to_dict()}
    json.dump(share, open(OUT / "siren_share.json", "w"), indent=2)

    # Clip counts per split and label; Table V mapping.
    idx = pd.read_csv(OUT / "clip_index.csv")
    idx.assign(label=idx["label"].map({0: "Good", 1: "Bad"})).pivot_table(
        index=["corpus", "split"], columns="label", values="file", aggfunc="count", margins=True
    ).to_csv(OUT / "clip_counts_by_split.csv")
    idx.groupby(["corpus", "source_class"])["label"].first().map({0: "Good", 1: "Bad"}).reset_index() \
        .to_csv(OUT / "table_V_mapping.csv", index=False)

    # Fig. 5: PR curve.
    y, p = d["label"].values, d["p_bad"].values
    pr, rc, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(3.5, 3)); plt.plot(rc, pr, lw=1.5)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AP = {m['average_precision']:.3f})", fontsize=9); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "fig5_pr_curve.png", dpi=300); plt.close()

    # Fig. 6: (a) confusion matrix, (b) ROC.
    cm = np.array([[m["TN_good_correct"], m["FP_good_to_bad"]], [m["FN_bad_to_good"], m["TP_bad_correct"]]])
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[0].set_xticks([0, 1], ["Good", "Bad"]); ax[0].set_yticks([0, 1], ["Good", "Bad"])
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True"); ax[0].set_title("(a) Confusion matrix", fontsize=9)
    fpr, tpr, _ = roc_curve(y, p)
    ax[1].plot(fpr, tpr, lw=1.5); ax[1].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
    ax[1].set_title(f"(b) ROC (AUC = {m['roc_auc']:.3f})", fontsize=9); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig6_cm_roc.png", dpi=300); plt.close(fig)

    print(json.dumps(m, indent=2)); print(summ); print(json.dumps(share, indent=2))


if __name__ == "__main__":
    main()
