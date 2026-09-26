"""Build the 10,732-clip index with Good/Bad labels and official-fold splits, and cache log-Mel features."""
import numpy as np
import pandas as pd
import torch

from common import (BAD_ESC50, BAD_US8K, DATA, ESC50_SPLIT, OUT, US8K_SPLIT,
                    load_clip, logmel_transform)


def build_index():
    us = pd.read_csv(DATA / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv")
    us_rows = pd.DataFrame({
        "corpus": "UrbanSound8K",
        "file": [str(DATA / "UrbanSound8K" / "audio" / f"fold{f}" / n)
                 for f, n in zip(us["fold"], us["slice_file_name"])],
        "source_class": us["class"],
        "fold": us["fold"],
        "split": us["fold"].map(US8K_SPLIT),
        "label": us["class"].isin(BAD_US8K).astype(int),
    })
    esc = pd.read_csv(DATA / "ESC-50-master" / "meta" / "esc50.csv")
    esc_rows = pd.DataFrame({
        "corpus": "ESC-50",
        "file": [str(DATA / "ESC-50-master" / "audio" / n) for n in esc["filename"]],
        "source_class": esc["category"],
        "fold": esc["fold"],
        "split": esc["fold"].map(ESC50_SPLIT),
        "label": esc["category"].isin(BAD_ESC50).astype(int),
    })
    return pd.concat([us_rows, esc_rows], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = build_index()
    assert len(idx) == 10732, len(idx)
    idx.to_csv(OUT / "clip_index.csv", index=False)

    to_mel = logmel_transform()
    feats = []
    for i, path in enumerate(idx["file"]):
        with torch.no_grad():
            feats.append(to_mel(load_clip(path)).numpy().astype(np.float16))
        if i % 1000 == 0:
            print(f"{i}/{len(idx)}", flush=True)
    feats = np.stack(feats)
    print("features", feats.shape)
    np.save(DATA / "logmel_64x173.npy", feats)


if __name__ == "__main__":
    main()
