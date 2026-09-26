# Supplementary material: fold-respecting acoustic evaluation

Supplement to "Multimodal Autonomous Drone Surveillance for Campus Security: YOLO11-Based Person and
ID-Card Detection with Acoustic Threat Classification". It contains the code, per-clip scores and
configuration behind Table IV, Table V, Figs. 5–6, the per-source-class recall in Section IV-B, and the
indicative inference timings in Section IV-C.

## Data (not redistributed)

- UrbanSound8K: Salamon et al. (2014), https://zenodo.org/records/1203745 (8,732 clips)
- ESC-50: Piczak (2015), https://github.com/karolpiczak/ESC-50 (2,000 clips)

Put them at `<DATA>/UrbanSound8K/` and `<DATA>/ESC-50-master/`, where `<DATA>` is the `data/` folder next to `code/`.

## Protocol

| Split | UrbanSound8K folds | ESC-50 folds | Clips (Good / Bad) |
|---|---|---|---|
| Train | 1–8 | 1–3 | 8,279 (7,084 / 1,195) |
| Validation | 9 | 4 | 1,216 (1,063 / 153) |
| Test | 10 | 5 | 1,237 (1,082 / 155) |

- **Labels:** Bad = UrbanSound8K {gun_shot, siren}; ESC-50 {fireworks, glass_breaking, hand_saw, chainsaw, siren}.
  Everything else is Good (`results/table_V_mapping.csv`).
- **Features:** 22.05 kHz mono, 4-s pad/trim, 64-bin log-Mel (n_fft 1024, hop 512, 80 dB top), normalized with training-fold mean/std.
- **Model:** four blocks of Conv3×3–BN–ReLU–MaxPool (32/64/128/256 channels), then global average pooling and a
  Dropout–FC(128)–ReLU–Dropout–FC(2) classifier; 0.42 M parameters.
- **Training:** weighted cross-entropy (Bad 5.5×), SpecAugment (2 frequency masks ≤ 8 bins, 2 time masks ≤ 20 frames)
  plus Gaussian noise σ = 0.05, AdamW (lr 1e-3, wd 1e-4), batch 64, 40 epochs, cosine annealing.
- **Selection:** the checkpoint with the best validation average precision, and the threshold that maximizes validation F1
  (grid 0.01–0.99). The test folds were not used for any choice.
- **Seeds:** 0, 1 and 2. Seed 0 was declared the primary run before training.

## Results (test folds, 1,237 clips)

| Metric | Seed 0 (Table IV) | Mean ± std, seeds 0–2 |
|---|---|---|
| Accuracy | 93.13% | 93.18 ± 0.05% |
| Precision (Bad) | 84.31% | 82.87 ± 2.63% |
| Recall (Bad) | 55.48% | 57.63 ± 3.18% |
| F1 (Bad) | 66.93% | 67.90 ± 1.27% |
| ROC-AUC | 0.877 | 0.883 ± 0.012 |
| Average precision | 0.734 | 0.743 ± 0.008 |

Seed-0 confusion matrix: TN 1,066, FP 16, FN 69, TP 86 (threshold 0.94). Sirens are 91 of the 155 Bad
test clips (58.7%). Per-source-class recall with clip counts is in `results/per_class_recall_test_seed0.csv`.

## Inference timing (indicative, CPU only)

Intel Core Ultra 7 155U (12 cores / 14 threads), 31.5 GB RAM, integrated graphics only, Windows 11,
PyTorch 2.14.0 CPU, Ultralytics 8.4.163. Timings use batch 1 and a mean over 100–200 runs after warm-up (`results/benchmark_this_laptop.json`).

- YOLO11n / s / m (COCO weights, 640×640 `predict()` including pre/post-processing):
  66.8 / 157.3 / 385.1 ms (≈15.0 / 6.4 / 2.6 FPS)
- AudioCNN, one 4-s window including log-Mel extraction: 5.4 ms

## Reproduce

```
python -m venv .venv && .venv\Scripts\pip install -r code/requirements.txt   # torch/torchaudio from the PyTorch CPU index
cd code
python prepare_features.py          # builds clip index + cached log-Mel features
python train_audio.py --seed 0      # repeat for seeds 1 and 2
python evaluate_audio.py            # Table IV, per-class recall, siren share, Figs. 5-6
python benchmark.py                 # timing
```

## Contents

- `code/`: pipeline scripts and pinned package versions
- `results/`: tables, metrics for every seed, per-class recall, clip counts, siren share, timing, and
  per-clip validation/test scores for every seed (`per_seed/`)
- `figures/`: Fig. 5 (PR curve) and Fig. 6 (confusion matrix and ROC)
- `model/audiocnn_seed0.pt`: trained weights of the primary run (with normalization mean/std)
