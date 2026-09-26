"""Shared settings for the re-created AudioCNN pipeline (paper Sec. III-C, Table II)."""
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SR = 22050            # 22.05 kHz mono
DUR = 4.0             # 4-s pad/trim
N_SAMPLES = int(SR * DUR)
N_FFT = 1024
HOP = 512
N_MELS = 64           # 64-bin log-Mel

# Table V mapping: Bad = event of interest; every other class is Good.
BAD_US8K = {"gun_shot", "siren"}
BAD_ESC50 = {"fireworks", "glass_breaking", "hand_saw", "chainsaw", "siren"}

# Official folds (Table II): train / validation / test.
US8K_SPLIT = {**{f: "train" for f in range(1, 9)}, 9: "val", 10: "test"}
ESC50_SPLIT = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}

BAD_WEIGHT = 5.5      # Bad-class weight 5.5x


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class AudioCNN(nn.Module):
    """Four convolution blocks followed by a classifier (binary Good/Bad)."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def logmel_transform():
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    return lambda wav: db(mel(wav))


def load_clip(path):
    """Read, mono, resample to 22.05 kHz, pad/trim to 4 s. Returns float32 tensor [N_SAMPLES]."""
    import soundfile as sf
    import torchaudio.functional as AF
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(x.mean(axis=1))
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    if wav.numel() >= N_SAMPLES:
        wav = wav[:N_SAMPLES]
    else:
        wav = torch.nn.functional.pad(wav, (0, N_SAMPLES - wav.numel()))
    return wav
