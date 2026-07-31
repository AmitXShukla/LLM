"""
ecg_arrhythmia.py — heartbeat -> arrhythmia, the same encoder+head recipe
=========================================================================
A 1D convolutional network (ResNet-style) that classifies 12-lead ECG windows
into rhythm classes. This is the canonical 1D biomedical-signal task, and it
teaches the two things that matter most in clinical ML:

  1. Signals need PREPROCESSING (normalize per lead; real projects also bandpass).
  2. The dangerous class is RARE, so accuracy lies — you weight the loss by class
     frequency and evaluate with per-class recall / AUPRC, never plain accuracy.

Runs out of the box on SYNTHETIC ECG so you can see the whole pipeline today.
Swap `make_synthetic_ecg` for a PTB-XL / MIT-BIH loader for the real thing.

    python ecg_arrhythmia.py            # full-ish run on synthetic data
    python ecg_arrhythmia.py --smoke    # a few steps, ~10 seconds
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

N_LEADS = 12
N_CLASSES = 5          # e.g. Normal, AFib, PVC, LBBB, Other
FS = 250               # samples per second
SECONDS = 4
T = FS * SECONDS       # window length


# ---------------------------------------------------------------------------
# 0. Data. Real datasets: PTB-XL (~21k 12-lead recordings), MIT-BIH. Here we
#    synthesize class-specific rhythms so the file runs with no download.
#    The classes are DELIBERATELY IMBALANCED to mirror clinical reality.
# ---------------------------------------------------------------------------
def make_synthetic_ecg(n_per_class=(4000, 250, 600, 150, 500), seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    t = np.linspace(0, SECONDS, T, endpoint=False)
    for cls, n in enumerate(n_per_class):
        for _ in range(n):
            hr = rng.uniform(60, 100) if cls == 0 else rng.uniform(50, 160)
            base_freq = hr / 60.0
            sig = np.zeros((N_LEADS, T), dtype=np.float32)
            for lead in range(N_LEADS):
                phase = rng.uniform(0, 2 * np.pi)
                wave = np.sin(2 * np.pi * base_freq * t + phase)
                wave += 0.3 * np.sin(2 * np.pi * 2 * base_freq * t)  # QRS-ish harmonic
                if cls == 1:                       # AFib: irregular, add jitter
                    wave += 0.5 * rng.standard_normal(T) * (rng.random(T) < 0.1)
                if cls == 2:                       # PVC: occasional big spikes
                    spikes = rng.random(T) < 0.01
                    wave += 3.0 * spikes
                wave += 0.05 * rng.standard_normal(T)               # baseline noise
                sig[lead] = wave
            xs.append(sig); ys.append(cls)
    x = np.stack(xs); y = np.array(ys, dtype=np.int64)
    idx = rng.permutation(len(y))                  # shuffle
    return x[idx], y[idx]


def preprocess(x):
    """Per-lead z-score. (Real pipelines also bandpass 0.5-40 Hz with scipy and
    resample to a common rate — see the book chapter.)"""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return (x - mean) / std


# ---------------------------------------------------------------------------
# 1. The model: a small 1D ResNet. Conv1d slides over TIME instead of space.
# ---------------------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 7, stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 7, 1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.down = (nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False),
                                   nn.BatchNorm1d(c_out))
                     if (stride != 1 or c_in != c_out) else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)


class ECGResNet(nn.Module):
    def __init__(self, n_leads=N_LEADS, n_classes=N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_leads, 64, 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResBlock1D(64, 64), ResBlock1D(64, 128, stride=2),
            ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256, stride=2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):                    # x: (batch, leads, samples)
        return self.head(self.layers(self.stem(x)))


# ---------------------------------------------------------------------------
# 2. Train + evaluate. Note the class-weighted loss and the recall-based eval.
# ---------------------------------------------------------------------------
def per_class_recall(model, loader, device, n_classes=N_CLASSES):
    model.eval()
    correct = torch.zeros(n_classes); total = torch.zeros(n_classes)
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            for c in range(n_classes):
                m = yb == c
                total[c] += m.sum()
                correct[c] += (pred[m] == c).sum()
    model.train()
    return (correct / total.clamp(min=1)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    counts = (400, 30, 60, 20, 50) if args.smoke else (4000, 250, 600, 150, 500)
    epochs = 1 if args.smoke else 8

    x, y = make_synthetic_ecg(counts)
    x = preprocess(x)
    n = int(0.8 * len(y))
    tr = TensorDataset(torch.tensor(x[:n]), torch.tensor(y[:n]))
    va = TensorDataset(torch.tensor(x[n:]), torch.tensor(y[n:]))
    tl = DataLoader(tr, batch_size=64, shuffle=True)
    vl = DataLoader(va, batch_size=128)

    # inverse-frequency class weights so the rare arrhythmias are not ignored
    cls_counts = torch.tensor(counts, dtype=torch.float)
    weights = cls_counts.sum() / (len(cls_counts) * cls_counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    model = ECGResNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"device={device}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"class weights={[round(w,2) for w in weights.tolist()]}")

    for ep in range(epochs):
        for xb, yb in tl:
            opt.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward(); opt.step()
        rec = per_class_recall(model, vl, device)
        print(f"epoch {ep}: loss={loss.item():.3f}  per-class recall="
              f"{[round(r,2) for r in rec]}  macro={sum(rec)/len(rec):.2f}")

    print("\nWhy recall, not accuracy? A model that always predicts class 0 would")
    print("score ~80% accuracy here and MISS EVERY ARRHYTHMIA. Recall exposes that.")


if __name__ == "__main__":
    main()
