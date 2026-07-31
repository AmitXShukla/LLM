"""
video_behavior.py — patient video -> behaviour class
====================================================
Video is just vision + TIME. The simplest strong approach: run a per-frame CNN to
turn each frame into a feature vector, then a small Transformer over the SEQUENCE
of frame features to read the temporal pattern (a fall, a seizure, agitation).

This is again the encoder+head recipe — the encoder now spans space AND time.

Runs out of the box on SYNTHETIC clips (--smoke). For real use, replace
`synthetic_clips` with a loader that samples frames from your videos.

    python video_behavior.py --smoke     # random clips, offline, ~15s

Privacy note (important for patient video): for behaviour tasks, prefer a
POSE-BASED pipeline — extract skeletal keypoints first, then model the pose
sequence. It's accurate, cheaper, and stores no faces.
"""

import argparse
import torch
import torch.nn as nn
import torchvision


class VideoBehaviorNet(nn.Module):
    """Per-frame CNN features -> temporal Transformer -> behaviour class."""

    def __init__(self, n_classes=4, d=512, n_heads=8, n_layers=2, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        self.feat = nn.Sequential(*list(backbone.children())[:-1])   # -> (B,512,1,1) per frame
        enc = nn.TransformerEncoderLayer(d, n_heads, batch_first=True)
        self.temporal = nn.TransformerEncoder(enc, n_layers)
        self.cls = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x):                      # x: (B, T, C, H, W)
        B, T = x.shape[:2]
        f = self.feat(x.flatten(0, 1)).flatten(1)     # (B*T, 512)
        f = f.view(B, T, -1)                           # (B, T, 512)
        h = self.temporal(f).mean(dim=1)               # temporal pooling -> (B, 512)
        return self.cls(h)


def synthetic_clips(n=8, frames=8, size=112, n_classes=4):
    """Fake clips: each class has a different moving bright blob trajectory."""
    x = torch.rand(n, frames, 3, size, size) * 0.4
    y = torch.randint(0, n_classes, (n,))
    for i in range(n):
        for t in range(frames):
            p = int((t / frames) * (size - 10)) if y[i] % 2 == 0 else int((1 - t / frames) * (size - 10))
            x[i, t, :, p:p + 8, p:p + 8] += 0.6        # a blob moving by class rule
    return x.clamp(0, 1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VideoBehaviorNet(n_classes=4, pretrained=not args.smoke).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    print(f"device={device}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    model.train()
    steps = 15 if args.smoke else 100
    for step in range(steps):
        xb, yb = synthetic_clips()
        opt.zero_grad()
        loss = criterion(model(xb.to(device)), yb.to(device))
        loss.backward(); opt.step()
        if step % 5 == 0:
            print(f"  step {step:>3}: loss={loss.item():.3f}")

    print("\nReal pipelines: sample frames (you rarely need 30 fps), start from a")
    print("video-pretrained backbone (VideoMAE / SlowFast), and prefer pose-only")
    print("features for patient privacy.")


if __name__ == "__main__":
    main()
