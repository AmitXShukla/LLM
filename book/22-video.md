---
title: "Step 22 — Video"
short_title: "22. Video"
---

# Step 22 — Video

:::{note} Chapter status
Outline. To be expanded. Consider skipping this one entirely.
:::

**Goal:** understand what makes video hard.

---

## Why this step matters

Video is images plus time, which mostly means video is images plus an enormous
number of tokens.

Everything you learned still applies. The problem is purely scale.

---

## What to cover

1. **Frame sampling.** You cannot use every frame. Deciding which ones to keep
   is most of the engineering.
2. **Temporal tokenization.** Turning a sequence of frames into tokens without
   producing a million of them.
3. **The context length problem.** This connects directly to
   [Step 19](19-long-context-rag.md).
4. **Video-language models.** The same projector pattern as
   [Step 21](21-vision.md), with a time dimension added.

---

## An honest note

This is expensive and it is the least relevant part of the book for a Sanskrit
or Urdu project.

Possible uses: recitation videos where mouth position matters, teaching
material, ritual documentation. Genuine, but narrow.

Do it last, if at all.

---

## 🧑‍💻 Build a video behaviour model

Video is just **vision + time**. The simplest strong approach: run a per-frame CNN
to turn each frame into a feature vector, then a small **Transformer over the
sequence** of frame features to read the temporal pattern. Full runnable file
(with synthetic clips so it runs today): [`code/step-22-video/video_behavior.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-22-video).

```{mermaid}
flowchart LR
    A[🎞️ frames] --> B[per-frame CNN<br/>ResNet18]
    B --> C[feature per frame]
    C --> D[⏱️ temporal Transformer]
    D --> E[pool over time]
    E --> F[behaviour class]
```

```python
import torch, torch.nn as nn, torchvision

class VideoBehaviorNet(nn.Module):
    """Per-frame CNN features -> temporal Transformer -> behaviour class."""
    def __init__(self, n_classes=4, d=512, n_heads=8, n_layers=2):
        super().__init__()
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.feat = nn.Sequential(*list(backbone.children())[:-1])   # feature per frame
        enc = nn.TransformerEncoderLayer(d, n_heads, batch_first=True)
        self.temporal = nn.TransformerEncoder(enc, n_layers)
        self.cls = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))
    def forward(self, x):                      # x: (B, T, C, H, W)
        B, T = x.shape[:2]
        f = self.feat(x.flatten(0, 1)).flatten(1).view(B, T, -1)     # (B, T, 512)
        return self.cls(self.temporal(f).mean(dim=1))                # pool over time
```

The three ways to model video, in order of cost: **(1)** frame features + a
temporal model (above — simple, strong, cheap); **(2)** 3D CNNs / video
transformers (VideoMAE, SlowFast — higher ceiling); **(3)** pose-based (extract
keypoints first, model the pose sequence).

:::{important} 🕶️ Privacy first for patient video
Faces are PHI-adjacent. For behaviour tasks (falls, gait, agitation), prefer the
**pose-based** route: pose estimation → temporal model on keypoints. It's
accurate, cheaper, and stores **no faces** — an ethics *and* a cost win. Proposing
this in an interview shows you weigh privacy, not just accuracy.
:::

:::{note} 🏷️ The real bottleneck is labelling
Temporal annotation (when does the event start and stop?) is expensive. Sample
frames (you rarely need 30 fps), start from a video-pretrained backbone, and lean
on weak/self-supervised pretraining where you can.
:::
