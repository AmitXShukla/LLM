---
title: "Step 24 — Medical images and heart sounds"
short_title: "24. Medical data"
---

# Step 24 — Medical images and heart sounds

:::{note} Chapter status
Outline. To be expanded.
:::

**Goal:** see how much of this book transfers to a completely different domain.

---

## Why this step matters

This chapter exists to prove a claim made in the [introduction](intro.md):
almost everything you learned here transfers.

Medical imaging and biosignals are classic **low-resource domains**, with the
same shape of problem as Sanskrit: scarce high-quality labelled data, expensive
expert annotation, no good general benchmarks, and domain shift between
sources.

If you can build a Sanskrit model, you can build a chest X-ray classifier. The
thinking is roughly 70 to 80 percent the same.

---

## What transfers directly

**The data-centric mindset.** Cleaning, deduplication, careful splits, handling
imbalance, expert validation. Medical data has the same problems as low-resource
text, often worse.

**Adaptation strategy.** Start from a strong pretrained backbone. Do continued
pretraining on your domain. Use LoRA or QLoRA for the fine-tuning. This
transfers almost exactly.

**Training fundamentals.** Optimizers, learning rate schedules, reading loss
curves, spotting overfitting. Identical.

**Efficiency work.** Mixed precision, gradient checkpointing, quantization,
local serving. Identical.

**Evaluation discipline.** Held-out sets, careful metric choice, expert review.
Even more important here, because a confident wrong answer has consequences.

---

## What changes

| Aspect | Text | Images (X-ray, CT) | Audio (heart sounds, ECG) |
|---|---|---|---|
| Input | Subword tokens | Image patches | Spectrograms or waveforms |
| Architecture | Decoder-only transformer | Vision transformer or ConvNeXt | Audio spectrogram transformer |
| Pretraining task | Predict the next token | Masked image modelling, or contrastive | Masked spectrogram, or contrastive |
| Fine-tuning | Vocabulary extension, continued pretraining | Backbone plus a task head | Feature extraction or end-to-end |
| Evaluation | Perplexity plus expert review | Clinical metrics plus radiologist review | Clinical metrics plus cardiologist review |

**The biggest conceptual difference:** there is no direct equivalent of
tokenization. But you still make critical preprocessing choices — patch size
and resolution for images, spectrogram parameters for audio — and those choices
have the same kind of downstream importance that tokenization does. The lesson
of [Step 3](03-tokenizers.md) transfers even though the mechanism does not.

---

## A practical starting path

1. **Start from existing strong backbones.** Do not build from scratch again
   unless you want the learning experience a second time.
   - Images: MONAI is the standard medical imaging library, plus vision
     transformer backbones or a medical foundation model
   - Audio: torchaudio, plus an audio spectrogram transformer or a
     self-supervised speech backbone

2. **Use the same efficient fine-tuning toolkit** you used in
   [Step 11](11-adapt-base-model.md).

3. **Spend most of your time on data preparation and expert validation.** This
   will feel very familiar.

4. **For multimodal work** — X-ray plus radiology report — the pattern is the
   same as [Step 21](21-vision.md).

---

:::{warning} A different kind of responsibility
Medical models can hurt people. Regulatory requirements, clinical validation,
and bias across patient populations are not optional extras.

Nothing in this book qualifies you to deploy a clinical tool. It qualifies you
to build a research prototype and to talk sensibly with the people who can.
:::

---

:::{seealso} Related
- [Step 11](11-adapt-base-model.md) — the same adaptation pattern
- [Step 21](21-vision.md) — the vision-language pattern
:::

---

## 🧑‍💻 The universal recipe: encoder → head

Here is the idea that makes ECG, X-ray, and video feel like *one* subject instead
of three: **almost every model is an encoder that turns raw input into vectors,
then a head (or decoder) that produces the output.** Fine-tuning is just "adapt
part of that stack to my data" — the exact same transfer-learning spectrum you met
with LoRA in [Step 11](11-adapt-base-model.md).

```{mermaid}
flowchart LR
    A[raw input<br/>❤️ ECG · 🦴 X-ray · 🎥 video] --> B[ENCODER<br/>1D-CNN / ResNet / ViT]
    B --> C[embeddings]
    C --> D[HEAD<br/>classifier or decoder]
    D --> E[label / report]
```

:::{note} 🎓 Same skills, new encoder
Everything from the language chapters carries over: transfer learning, the
freeze-vs-LoRA-vs-full choice, and the imbalance discipline. Only the encoder and
the input preprocessing change. If you can fine-tune an LLM, you can fine-tune
these.
:::

---

## ❤️ Heartbeat → arrhythmia (a 1D signal)

An ECG is a time series of voltages, usually 12 leads. Real datasets: **PTB-XL**
(~21k clinical 12-lead recordings) and **MIT-BIH**. The encoder is a **1D CNN** —
`Conv1d` slides over *time* instead of space. Full runnable file (with synthetic
data so it runs today): [`code/step-24-medical-ecg/ecg_arrhythmia.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-24-medical-ecg).

### 🧱 A 1D ResNet block

```python
import torch.nn as nn

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
        return self.act(x + r)      # residual — same trick as the transformer block
```

### ⚖️ The clinical move that matters most: imbalance

Dangerous arrhythmias are **rare**. A model that always says "normal" can score
98% accuracy and catch *zero* arrhythmias. So we weight the loss by inverse class
frequency and evaluate with **per-class recall / AUPRC**, never accuracy:

```python
import torch, torch.nn as nn
counts  = torch.tensor([4000., 250., 600., 150., 500.])       # very imbalanced
weights = counts.sum() / (len(counts) * counts)               # up-weight rare classes
criterion = nn.CrossEntropyLoss(weight=weights)               # evaluate with recall, not accuracy
```

:::{warning} 🚨 Accuracy is a trap in healthcare
This is the single most common mistake in clinical ML interviews and projects.
Whenever the important class is rare, report **recall, F1, AUROC, and especially
AUPRC** — and pick your decision threshold by the clinical cost of a miss.
:::

---

## 🦴 X-ray → fracture / findings (a 2D image)

Datasets: **ChestX-ray14**, **CheXpert**, **MURA**. The recipe is *transfer
learning*: take a pretrained backbone (or a **medical** one like MedSigLIP),
replace the head, fine-tune. Full runnable file:
[`code/step-24-medical-xray/xray_finetune.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-24-medical-xray).

```python
import torch, torch.nn as nn, torchvision

def build_model(n_classes=2, pretrained=True, freeze_backbone=False):
    m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    if freeze_backbone:                        # "linear probing": train only the head
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_classes)   # new head: fracture / normal
    return m

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))  # up-weight rare "fracture"
```

:::{caution} 🩻 Watch for "shortcut learning"
Medical models love spurious shortcuts — a scanner artifact, a laterality token, a
hospital marker burned into the image. Always inspect **Grad-CAM** saliency maps
(does the model look at the *anatomy*?) and evaluate on **external** data from a
different site before believing your numbers. And never horizontally flip a chest
X-ray — it puts the heart on the wrong side.
:::

:::{tip} 💊 Small data? Reach for a medical foundation model
Clinical datasets are small, so PEFT shines here just like LoRA did for text.
Start from **MedSigLIP** (the ~400M encoder behind MedGemma) or TorchXRayVision
and LoRA-fine-tune — you need far less data and get better calibration.
:::

---

## 🖼️➡️📝 Generative: fine-tuning a medical VLM (MedGemma)

A **vision-language model** reads an image *and* a question and writes *text* — a
radiology report, a visual answer. It's the encoder→projector→LLM pattern:
**MedGemma** is Gemma 3 with **MedSigLIP** as its eyes. You fine-tune it on
*(image, prompt, target-text)* triples — mechanically it's LoRA SFT
([Step 11](11-adapt-base-model.md)) with the image passed through the processor.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

proc  = AutoProcessor.from_pretrained("google/medgemma-4b-it")
model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-4b-it", dtype=torch.bfloat16, device_map="auto")
model = get_peft_model(model, LoraConfig(     # keep the heavy vision tower frozen
    r=16, lora_alpha=32, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
```

:::{important} 🩺 Two failure modes, two evaluations
A **classifier** (ECG/X-ray) gives a calibrated number for triage → judge it with
**AUPRC + calibration**. A **VLM** drafts a report a clinician edits → judge it
with **faithfulness / hallucination** checks and clinical-entity metrics like
**RadGraph F1**, not fluency. Use each where its failure mode is acceptable, and
keep a human in the loop.
:::

:::{seealso} 📚 Follow-along resources
- ❤️ Runnable ECG model: [`code/step-24-medical-ecg/`](https://github.com/AmitXShukla/LLM/tree/main/code/step-24-medical-ecg)
- 🦴 Runnable X-ray model: [`code/step-24-medical-xray/`](https://github.com/AmitXShukla/LLM/tree/main/code/step-24-medical-xray)
- 📘 Deep dive (Part III & IV of the 58-page PDF): [`docs/reports/fine-tuning-foundation-models.pdf`](https://github.com/AmitXShukla/LLM/tree/main/docs/reports/fine-tuning-foundation-models.pdf)
:::
