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
