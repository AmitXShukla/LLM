---
title: "Step 21 — Images and manuscript OCR"
short_title: "21. Images and OCR"
---

# Step 21 — Images and manuscript OCR

:::{note} Chapter status
Outline. To be expanded.
:::

**Goal:** connect a vision encoder to your language model, and use it to read
manuscripts.

---

## Why this step matters

**The killer application here is OCR**, and it feeds directly back into
[Step 6](06-collect-data.md).

There are millions of unscanned or badly-scanned pages of Sanskrit and Urdu.
The open tools for reading them are genuinely weak. Better OCR means more data,
which means a better model, which means better OCR.

Large digitisation projects are working on exactly this problem. Their output
is often openly published, which means you can build on it rather than
duplicating it.

---

## What to cover

### The basic pattern

It is simpler than people expect and has three parts:

1. A **vision encoder** turns an image into a sequence of vectors.
2. A small **projector** — usually just a two-layer network — maps those
   vectors into your language model's space.
3. Your **language model** reads those vectors exactly as if they were text
   tokens.

That is the whole architecture. The projector is the only new part.

### Training, in two stages

1. Freeze both large models. Train **only** the projector, so the two learn to
   speak a common language. Cheap.
2. Unfreeze and instruction-tune together.

### For manuscripts specifically

- Handwritten Devanagari, and the many regional and historical letter forms
- Nastaliq script for Urdu, which is genuinely hard because letters overlap
  and cascade
- Palm leaf manuscripts, damage, staining, and faded ink
- Layout: marginal notes, interlinear commentary, multiple hands on one page
- Output should be structured, not a flat wall of text

### Realistic scope

Building a small vision-language model on top of a small language model plus an
off-the-shelf vision encoder is now a realistic weekend project. Building a
production OCR system for damaged palm leaf manuscripts is not. Scope
accordingly.

---

:::{seealso} Related
- [Step 6](06-collect-data.md) — OCR is how you get more data
- [Step 7](07-clean-data.md) — cleaning up OCR errors
:::

---

## 🧑‍💻 The universal recipe applies here too

Whether it's a manuscript image for OCR or a medical X-ray, the pattern is the
same: **encoder → head**, using transfer learning.

```{mermaid}
flowchart LR
    A[🖼️ image] --> B[vision encoder<br/>ResNet / ViT]
    B --> C[embeddings]
    C --> D["head or projector"]
    D --> E[label / text]
```

For **manuscript OCR**, the transfer-learning spectrum is your friend: start from
a pretrained vision (or vision-language) model and fine-tune on your script — you
need far less labelled data than training from scratch. The same encoder that
reads an X-ray reads a page of Devanagari; only the head and the training data
change.

:::{tip} Reuse the medical vision code
The transfer-learning + LoRA pattern for images is shown with runnable code in
[Step 24 — Medical](24-medical.md) (🦴 X-ray section). It transfers directly to
manuscript classification and OCR fine-tuning.
:::
