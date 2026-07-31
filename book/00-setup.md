---
title: "Step 0 — Get your workspace ready"
short_title: "0. Setup"
---

# Step 0 — Get your workspace ready

**Goal:** have a machine, a working environment, and a way to record your
experiments.

---

## Why this step matters

You are about to run hundreds of experiments. If you do not record them, you
will not remember which settings gave which result.

This sounds like advice you can safely ignore. It is not. People skip this step
and regret it in week two, when they have sixty training runs, no notes, and no
idea which one produced the good model.

---

## What you do

### 1. Pick your hardware

Start with what you have. Move up only when a step actually blocks you.

| What you have | How far you get |
|---|---|
| Free Colab or Kaggle GPU | Steps 0 to 5 |
| One 24 GB GPU (3090, 4090) | Steps 0 to 19 |
| DGX Spark, 128 GB unified memory | Everything, comfortably |
| Rented A100 or H100 by the hour | Everything |

Do not buy a GPU yet. Rent by the hour first and find out what you actually
need. A strong single GPU costs roughly 1 to 3 US dollars per hour on the
common cloud providers. If you are careful to shut instances down, the whole
book costs somewhere between 200 and 600 dollars in rented compute.

If you already own a workstation, read the
[hardware appendix](appendix/hardware.md) before you start. It explains where a
DGX Spark is excellent, where it is merely fine, and one thing it is genuinely
poor at.

### 2. Install the tools

You need PyTorch, plus these libraries:

- `transformers` — model loading and standard architectures
- `datasets` — data loading and processing
- `tokenizers` — building tokenizers
- `accelerate` — multi-GPU and mixed precision handling
- `peft` — LoRA and other efficient fine-tuning methods
- `trl` — instruction tuning and preference training
- `vllm` — fast serving, later in the book

### 3. Write down your exact versions today

Create a lock file now, before anything works, and update it whenever things
break.

```bash
pip freeze > requirements.lock.txt
```

Version drift causes about half of all "it worked yesterday" problems in this
field. Thirty seconds now saves a whole afternoon later.

### 4. Set up experiment tracking

Use Weights and Biases, Trackio, or even a plain CSV file. Anything is better
than nothing.

For every run, log at minimum:

- The loss, over time
- The learning rate
- The full configuration
- The git commit hash

This is the single highest-value thirty minutes in the whole book.

### 5. Check that Sanskrit and Urdu display correctly

Print one line of each in your terminal and open a file containing both in your
editor.

```python
print("तत् त्वम् असि")     # Sanskrit, Devanagari
print("اردو ایک زبان ہے")   # Urdu, Perso-Arabic
```

If you see boxes, question marks, or blank space, you have a font problem. Fix
it now. Debugging a tokenizer while you cannot read your own data is
unnecessarily painful.

### 6. Make a project folder

```
my-sanskrit-llm/
├── data/          # raw and cleaned text (never commit this)
├── tokenizers/    # tokenizers you train
├── models/        # checkpoints (never commit these either)
├── notebooks/     # exploration
├── src/           # real code
└── notes.md       # what you tried and what happened
```

The `notes.md` file matters more than it looks. One line per experiment, in
your own words. Future you will be grateful.

---

## Where people usually get stuck

Skipping the tracking setup, then having sixty unnamed runs with no record of
what changed between them.

The second most common problem is font rendering. Do not put it off. Everything
in Part 2 involves staring at text.

---

## You are ready to move on when

You can run a one-line PyTorch script on your GPU, see the run appear in your
tracking dashboard, and print a Sanskrit line that displays correctly.

---

:::{seealso} Related
- [Hardware appendix](appendix/hardware.md) — what each tier can do, and DGX Spark notes
- [Step 1](01-build-transformer.md) — your first model
:::
