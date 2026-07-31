---
title: Hardware notes
short_title: Hardware
---

# Hardware notes

What each tier of machine can and cannot do, and what things cost.

---

## The tiers

| Tier | Example | Gets you through |
|---|---|---|
| Free | Colab, Kaggle T4 | Steps 0 to 5 |
| Consumer | One 24 to 32 GB GPU | Steps 0 to 19 |
| Workstation | DGX Spark, 128 GB unified | Everything, comfortably |
| Rented | A100 or H100 by the hour | Everything |
| Cluster | 8 GPU node | Everything, fast |

---

## Renting

Rent before you buy. You will learn what you actually need, which is almost
never what you first assumed.

Rough costs: a strong single GPU runs about 1 to 3 US dollars per hour on the
common providers. Spot and preemptible instances are cheaper if your job can
survive being interrupted — and with checkpointing from
[Step 9](09-training-run.md), it can.

**Budget for the whole book on rented compute: roughly 200 to 600 dollars**, if
you are disciplined about shutting instances down. Set a billing alert on day
one.

---

## Notes for the NVIDIA DGX Spark

The DGX Spark is a desktop machine built around the GB10 Grace Blackwell
superchip, with 128 GB of unified memory shared between CPU and GPU, and very
high theoretical throughput at low precision.

It is an unusually good fit for the path this book takes, with one real
weakness.

### Where it is excellent

**Anything limited by memory capacity rather than memory speed.** 128 GB of
unified memory means you can load models that simply will not fit on a 24 GB
consumer card, and you can do it without splitting the model across devices.

**Continued pretraining and QLoRA fine-tuning** — [Step 11](11-adapt-base-model.md)
and [Step 12](12-instruction-tuning.md). This is the machine's sweet spot. A 7B
to 14B model with QLoRA runs comfortably, and tools like Unsloth roughly double
the speed and halve the memory again.

**Local inference of large models.** You can serve something substantial without
a cloud, which matters if privacy is part of your project's point.

**Iterating without a meter running.** The value of not watching a per-hour bill
while you experiment is easy to underestimate.

### Where it is merely fine

Steps 1 through 10. Your toy model does not need this machine, and you should
not wait for one to start.

### Where it is genuinely weak

:::{warning} Memory bandwidth
Unified memory is large but its **bandwidth** is lower than a high-end discrete
GPU's dedicated memory.

Full-precision pretraining from scratch is bandwidth-hungry, so it will run
noticeably slower here than on a multi-GPU workstation.

This matters less than it sounds, because [Step 6](06-collect-data.md) already
told you that from-scratch pretraining is not your path. The machine is weak at
precisely the thing you should not be doing.
:::

### A realistic plan on a DGX Spark

| Stage | Time |
|---|---|
| Data audit and cleaning (Steps 6 to 7) | Days |
| Tokenizer work (Steps 3 to 5) | Days |
| Vocabulary extension plus QLoRA continued pretraining on a 7B model | Hours to a few days |
| Instruction tuning (Step 12) | Hours |
| Evaluation and a basic chat interface | Days |
| **Full working prototype** | **1 to 3 weeks of focused work** |

That is a realistic schedule for one person. It is also why this project is a
"few weeks" project rather than a "years of cluster time" project.

---

## What to check before you commit

1. Does your framework support your hardware properly? Check before buying, not
   after.
2. Does the specific quantization method you want work on it?
3. Can you actually cool it where you plan to put it?
4. Is your electricity supply adequate?

---

:::{seealso} Related
- [Step 0](../00-setup.md) — setting up
- [Step 9](../09-training-run.md) — where memory and bandwidth bite
:::
