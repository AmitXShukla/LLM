---
title: "Step 8 — Rebuild with the modern design"
short_title: "8. Modern architecture"
---

# Step 8 — Rebuild with the modern design

**Goal:** replace your Step 1 model with the architecture people actually use
today.

---

## Why this step matters

The transformer from the 2017 paper *Attention Is All You Need* is history, not
a target. Almost nothing in it survived unchanged.

You built it in [Step 1](01-build-transformer.md) so you would understand it.
Now build the current one, so that when you open a modern model's config file
in [Step 11](11-adapt-base-model.md), every line means something to you.

:::{warning} A real risk worth naming
If you spend three days building an exact GPT-2 replica and stop there, you
walk away with **outdated assumptions**. You will not have touched RoPE,
RMSNorm, SwiGLU, or grouped-query attention — and those are what you will
actually meet in every model you work with.

This chapter is the fix. Do not skip from Step 2 straight to Step 11.
:::

---

## What changed, and why

| Part | Then (2017) | Now | Why |
|---|---|---|---|
| Normalization | LayerNorm, after the block | **RMSNorm**, before the block | Simpler, faster, more stable |
| Position | Learned, absolute | **RoPE** | Handles long and flexible word order better |
| Activation | ReLU or GeLU | **SwiGLU** | Better quality for the same cost |
| Attention | Multi-head (MHA) | **Grouped-query (GQA)** | Much less memory when serving |
| Bias terms | Yes | No | They were not helping |

Sanskrit gives you a specific reason to care about RoPE: word order in Sanskrit
is unusually free, because the grammatical role is carried by the ending rather
than the position. A position scheme that encodes *distance between words*
rather than *absolute slot number* is a better match for that.

---

## What you do

### 1. Change one thing at a time

Measure after each change. This is the same discipline as
[Step 2](02-break-it.md) and for the same reason.

### 2. Swap LayerNorm for RMSNorm

LayerNorm subtracts the mean, then divides by the standard deviation. RMSNorm
skips the mean entirely and just divides by the root-mean-square.

It turns out the mean subtraction was not doing much. Removing it is faster and
slightly more stable.

Also confirm your normalization runs *before* the block, not after. You proved
why in Step 2.

### 3. Replace position embeddings with RoPE

RoPE stands for Rotary Position Embeddings.

Instead of adding a "position number" to each token, RoPE **rotates** the query
and key vectors by an angle based on their position.

The useful consequence: when the model compares a query at position 10 with a
key at position 3, the result depends on the gap between them, not on where
they sit in the document. The model naturally learns about *distance*.

This is also what makes context extension possible later, in
[Step 19](19-long-context-rag.md).

### 4. Replace the feed-forward activation with SwiGLU

SwiGLU adds a **gate**: a second path that decides how much of the first path
gets through. It costs a little more compute per parameter and reliably gives
better quality.

Because it uses three weight matrices instead of two, you usually shrink the
hidden size to keep the parameter count the same.

### 5. Switch to grouped-query attention

In standard multi-head attention, every head has its own keys and values. In
grouped-query attention, several query heads **share** one set of keys and
values.

Quality barely changes. The memory needed at serving time drops a lot, because
the thing you cache during generation is exactly those keys and values.

You will feel the benefit in [Step 18](18-quantization-serving.md). Set it up
now.

### 6. Remove every bias term

From the attention projections, from the feed-forward layers, from the
normalization. They were not earning their place.

### 7. Add QK-norm if training is unstable

Normalizing the queries and keys before the attention dot product stops the
values getting large, which lets you use a higher learning rate without the
loss spiking.

Add it only if you need it.

### 8. Write down the measured effect of each change

On *your* Sanskrit data. Do not trust the table above. Trust your own numbers.

---

## Where people usually get stuck

**Copying a modern architecture wholesale and never learning which piece did
what.**

You end up with a working model and no understanding, which is exactly the
situation this book exists to avoid.

---

## You are ready to move on when

You have a small table showing the loss before and after each individual
change, and you can open any modern model's `config.json` and explain what
every field means.

---

:::{seealso} Related
- [Step 1](01-build-transformer.md) — the old version
- [Step 11](11-adapt-base-model.md) — where you meet these in a real model
:::
