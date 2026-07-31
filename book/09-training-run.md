---
title: "Step 9 — Run a real training job"
short_title: "9. Real training run"
---

# Step 9 — Run a real training job

**Goal:** train the biggest Sanskrit model your data and budget actually
justify.

---

## Why this step matters

Everything before this was on toy scale. Now you meet the real problems: memory
limits, speed, instability, and cost.

Based on [Step 6](06-collect-data.md), that is probably a model somewhere
between 100 and 500 million parameters. Be honest about the size your data
supports. A larger model trained on too little data is worse than a smaller one
trained properly, and it costs more.

---

## What you do

### 1. Use bf16 mixed precision

This stores most numbers in 16 bits instead of 32. It roughly halves memory and
speeds everything up.

Use **bf16**, not fp16. Both are 16-bit, but bf16 keeps a much wider range of
magnitudes at the cost of precision, and language model training cares far more
about range than precision. fp16 causes overflow problems that will waste your
time.

### 2. Use FlashAttention

FlashAttention computes exactly the same attention as before, but never writes
the huge attention matrix to memory. It works in tiles and keeps things in fast
on-chip memory.

Faster, far less memory, identical results. There is no downside. Turn it on.

### 3. Turn on gradient checkpointing if you run out of memory

Normally the model saves every intermediate value during the forward pass so it
can use them in the backward pass. Gradient checkpointing throws most of them
away and recomputes them when needed.

You trade about 30 percent extra time for a large memory saving. Worth it
whenever memory is your limit.

### 4. Use gradient accumulation

To get a large effective batch size on a small GPU, run several small batches,
add up their gradients, and only update the weights once at the end.

Four batches of 8 with accumulation behaves almost like one batch of 32.

### 5. Choose your optimizer

**AdamW** is the safe default and always works. Start here.

Then look at **Muon**, which came out of the open nanoGPT speedrun community.
Before applying an update, it adjusts the *shape* of the update matrix so the
change is spread more evenly across directions. For small and medium models it
trains noticeably faster than AdamW.

Understand roughly why it works before you use it. The speedrun repository's
history is a good place to see the effect measured.

### 6. Choose a learning rate schedule

**Cosine decay** is standard: warm up, then smoothly decay to near zero.

**Warmup-Stable-Decay (WSD)** is worth knowing: warm up, hold steady for a long
time, then decay only at the end. The advantage is that you can branch a new run
off any point in the stable phase without redoing the warmup — very useful when
you are experimenting.

### 7. Scale up in stages

1. **One GPU.** Get it working.
2. **DDP** — copy the model to each GPU, split the data. Simple.
3. **FSDP** — split the model itself across GPUs. Use when the model no longer
   fits on one.
4. **Tensor and pipeline parallel** — most readers never need these.

### 8. Watch your gradient norm

This is your best early warning signal. It usually starts climbing before the
loss does anything visible on the chart.

If the gradient norm spikes, you have a few hundred steps to react before the
loss follows.

### 9. Save checkpoints often

When a run goes bad, you roll back to the last good checkpoint, lower the
learning rate, and continue.

Everyone does this. It is normal. It is not a sign of failure.

### 10. Watch for the two failure modes you already know

From [Step 2](02-break-it.md):

- **Loss spike** — you saw this with the learning rate too high.
- **Overfitting** — training loss down, validation loss up. Very likely here,
  because your Sanskrit data is small.

You have seen both before. You will recognise them.

---

## Where people usually get stuck

**Launching one giant expensive run without a small test run first.**

Always do a short run at one-hundredth the scale to check the whole pipeline
end to end: data loading, training, checkpointing, evaluation, logging. Then
scale up.

---

## You are ready to move on when

You have a finished training run, a saved checkpoint, and a loss curve you can
explain to someone else.

---

:::{seealso} Related
- [Step 2](02-break-it.md) — the failures you are now recognising
- [Hardware appendix](appendix/hardware.md) — what your machine can handle
:::

---

## 🧑‍💻 Runnable code for this step

:::{tip} The full, tested file
[`code/step-01-tiny-transformer/train_sanskrit_gpt.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-01-tiny-transformer) is the complete ~250-line model + training loop + sampler. Run `python train_sanskrit_gpt.py --smoke` for a 30-second sanity run.
:::

The entire training loop is four lines repeated a few thousand times. If you
understand these four, you understand training:

```python
for step in range(max_iters):
    xb, yb = get_batch(train_data)   # a window of aksharas, and the same shifted by 1
    _, loss = model(xb, yb)          # forward: predict, measure surprise (cross-entropy)
    optimizer.zero_grad()            # clear last step's gradients
    loss.backward()                  # backward: blame each weight for the error
    optimizer.step()                 # nudge every weight a little the right way
```

```{mermaid}
flowchart LR
    A[batch of aksharas] --> B[model predicts<br/>next akshara]
    B --> C[loss = how surprised?]
    C --> D[loss.backward<br/>compute gradients]
    D --> E[optimizer.step<br/>nudge weights]
    E --> A
```

:::{note} What to watch 👀
**Train loss** should fall then flatten. **Val loss** should follow, then turn
*up* when you start overfitting. On a tiny corpus that happens fast — that's not
a bug, it's the model telling you *"feed me more data."*
:::

:::{seealso} Go deeper
- 🧠 Teaching notes (every concept, mapped to the code): [`docs/notes/weekend1-tiny-transformer-teaching.md`](https://github.com/AmitXShukla/LLM/tree/main/docs/notes/weekend1-tiny-transformer-teaching.md)
:::
