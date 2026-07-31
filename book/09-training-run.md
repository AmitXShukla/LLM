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

## 🧑‍💻 The training loop, in full

The whole of training is four lines repeated a few thousand times. If you
understand these four, you understand how *every* neural network is trained. Full
file: [`code/step-01-tiny-transformer/train_sanskrit_gpt.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-01-tiny-transformer).

```python
for it in range(cfg.max_iters + 1):
    if it % cfg.eval_interval == 0:
        losses = estimate_loss(model, splits, cfg, device)   # check train & val loss
        print(f"iter {it:>5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch(splits["train"], cfg, device)         # a batch of aksharas
    _, loss = model(xb, yb)                                   # forward: predict, measure surprise
    optimizer.zero_grad(set_to_none=True)                    # clear last step's gradients
    loss.backward()                                          # backward: blame each weight
    optimizer.step()                                         # nudge every weight the right way
```

```{mermaid}
flowchart LR
    A[📥 batch of aksharas] --> B[🔮 model predicts next]
    B --> C[😲 loss = how surprised?]
    C --> D[⬅️ loss.backward: gradients]
    D --> E[🎚️ optimizer.step: nudge weights]
    E --> A
```

`loss.backward()` is autograd computing the gradient of the loss with respect to
every parameter (the chain rule, automated). `AdamW` turns those gradients into
smart weight updates. That's it — that's "training a neural net."

### 🧪 Measuring honestly (`estimate_loss`)

We periodically check a held-out **val** split. If train loss keeps dropping while
val loss climbs, you're *overfitting* — memorising the corpus instead of learning
the language.

```python
@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out
```

### 🚀 Putting it together (`main`)

```python
text = Path(args.corpus).read_text(encoding="utf-8")
tok  = build_tokenizer(text, args.tokenizer)             # grapheme by default (Step 4)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
splits = {"train": data[:n], "val": data[n:]}            # 90/10 split

model = SanskritGPT(cfg, tok.vocab_size).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
# ... the training loop above ...

# then sample from the trained model:
start = torch.tensor([tok.encode("विद्या")], dtype=torch.long, device=device)
out = model.generate(start, max_new_tokens=200, temperature=0.8, top_k=20)
print(tok.decode(out[0].tolist()))
```

:::{note} 👀 What healthy training looks like
```
iter     0: train loss 4.7521, val loss 4.7503
iter   500: train loss 2.4133, val loss 2.98
iter  3000: train loss 1.31xx, val loss 2.6xx   ← val flattening/rising = overfitting on a tiny corpus
```
On 20 verses it overfits almost instantly — and that is the lesson, not a bug.
**The model is data-starved, not brain-starved.** The fix is more data, which is
what [Steps 6–7](06-collect-data.md) are all about.
:::

:::{important} ✅ Read the output charitably
Early samples are gibberish — but *valid* gibberish: every "word" is built from
whole aksharas, no orphan vowel signs, because of the grapheme tokenizer from
[Step 4](04-sanskrit-tokenizer.md). Nonsense-but-valid means the pipeline works;
it just needs more data to become nonsense-that-means-something.
:::
