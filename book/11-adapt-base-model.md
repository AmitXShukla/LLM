---
title: "Step 11 — Adapt a large open model"
short_title: "11. Adapt a base model"
---

# Step 11 — Adapt a large open model

**Goal:** get a genuinely useful Sanskrit model by starting from someone else's
trained model.

---

## Why this step matters

In [Step 6](06-collect-data.md) you learned there is not enough Sanskrit text
to train a strong model from zero. This step is the answer to that problem.

Think of it like this. Training from scratch is raising a child from birth: it
needs years of input before it can hold a conversation. Adapting a trained
model is teaching a new language to an educated adult. They already know how
the world works, how to reason, and how to answer a question. You only have to
teach them Sanskrit.

:::{important} This is not cheating
It is how essentially every credible low-resource language model has actually
been built. The successful examples — for Sanskrit, for Kazakh, for Latin, for
Ancient Greek — almost all use continued pretraining and vocabulary extension
on a strong base model, not random initialization.

If you only do one thing from this book, do this step.
:::

---

## What you do

### 1. Choose a base model

In this order of importance:

**Licence first.** Apache 2.0 and MIT are the most permissive. Some licences
restrict commercial use, or add user-count limits, or restrict using outputs to
train other models. Read the actual text, not the summary.

**Existing coverage of your language second.** Some model families already
handle Indic scripts far better than others. Test before you commit: run your
[Step 3](03-tokenizers.md) fertility script on the candidates, and try a few
Sanskrit prompts.

**Size third.** Pick the largest you can comfortably fine-tune with your
hardware. On a 128 GB unified-memory machine, a 7B to 14B model with QLoRA is
very comfortable.

:::{warning} Do not trust any list of "best models"
The specific best model changes every few months, including any list in this
book. Check current leaderboards, then test the candidates yourself on Sanskrit.
:::

### 2. Measure the base model first

Run your [Step 10](10-evaluation.md) tests on it **before** you change
anything.

This is your baseline. Without it you will never know whether you helped.
People skip this constantly and then cannot answer the only question that
matters.

### 3. Do tokenizer surgery

The base model's tokenizer probably shreds Sanskrit. Fix it:

1. Add your [Step 4](04-sanskrit-tokenizer.md) Sanskrit tokens to its
   vocabulary.
2. Make the embedding table and the output layer bigger to fit them.
3. **Initialize the new embeddings properly.**

That third point deserves emphasis. Do not use random values. Set each new
token's embedding to the **average of the embeddings of the old pieces that
used to make up that token**.

Example: if your new token is `तस्मात्` and the old tokenizer broke it into four
pieces, average those four embeddings and use the result.

This one detail makes a very large difference to how quickly training
stabilises, and it is skipped constantly.

### 4. Run continued pretraining

Train on your clean Sanskrit corpus with the ordinary next-token objective.

Use a **much lower learning rate** than [Step 9](09-training-run.md). You are
adjusting a trained model, not building one. A good starting point is near the
minimum learning rate the base model finished its own pretraining at, then
cosine decay from there.

On a 128 GB machine, QLoRA on a 7B model makes this run in hours rather than
days. Tools like Unsloth roughly double the speed and halve the memory compared
to a plain setup.

### 5. Mix in replay data

Include some general text from the original distribution — perhaps 5 to 30
percent of your batches.

This is what stops the next problem.

### 6. Measure catastrophic forgetting

**Catastrophic forgetting** is when the model gets better at Sanskrit while
getting worse at everything else.

It is easy to cause and easy to miss, because you are only looking at your
Sanskrit numbers.

Re-run general benchmarks after training. If reasoning and maths fell off a
cliff, raise your replay percentage and try again.

### 7. Decide: full fine-tune, LoRA, or QLoRA

- **Full** — updates every weight. Best quality, most memory.
- **LoRA** — freezes the model and trains small add-on matrices. Like adding a
  thin removable layer instead of repainting the house.
- **QLoRA** — LoRA on top of a compressed model. Cheapest.

For continued pretraining, where you *are* teaching genuinely new knowledge,
full fine-tuning is better if you can afford it. LoRA with a high rank is a
reasonable compromise. This is the opposite of the advice in
[Step 12](12-instruction-tuning.md), and the reason is worth understanding: new
knowledge needs more capacity than new behaviour.

:::{note} What the LoRA target names mean
When you configure LoRA you choose which matrices to adapt:

- `q_proj`, `k_proj`, `v_proj` — the query, key, and value projections inside
  attention. You built these by hand in [Step 1](01-build-transformer.md).
- `o_proj` — the output projection after attention.
- `gate_proj`, `up_proj`, `down_proj` — the three matrices of the SwiGLU
  feed-forward block from [Step 8](08-modern-architecture.md).

This is why Steps 1 and 8 were worth doing. These are not opaque strings. You
know exactly what each one does.
:::

---

## Where people usually get stuck

**Randomly initializing the new embeddings**, then wondering why the model
produces garbage for the first several thousand steps and never fully recovers.

---

## You are ready to move on when

Your adapted model beats the base model on your Sanskrit tests, without losing
much of its general ability.

---

:::{seealso} Related
- [Step 6](06-collect-data.md) — why you are here
- [Step 4](04-sanskrit-tokenizer.md) — the tokenizer you are grafting on
- [Should you build from scratch at all?](appendix/scratch-or-finetune.md)
:::
