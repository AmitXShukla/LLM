---
title: "Step 2 — Break your model on purpose"
short_title: "2. Break it"
---

# Step 2 — Break your model on purpose

**Goal:** learn what each part actually does, by removing it and watching what
happens.

---

## Why this step matters

You cannot understand a part by reading about it. You understand it by seeing
what breaks without it.

This step takes one afternoon and is worth more than a month of tutorials. It
is also the step people skip, because it feels like going backwards.

It is not going backwards. Every hour here saves you a day of confused
debugging in [Step 9](09-training-run.md).

---

## What you do

Run each experiment below, one at a time, and write down what happened in your
own words.

### 1. Remove the causal mask

**What you will see:** the loss crashes almost to zero within a few hundred
steps. It looks like a spectacular success. Then you sample from the model and
get pure noise.

**Why:** the model was allowed to see the answer. It learned to copy, not to
predict. Predicting the next character is trivial when you can already see it.

This is the most important bug in this entire field, and now you have seen it
with your own eyes. Any time a loss looks too good, this is the first thing to
check.

### 2. Move the normalization to after the residual

This is how the original 2017 paper did it. It is called post-normalization.

**What you will see:** training gets shaky, especially if you add more blocks.

**Why:** normalizing *before* the block keeps a clean, unchanged path running
from the input all the way to the output. Normalizing after means every block
disturbs that path.

This is why every modern model normalizes first. You have just discovered a
real architectural change by breaking something.

### 3. Remove the residual connections

**What you will see:** a deep model refuses to learn at all. The loss barely
moves.

**Why:** without the side road, the learning signal has to squeeze back through
every single block on its way to the early layers. It fades to nothing before
it gets there.

### 4. Set the learning rate ten times too high

**What you will see:** the loss jumps upward suddenly, or turns into `NaN`.

**Why:** now you know what a loss spike looks like. You will meet a real one in
Step 9, and you will recognise it immediately instead of losing a day.

### 5. Train on only 100 KB of text, for a long time

**What you will see:** training loss goes near zero. Validation loss goes up.
Samples are copied word for word from your file.

**Why:** this is **memorisation**, also called overfitting. The model stopped
learning patterns and started learning your specific file.

Recognising this early saves weeks. It is especially important for Sanskrit,
where your total data is small and the same verses appear in many sources. You
will deal with this properly in [Step 7](07-clean-data.md).

---

## Where people usually get stuck

**Doing this too fast.** Change one thing at a time. Two changes at once and you
learn nothing, because you cannot tell which change caused which effect.

**Not writing it down.** The notes are the output of this step, not the
experiments.

---

## You are ready to move on when

You have five short written notes, one per experiment, in your own words.

If you can hand those notes to another engineer and they learn something, you
did this properly.

---

:::{seealso} Related
- [Step 1](01-build-transformer.md) — the model you are breaking
- [Step 9](09-training-run.md) — where these failures show up for real
:::
