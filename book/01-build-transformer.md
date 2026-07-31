---
title: "Step 1 — Build a tiny Sanskrit transformer"
short_title: "1. Tiny transformer"
---

# Step 1 — Build a tiny Sanskrit transformer by hand

**Goal:** write a small character-level language model from a blank file, with
no `transformers` import.

---

## Why this step matters

A language model does one thing: it looks at some text and guesses the next
piece. Everything else — all the size, all the tricks, all the products built
on top — sits on that one idea.

You need to feel this in your hands before anything else makes sense.

A **character-level** model means the model works with single letters instead
of words. It is the simplest possible version. It is also slow and weak, and
you will replace it in [Step 3](03-tokenizers.md). That is fine. Right now you
are learning, not building a product.

:::{important} Time-box this
Give this step two days. No more. The goal is understanding, not a good model.
If you find yourself tuning hyperparameters to squeeze out a better loss, stop
— you have finished the useful part and started the useless part.
:::

---

## What you do

### 1. Get a small Sanskrit text file

A few megabytes is plenty. The Ramayana, the Bhagavad Gita, or a Sanskrit
Wikipedia dump all work. See the
[corpora appendix](appendix/corpora.md) for places to get text.

Use *your own* text if you have some. That matters, because it forces you to
meet Sanskrit's real problems on day one instead of reading about them later.

### 2. Build a character vocabulary

List every unique character in your file. Give each one a number. Write a
function that turns text into numbers, and one that turns numbers back into
text.

Look at how many unique characters you got. Devanagari will give you more than
you expect, because vowel signs, the virama, and conjunct forms all count
separately. This is your first small hint that tokenization is going to be the
real problem.

### 3. Write a data loader

Pick a random point in the text. Take the next 128 characters as the input.
Take the same 128 characters shifted along by one as the answer.

That shift is the entire training task. The model sees `त`, and must predict
`त्`. Then it sees `तत्`, and must predict the space. And so on.

### 4. Write single-head self-attention by hand

For each position, the model produces three vectors: a **query**, a **key**,
and a **value**.

Here is the intuition. Each character asks a question — that is the query.
Every earlier character offers a label describing what it is — that is the key.
The model compares the question against all the labels, and mixes together the
actual content — the values — of whichever earlier characters matched best.

That is it. Attention is a weighted average, where the weights are learned by
matching questions against labels.

### 5. Add the causal mask

The model must not look at future characters.

If it can see the answer, it will simply copy it. It will look like it is
learning brilliantly and it will have learned nothing. The mask covers the
future so this cannot happen.

You will prove this to yourself in [Step 2](02-break-it.md).

### 6. Add multiple heads

Run several attention operations side by side and join the results together.
Different heads learn to look for different things: one might track word
boundaries, another might track long-range agreement.

### 7. Build the full block

One block is:

1. Normalize
2. Attention
3. Add the input back in
4. Normalize
5. A small feed-forward network
6. Add the input back in again

Steps 3 and 6 are called **residual connections**. They give information a side
road around the block. This is what lets you stack many blocks without the
learning signal fading away.

### 8. Stack four to six blocks, and add a training loop

Use the AdamW optimizer and a cosine learning rate schedule. Keep it simple.

### 9. Write a sampling function

You need to be able to read what your model produces. Add temperature and
top-k sampling so you can control how adventurous it is.

### 10. Train it, and read the output

---

## What you should see

The loss goes down. The samples look like Sanskrit at a glance. They have the
right letters, roughly the right word lengths, and roughly the right rhythm.

They mean nothing at all.

**This is the correct result.** Your model is tiny and your data is tiny. It has
learned what Sanskrit *looks like*, not what Sanskrit *says*. The rest of this
book is about closing that gap.

If your samples look like meaningless but plausible Sanskrit, you have
succeeded. Do not try to fix it. It is not broken.

---

## Where people usually get stuck

**Shape errors in the attention code.** Print the shape of every tensor on
every line until it works. Everyone does this, including people who have
written transformers before. It is not a sign that you are bad at this.

**Trying to make the model good.** It cannot be good. Move on.

---

## You are ready to move on when

You can explain, out loud, what every line of your model does and why it is
there.

Not "this is the attention layer." More like: "this line divides by the square
root of the head dimension, because without it the dot products get large when
the dimension is large, and then softmax turns into a hard maximum and the
gradients vanish."

---

:::{seealso} Related
- [Step 2](02-break-it.md) — break what you just built
- [Should you build from scratch at all?](appendix/scratch-or-finetune.md) — the honest debate
- [Things worth reading](appendix/reading.md) — nanoGPT and other starting points
:::
