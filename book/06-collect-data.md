---
title: "Step 6 — Collect your data"
short_title: "6. Collect data"
---

# Step 6 — Collect your data

**Goal:** build the biggest, cleanest Sanskrit corpus you can, and find out
honestly how big that actually is.

---

## Why this step matters

This is not a preparation step. For a low-resource language, **this is the
project**. Everything else in this book is easier than this.

It is also the step where you discover the single most important fact about
your project. Do not skip it, and do not guess the answer.

---

## What you do

### 1. Gather from every source you can find

- Sanskrit Wikipedia and Wikisource dumps
- Digital Sanskrit archives — GRETIL, SARIT, the Digital Corpus of Sanskrit
- Public-domain printed books
- The language splits of large web crawls
- Scanned books put through OCR, with all the noise that brings
- **Your own collection**, if you have one — this is your moat

See the [corpora appendix](appendix/corpora.md) for a working list.

### 2. Record where every single file came from

Source, date, licence, and how you got it. Do this **while** you collect, not at
release time.

Reconstructing this later is miserable, and without it you cannot legally
release anything.

### 3. Count your tokens

Use your Step 4 tokenizer, not a general-purpose one. The number will be
different, and yours is the one that matters.

### 4. Compare against what you need

A rough rule of thumb: a useful model wants somewhere between **5 and 20 tokens
of training data for every parameter it has**.

So:

| Model size | Tokens wanted |
|---|---|
| 100 million parameters | 0.5 to 2 billion |
| 500 million parameters | 2.5 to 10 billion |
| 1 billion parameters | 5 to 20 billion |
| 7 billion parameters | 35 to 140 billion |

### 5. Face the number

You will almost certainly find that all the clean Sanskrit text in the world
adds up to far less than the smallest row in that table. Perhaps a few hundred
million tokens, and much of that repeated across sources.

:::{important} This is the most important discovery in the book
It is not a failure of your search. It is the actual state of the world.

Every ancient and low-resource language has this problem. Even
well-digitised ones — Latin, Ancient Greek, Sanskrit — have corpora in the
millions to low billions of tokens at best. That is plenty for
*specialisation* and nowhere near enough for from-scratch pretraining of a
capable model.

This changes what you should build. It is why [Step 11](11-adapt-base-model.md)
exists, and it is why the successful low-resource models you can find were
almost all built by adapting a strong base model rather than starting from
random weights.
:::

### 6. Do the same for Urdu

You will find much more text, and much of it much dirtier. A different problem,
needing different solutions.

### 7. Write down your answer to one question

**What do you actually want this model to do?**

Not "understand Sanskrit". Something you could test:

- Translate classical texts into English faithfully?
- Split sandhi and parse grammar?
- Answer questions about a specific body of texts?
- Generate text in a particular classical style?
- Check and correct metre?

Your answer changes your data mix, your evaluation, and your architecture. A
vague answer here produces a vague model later.

---

## Where people usually get stuck

**Assuming more data exists and that they just have not found it yet.**

Do the count. Trust the count. The count is the plan.

---

## You are ready to move on when

You have a token count for both languages, you believe it, and you have written
one clear sentence saying what your model is for.

---

:::{seealso} Related
- [Corpora appendix](appendix/corpora.md) — where to find text
- [Step 7](07-clean-data.md) — turning it into a corpus
- [Step 11](11-adapt-base-model.md) — what to do about the number you just found
:::
