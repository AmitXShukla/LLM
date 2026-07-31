---
title: "Step 3 — Understand tokenizers"
short_title: "3. Tokenizers"
---

# Step 3 — Understand tokenizers

**Goal:** understand how text gets cut into pieces, and why the cutting matters
more than almost anything else you will do.

---

## Why this step matters

Think of tokenization as cutting vegetables before you cook. If you cut them
badly, everything after that is harder, slower, and worse. Nothing you do later
fixes a bad cut.

A **token** is one piece of text. It might be a whole word, part of a word, or
a single letter. A **tokenizer** is the tool that does the cutting.

Character-level, from [Step 1](01-build-transformer.md), cuts too finely. One
sentence becomes hundreds of tokens. Attention cost grows with the *square* of
the number of tokens, so this gets expensive very fast.

### The number that matters: fertility

**Fertility** is simply this: how many tokens does one average word become?

- English, with a normal tokenizer: about **1.3** tokens per word.
- Sanskrit, Tamil, and Telugu with the same tokenizer: often **3 to 4**, and
  sometimes worse.

Sit with that number for a moment. If your tokenizer needs 3 tokens where
English needs 1.3, then:

- Your context window holds **less than half** as much Sanskrit.
- Training costs **more than twice** as much for the same amount of meaning.
- Your model wastes capacity gluing word pieces back together.
- Every user query costs you more to serve, forever.

Purpose-built Indic tokenizers get this down to roughly **1.4 to 2.1**.

**That improvement is free performance, and it is available to you in one
afternoon.** This is the highest-leverage chapter in Part 2.

---

## What you do

### 1. Write Byte Pair Encoding from scratch

BPE is about 80 lines of Python and the idea is simple:

1. Start with single characters.
2. Find the most common pair of neighbouring pieces.
3. Glue that pair into one new token.
4. Repeat until you have the vocabulary size you want.

That is the whole algorithm. Write it. It permanently demystifies the subject.

### 2. Learn the two main families

- **BPE** glues pieces together, from small to large.
- **Unigram** (used by SentencePiece) starts with a large candidate list and
  throws away the least useful pieces, from large to small.

Both work. BPE is more common now.

### 3. Understand pre-tokenization

Before BPE ever runs, a regular expression splits the text into rough chunks —
usually at spaces and punctuation.

**This is where most of the damage to Indic scripts happens.** Not in the BPE
algorithm. In the regex that runs before it.

Rules written for English split Devanagari in silly places. Published work has
shown that fixing only the pre-tokenization rules can cut Indic fertility from
roughly 4.3 down to roughly 2.0 — before changing anything else at all.

Read the pre-tokenizer configuration of any tokenizer you plan to use. It is
usually in `tokenizer.json`.

### 4. Understand Unicode normalization

Two Devanagari strings can look completely identical on screen but be stored as
different sequences of code points.

To you, they are the same word. To your tokenizer, they are two different
words, taking two vocabulary slots and splitting your training signal in half.

Always normalize before doing anything else. NFC is the usual choice.

```python
import unicodedata
text = unicodedata.normalize("NFC", text)
```

This is not optional and it is not a detail.

### 5. Build a fertility measuring script

This is your deliverable for this step.

```python
def fertility(tokenizer, text):
    words = text.split()
    tokens = tokenizer.encode(text)
    return len(tokens) / len(words)
```

Run it on held-out Sanskrit text against three or four tokenizers from
different model families. Write the numbers in a table.

You now have a measurement tool you will use for the rest of the book.

---

## Where people usually get stuck

**Believing that a big multilingual tokenizer "supports" your language because
the letters do not turn into question marks.**

Displaying correctly and tokenizing well are completely different things. A
tokenizer can handle every Devanagari character perfectly and still be terrible
at Sanskrit.

Always measure fertility yourself. Never trust a claim of support.

---

## You are ready to move on when

You can measure the fertility of any tokenizer on any text file, using a script
you wrote, and you have a table of results for Sanskrit.

---

:::{seealso} Related
- [Step 4](04-sanskrit-tokenizer.md) — build one for Sanskrit
- [Step 5](05-urdu-tokenizer.md) — build one for Urdu
- [Step 11](11-adapt-base-model.md) — where you will use it on a large model
:::
