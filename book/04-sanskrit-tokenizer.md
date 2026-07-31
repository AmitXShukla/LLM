---
title: "Step 4 — Build a Sanskrit tokenizer"
short_title: "4. Sanskrit tokenizer"
---

# Step 4 — Build a Sanskrit tokenizer

**Goal:** train a tokenizer for Sanskrit, and prove it beats the
general-purpose ones.

---

## Why this step matters

This is your first real, original contribution. It is small, it is measurable,
and it is genuinely useful to other people.

It is also the step where Sanskrit stops being an example and starts being an
interesting technical problem.

---

## What you do

### 1. Normalize everything first

Apply NFC normalization to your whole corpus. Then look for the common
Devanagari problems:

- **The virama** (halant, `्`) marks a consonant with no vowel, and is how
  conjunct letters are formed. Different sources use it inconsistently.
- **The nukta** modifies a consonant and is sometimes present as a separate
  character and sometimes baked into a single code point.
- **Vowel signs** are occasionally written in different orders that render
  identically.

Count your unique characters before and after normalizing. The drop tells you
how much silent duplication you had.

### 2. Decide what to do about sandhi

This is the interesting Sanskrit-specific question, and it has no settled
answer.

**Sandhi** is the set of rules by which sounds change where words meet. When
words join, the join changes both of them:

```
तत् + अपि  →  तदपि
```

So the "word" sitting in your text may be three joined words wearing a
disguise. A tokenizer trained on raw text will learn the disguise, not the
words.

You have two options:

**Option A — leave it joined.** Let BPE learn the joined forms. Simple, no
extra failure points, but your vocabulary fills up with combinations, and
fertility stays high.

**Option B — split sandhi first.** Run a sandhi splitter over the corpus, train
the tokenizer on split text, and re-join at generation time. Lower fertility
and cleaner units, but you have added a component that can be wrong.

**Do both. Measure both.** Report the numbers. This is exactly the kind of small
honest experiment that is worth publishing, and it is the kind of thing large
generalist teams never bother to do.

### 3. Handle compound words

Sanskrit compounds (**samasa**) can be extremely long — sometimes an entire
descriptive phrase becomes one written word.

Decide whether your tokenizer should split them at their natural joins or treat
them as single units. Again: measure, do not guess.

### 4. Pick a vocabulary size

For a single language, **32,000 to 64,000** tokens is usually the sweet spot.

Bigger is not automatically better. A bigger vocabulary means fewer tokens per
document, but a much larger embedding table. In a small model, that table can
end up being most of your parameters — so you pay for vocabulary with model
capacity.

Published work on Indic tokenizers finds returns flattening out well before
200,000 tokens, and training corpus size mattering less than you would expect
past about 10 GB.

### 5. Measure fertility against the general-purpose tokenizers

Use the script from [Step 3](03-tokenizers.md), on held-out Sanskrit text that
your tokenizer never saw.

### 6. Publish the numbers

A simple table of tokenizer versus fertility, on real Sanskrit text, with the
test set included, is something people will actually use and cite.

---

## Where people usually get stuck

**Training the tokenizer on text that was never normalized.**

You get near-duplicate tokens that look identical on screen, your vocabulary
fills with junk, and you will never work out why. Normalize first. Always.

---

## You are ready to move on when

Your tokenizer beats the general-purpose ones on fertility, and you have the
table to prove it.

---

:::{seealso} Related
- [Step 3](03-tokenizers.md) — the concepts
- [Step 15](15-panini-neurosymbolic.md) — sandhi as a rule system, not a nuisance
- [Corpora appendix](appendix/corpora.md) — where to get Sanskrit text
:::
