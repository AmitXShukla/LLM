---
title: "Step 10 — Test it honestly"
short_title: "10. Evaluation"
---

# Step 10 — Test it honestly

**Goal:** find out whether your model is actually any good.

---

## Why this step matters

A falling loss curve does not mean a good model.

This step separates a real project from a demo. It is also, for Sanskrit and
Urdu, mostly unexplored ground — which means the work you do here may be more
valuable to other people than your model.

---

## What you do

### 1. Compute perplexity on held-out text

Perplexity roughly means "how surprised is the model by this text". Lower is
better.

It is easy to compute and genuinely useful as a sanity check. It is not
sufficient.

### 2. Never compare perplexity across different tokenizers

A model with a bigger vocabulary shows lower perplexity while being no better,
because each token carries more text.

If you must compare across tokenizers, normalize **per character** or **per
byte**, not per token. Otherwise you are measuring your tokenizer, not your
model.

### 3. Accept that the benchmarks you need do not exist

Most multilingual benchmarks for these languages are machine translations of
English benchmarks. They test translation quality as much as language ability.

For classical Sanskrit specifically, there is very little that tests what you
actually care about.

### 4. Build your own tests

This is real work and it is publishable. Good Sanskrit tasks:

- **Sandhi splitting.** Give a joined form, ask for the parts. Mechanically
  checkable.
- **Compound splitting.** Break a samasa into its members.
- **Grammatical agreement.** Which form is correct in this sentence?
- **Metre.** Does this line scan correctly? Fully rule-based, so fully
  checkable.
- **Verse completion.** Complete a verse from a known text, check against the
  real one.
- **Faithful translation.** Into English or Hindi, scored by someone who knows
  both.

Good Urdu tasks:

- Correct reading of a word with unwritten short vowels, given context
- Roman Urdu to Urdu script conversion
- Formal versus informal register
- Poetry metre, which Urdu also has strict rules for

:::{tip} These tests are worth more than your model
Your model will be superseded. A careful, native-checked evaluation set for
classical Sanskrit will still be useful in ten years. Release it separately,
and release it early.
:::

### 5. Use an LLM as a judge carefully

Having a large model score your outputs is fast and useful. It also has known
biases:

- It prefers longer answers
- It prefers whichever answer it saw first
- It prefers text that sounds like its own writing

Never use it as your only measure.

### 6. Do human evaluation

Even 30 examples reviewed by one person who genuinely knows the language will
teach you more than 3,000 automatic scores.

For Sanskrit this usually means finding a scholar. It is worth the effort. In
philosophical and religious material, a confident wrong answer is much more
costly than an uncertain one, and only a human will catch the difference.

### 7. Check you did not break the base model

If you came from [Step 11](11-adapt-base-model.md), also re-run general
benchmarks. See the note on catastrophic forgetting there.

---

## Where people usually get stuck

**Reporting only perplexity, because it is easy, and never finding out that the
model produces fluent nonsense.**

Fluent nonsense is the characteristic failure of a small model on a rich
language. Perplexity will not show it to you. A human will, in about four
minutes.

---

## You are ready to move on when

You have a small evaluation suite you built yourself, honest numbers from it,
and at least one human review from someone who knows the language.

---

:::{seealso} Related
- [Step 7](07-clean-data.md) — why contamination checking mattered
- [Step 15](15-panini-neurosymbolic.md) — rule-based checking as evaluation
:::
