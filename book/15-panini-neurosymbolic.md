---
title: "Step 15 — Panini: add the rules back in"
short_title: "15. Panini and rules"
---

# Step 15 — Panini: add the rules back in

**Goal:** combine your neural model with ordinary code that enforces Sanskrit's
grammatical rules, and build something neither could do alone.

---

## Why this step matters

This is the most original chapter in the book, and the one with the least prior
work behind it.

Every other step in this book applies to any language. This one applies to
Sanskrit specifically, and it is the reason a single engineer with one machine
can build something a large generalist team will not.

### The core idea

Around two and a half thousand years ago, the grammarian Panini wrote the
*Ashtadhyayi*: roughly four thousand short rules, called *sutras*, describing
Sanskrit.

They are not casual observations. They form something very close to a formal
system, with defined ordering, conditions, and conflict resolution. Sanskrit is
often described as a language that shipped with a specification.

Now consider what a normal language model has to do.

**English has no specification.** It is messy, irregular, and full of exceptions
that exist for historical reasons. A model has to see trillions of words before
it works out the patterns, because there is no underlying rule set to find. That
is why scale is the only lever.

**Sanskrit has a specification, and it is written down.** A large part of what a
model would otherwise have to infer from enormous amounts of data is already
available as explicit rules you can run as code.

**This changes the trade.** You can lean on depth instead of scale. You can
check output against rules. You can generate training data from rules. You can
build a system where the neural model handles meaning and nuance while ordinary
Python enforces the grammar.

This approach — a neural model plus a symbolic rule system — is called
**neuro-symbolic**.

:::{warning} An honest caution before you start
The Ashtadhyayi is a formal system, but it is **not** a computer program.

Scholars still disagree about how several rules interact, the metarules
governing rule conflicts are subtle, and every existing software implementation
is partial. Some rules depend on meaning, which is exactly the part you cannot
easily encode.

Treat "Sanskrit is code" as a strong and useful direction, not as a solved
fact. Build the parts that are clearly mechanical. Let the neural model handle
the rest. Do not claim more than you can demonstrate.
:::

---

## What you do

### 1. Start with the clearly mechanical parts

Not all of Panini at once. Pick the rules that are unambiguous and easy to
verify:

- **Sandhi joining.** Given two words, produce the joined form. Highly
  rule-based.
- **Sandhi splitting.** The reverse. Harder, because it is ambiguous — several
  splits may be valid — which makes it a good candidate for the neural model to
  propose and the rules to check.
- **Declension.** Given a stem, a gender, a case, and a number, produce the
  form.
- **Conjugation.** Given a root, a tense, a person, and a number, produce the
  form.
- **Metre.** Given a line, work out its syllable pattern and check it against
  the known metres. Fully mechanical.

Each of these is a small Python module with tests. Build them one at a time.

### 2. Use the rules as a checker

Now you have automatic verifiers. Use them everywhere:

**As evaluation** ([Step 10](10-evaluation.md)) — a real, objective score
instead of vague quality judgements.

**As reward** ([Step 14](14-reasoning.md)) — this is exactly the verifiable
reward that RLVR needs, which is why Sanskrit is unusually well suited to
reasoning training.

**As a guard at generation time** — if the model produces a form the rules
reject, ask it again.

### 3. Use the rules as a data generator

This is the most interesting use, and it partly solves the problem you found in
[Step 6](06-collect-data.md).

You cannot generate meaningful Sanskrit *content* from rules. But you can
generate unlimited correct examples of *grammatical operations*:

- Millions of correct declension examples from a stem list
- Millions of correct sandhi joins from a word list
- Metre-annotated lines from any text you have

This is synthetic data that is **guaranteed correct**, which is unusual and
valuable. It will not teach your model philosophy. It will teach it grammar,
solidly.

### 4. Build the hybrid pipeline

A workable shape:

```
Input text
   ↓
Rule-based pre-processing   (normalize, split sandhi, tag metre)
   ↓
Neural model                (meaning, translation, interpretation)
   ↓
Rule-based checking         (is the output grammatically valid?)
   ↓
If invalid, ask again with the error as feedback
   ↓
Output
```

The division of labour: **rules handle what is deterministic, the neural model
handles what is not.**

### 5. Know when the rules should lose

The rules are not always right, or not always applicable:

- Vedic Sanskrit predates and sometimes contradicts Panini
- Real manuscripts contain genuine irregularities
- Some rules depend on meaning, which your code does not have
- Regional and later traditions differ

Build your system so a rule can be overridden, and log every time it happens.
Those logs are interesting data in themselves.

---

## Where people usually get stuck

**Trying to implement all four thousand sutras before shipping anything.**

You will not finish. Implement sandhi joining and metre checking. Get those
working and tested. Ship. Add more later.

A partial rule system that works is infinitely more useful than a complete one
that does not exist.

---

## You are ready to move on when

You have at least two working rule modules with tests, and you have used them
for at least one of: evaluation, reward, or data generation.

---

:::{seealso} Related
- [Step 10](10-evaluation.md) — rules as objective evaluation
- [Step 14](14-reasoning.md) — rules as verifiable reward
- [Step 6](06-collect-data.md) — rules as a partial answer to data scarcity
:::
