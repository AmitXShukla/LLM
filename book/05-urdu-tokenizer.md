---
title: "Step 5 — Build an Urdu tokenizer"
short_title: "5. Urdu tokenizer"
---

# Step 5 — Build an Urdu tokenizer

**Goal:** repeat Step 4 for a script that behaves completely differently.

---

## Why this step matters

Sanskrit taught you about **morphology** — how words are built and joined.

Urdu will teach you about **script and encoding** — a separate skill, and one
that catches almost everyone the first time.

---

## What you do

### 1. Separate rendering problems from encoding problems

Urdu letters change shape depending on where they sit in a word: one form at
the start, another in the middle, another at the end, another alone.

This is **rendering**. Unicode stores the same base letter regardless of shape.
Your tokenizer never sees the shapes.

Beginners lose days here, trying to handle something that is their font's job.
Do not be one of them.

### 2. Fix the Arabic-versus-Urdu character confusion

**This** is the real encoding problem, and it is everywhere in scraped text.

| Urdu uses | Arabic uses | They look nearly identical |
|---|---|---|
| `ی` (Farsi yeh) | `ي` (Arabic yeh) | yes |
| `ک` (keheh) | `ك` (Arabic kaf) | yes |
| `ہ` (heh goal) | `ه` (heh) | yes |

Web-scraped Urdu contains a mixture of both sets, often within the same
document. To your model these are different letters, so the same word becomes
several different words.

Map them all to one consistent form before you do anything else.

### 3. Deal with the zero-width non-joiner

The zero-width non-joiner (ZWNJ, `U+200C`) is an invisible character that stops
two letters from joining. Urdu uses it a lot, and uses it inconsistently.

Decide on one rule — keep it, strip it, or normalize it — and apply that rule
everywhere.

### 4. Decide about short vowels

Urdu usually does not write its short vowels. Sometimes they appear as
diacritics (zer, zabar, pesh), especially in poetry, religious text, and
teaching material.

This means the same written word can be read several ways, and the reader
resolves it from context. Your model will have to do the same.

Decide whether to keep or strip the diacritics, and be consistent.

### 5. Make a deliberate decision about Roman Urdu

A very large share of Urdu written online uses English letters. This is called
Roman Urdu, and there is no standard spelling for it.

Ask yourself directly: **is that your language or not?**

- If your users type in Roman Urdu, you must support it.
- If you only want Urdu script, you must filter it out on purpose.

**Do not let your data cleaning script make this decision by accident.** Decide
it, write down why, and be consistent.

### 6. Measure fertility and compare

You may be surprised here. Urdu often has *lower* fertility than other South
Asian languages under general-purpose tokenizers, so your gains may be smaller
than they were for Sanskrit.

That is a useful and honest result. Report it. Not every language has the same
amount of headroom, and knowing where the headroom is *not* is valuable
information.

---

## Where people usually get stuck

**Right-to-left text confusing them in the terminal.**

Text is stored in **logical order** — the order you would say it — not in the
order it appears on screen. Your editor reverses it for display.

Trust the code point sequence. Print `[hex(ord(c)) for c in text]` when you are
confused. Do not trust your eyes.

---

## You are ready to move on when

You have working tokenizers for both Sanskrit and Urdu, with measured fertility
numbers for both, and you can explain why the two languages needed different
work.

---

:::{seealso} Related
- [Step 4](04-sanskrit-tokenizer.md) — the Sanskrit version
- [Step 7](07-clean-data.md) — where the Roman Urdu decision bites
:::
