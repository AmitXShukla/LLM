---
title: "Step 7 — Clean your data"
short_title: "7. Clean data"
---

# Step 7 — Clean your data

**Goal:** turn a pile of scraped text into a training corpus you trust.

---

## Why this step matters

Small models are extremely sensitive to bad data. A large model trained on
trillions of tokens can absorb some noise. Yours cannot.

Digitised ancient texts have a particular problem: OCR errors. A scanner
reading a hundred-year-old Devanagari print will make mistakes, and those
mistakes become things your model confidently learns.

---

## What you do

### 1. Detect the language

Off-the-shelf language detectors are poor at telling apart closely related
South Asian languages, and they fall apart completely on code-mixed text.

For Sanskrit specifically, watch for **Hindi being misidentified as Sanskrit**.
Both use Devanagari, and a detector that only looks at the script will happily
let modern Hindi into your classical corpus.

Test your detector on text you have checked by hand before you trust it on a
million documents.

### 2. Remove exact duplicates

Hash every document. Drop repeats. Easy and fast.

### 3. Remove near-duplicates

Use MinHash with LSH (locality-sensitive hashing) to find documents that are
*almost* the same.

**This matters enormously for Sanskrit.** The same verse appears across dozens
of sources, with small differences in punctuation, transliteration, or
commentary. Exact-duplicate removal will not catch these.

If you skip this, your model memorises instead of learning. You saw exactly
what that looks like in [Step 2](02-break-it.md).

### 4. Fix OCR errors

For scanned text, expect:

- Confused similar-looking characters
- Lost or wrongly placed diacritics and vowel signs
- Broken conjunct letters
- Page numbers, headers, and footnotes mixed into the body text

Build a list of the most common errors in *your* sources and fix them with
rules. Perfect is not the goal. Reducing the top ten error patterns gets you
most of the benefit.

### 5. Filter for quality

Start simple:

- Documents that are too short to be useful
- Documents that are mostly repeated lines
- Documents that are mostly symbols, numbers, or markup

Then, if you want, filter using perplexity from a reference model. Anything the
reference model finds wildly surprising is often broken text rather than
interesting text.

### 6. Check for contamination

Remove anything that appears in the test sets you plan to evaluate on.

Do this **now**, before training. If you do it after, your
[Step 10](10-evaluation.md) numbers are a lie and you will not know.

### 7. Remove personal information

Names, phone numbers, addresses, and identifiers. Less of an issue for
classical texts, a real issue for scraped Urdu web data.

### 8. Consider transliteration as a data source

Sanskrit is written in several scripts: Devanagari, Grantha, Telugu, Kannada,
and Roman (IAST). Converting between them is mechanical and reliable.

This can meaningfully increase your usable data, and it also teaches your model
that the same text can wear different clothes.

The same trick works between Urdu script and Hindi Devanagari, since the spoken
languages are very close.

### 9. Consider synthetic data carefully

You can generate text with a large model, or translate text into Sanskrit.

Two warnings:

- **It copies the teacher's mistakes into your language.** If a large model
  produces slightly wrong Sanskrit, and you train on it, your model learns the
  wrong Sanskrit as truth.
- **Check the licence.** Many model licences restrict using their output to
  train competing models. Read the actual text before you build on it.

### 10. Split the data

Train, validation, and test. Split by *source document*, not by line, or the
same verse will appear on both sides of your split and your test scores will be
meaningless.

---

## Where people usually get stuck

**Skipping near-duplicate removal because exact-duplicate removal already ran.**

They are not the same thing, and the second one matters far more for Sanskrit.

---

## You are ready to move on when

You have a clean corpus, a documented recipe someone else could reproduce, and
a clean train/validation/test split.

---

:::{seealso} Related
- [Step 6](06-collect-data.md) — collecting it
- [Step 10](10-evaluation.md) — why contamination checking mattered
:::

---

## 🧑‍💻 Runnable code for this step

:::{tip} The full, tested file
[`code/step-06-data-audit/prepare_data.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-06-data-audit) turns PDFs/text in `./data` into one clean `corpus.txt` — and, crucially, *measures* how much real Devanagari it found so you know which files need OCR.
:::

The honest part of this script is that it doesn't pretend a scanned PDF worked.
It scores each file and tells you the truth:

```python
DEV_START, DEV_END = 0x0900, 0x097F      # the Devanagari Unicode block

def devanagari_ratio(text):
    meaningful = [c for c in text if not c.isspace()]
    if not meaningful: return 0.0
    dev = sum(1 for c in meaningful if DEV_START <= ord(c) <= DEV_END)
    return dev / len(meaningful)
# ...
#   sample_corpus.txt   devanagari=100.0%  chars_kept=845   [OK]
#   old_scan.pdf        devanagari=  2.3%  chars_kept= 11    [⚠ LOW → needs OCR]
```

```{mermaid}
flowchart LR
    A[📄 PDFs / text in ./data] --> B{Devanagari<br/>ratio ≥ 30%?}
    B -->|yes ✅| C[clean + NFC normalize] --> D[(corpus.txt)]
    B -->|no ⚠️| E[OCR with Tesseract<br/>-l san+hin] --> A
```

:::{warning} This is where projects actually get stuck
Not the neural network — the data. Half of real Sanskrit PDFs are scanned images
or legacy (non-Unicode) fonts that extract as garbage. Budget most of your time
here. 🧹
:::
