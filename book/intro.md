---
title: Introduction
short_title: Introduction
---

# Build Your Own Language Model

### A step-by-step book for engineers, using Sanskrit and Urdu

---

## Who this book is for

You know some Python. You have used PyTorch a little, or you are willing to
learn it as you go. You have heard about GPT, fine-tuning, and LoRA, but you
are not sure what happens inside.

You do not need a PhD. You do not need a room full of GPUs. You need one
machine, some patience, and a willingness to be wrong a few times.

Everything here is written in simple English. Every technical word is explained
the first time it appears, and all of them are collected in the
[glossary](appendix/glossary.md) at the end.

---

## What you will build

By the end of this book you will have built four things:

1. A tiny Sanskrit language model, written by you, line by line.
2. A tokenizer for Sanskrit and one for Urdu that work better than the
   general-purpose ones.
3. A useful Sanskrit model, made by adapting a large open model.
4. A small, fast version of that model that you can actually run and share.

You will also have failed many times along the way. That is planned. Failure is
the fastest teacher here.

---

## Why train your own model in 2026?

Most engineers today only fine-tune models that other people built. There is
nothing wrong with that. But there are good reasons to go one level deeper.

### Reason 1: You can fail cheaply and quickly

A small model trains in a few hours. You can be wrong ten times in one weekend.
Every mistake teaches you something a book cannot.

### Reason 2: Fine-tuning will not teach you how transformers work

This is worth saying clearly, because it is a common and expensive
misunderstanding.

Fine-tuning teaches you a **workflow**: how to format instruction pairs, how to
configure LoRA, how to drive a training harness, how to evaluate, how to
deploy. Those are real and valuable skills.

But the transformer itself stays a closed box that you call into. The attention
maths, the forward and backward pass, the reason the loss drops — none of that
gets exercised. So if your goal is "understand how these things are built",
fine-tuning quietly skips exactly the part you wanted.

### Reason 3: Sanskrit and Urdu are still badly served

Large models treat all South Asian languages as one group. They do not learn
what makes each language special. Sanskrit has sandhi and long compound words.
Urdu is written right to left, drops its short vowels, and mixes with English
online.

Research on Sanskrit keeps finding the same thing: current models produce
grammatically wrong and logically confused Sanskrit, and two of the main
reasons are poor tokenization and a shortage of good annotated data.

**That gap is your opening.**

### Reason 4: Everything you learn here transfers

The same ideas run vision models, speech models, and medical models. Text is
just the cheapest place to learn them. [Step 24](24-medical.md) shows how
directly this transfers to X-rays and heart sounds.

---

## The lone wolf argument

You may be thinking: large organisations have hundreds of GPUs and full-time
teams. What can one person with one machine possibly add?

More than you would expect, and the reason is structural rather than
motivational.

**Large organisations build generalists.** When a frontier model includes
Sanskrit, Sanskrit is a tiny slice of an enormous dataset. The result is a
model that can translate a verse in a shallow way but has no real grip on how
the language works. That is not a failure on their part. It is what optimising
for breadth does.

**You are not competing on scale. You are competing on focus.** Pick a narrow,
well-defined slice — a specific corpus, sandhi splitting, Vedic metre, OCR
cleanup for one manuscript style, one translation direction — and your domain
knowledge and careful data work matter more than raw compute.

Three concrete advantages you have:

**Your data is your moat.** If you hold a well-curated corpus that nobody else
has cleaned properly, no amount of someone else's compute replaces it.

**Local and private is a feature.** Sacred texts, personal study, and sensitive
cultural material are often better handled by a model that never sends anything
to a cloud.

**Open release compounds.** Publish your weights, tokenizer, cleaned data, and
evaluation set early. Sanskrit and Urdu both have passionate communities.
People will contribute corrections and test data once they see a serious
effort. This is how small teams punch above their weight.

There is a fourth advantage specific to Sanskrit, and it is a big one. See
[Step 15](15-panini-neurosymbolic.md).

---

## What this book will not promise you

Let us be honest before you spend weeks on this.

- You will **not** build a model as good as the frontier ones. The gap in
  compute is about five orders of magnitude. Anyone who tells you otherwise is
  selling something.
- You will **not** pretrain a large Sanskrit model from zero. There is not
  enough clean Sanskrit text in the world. You will find this out yourself in
  [Step 6](06-collect-data.md), and [Step 11](11-adapt-base-model.md) shows
  what to do instead.
- You will **not** get a working model by copying scripts without understanding
  them.

What you **will** get is a real, working, specialised model, and the ability to
debug any model anyone hands you.

---

## Why Sanskrit first, and Urdu second?

We use two languages on purpose, because they are different in almost every
way. Learning both teaches you far more than doing one language twice.

### Sanskrit — the primary language

- Written in Devanagari, left to right.
- Words join together and change sound at the join. This is called **sandhi**.
  So `तत् + अपि` becomes `तदपि`. Word boundaries are not fixed.
- Words can be joined into very long compound words. This is called **samasa**.
- The grammar is unusually regular. More on that in a moment.
- Very little text exists online, but what exists is clean and well edited.

### Urdu — the secondary language

- Written in the Perso-Arabic script, right to left.
- Short vowels are usually not written, so the same written word can be read in
  different ways.
- A very large amount of Urdu online is written in English letters, not Urdu
  script. This is called Roman Urdu.
- Much more text exists than Sanskrit, but it is far messier.

So Sanskrit teaches you **structure and scarcity**. Urdu teaches you **mess and
script**. Together they cover most of what you will ever face.

---

## The Panini advantage

This deserves its own section, because it is the most interesting thing about
this project and it has no equivalent in English or Urdu.

Around two and a half thousand years ago, the grammarian Panini wrote the
*Ashtadhyayi*: roughly four thousand short rules, called *sutras*, that
describe Sanskrit. They are not descriptive notes. They form something very
close to a formal system, with ordering, conditions, and exceptions that
resolve in a defined way. People often describe Sanskrit as a language with a
specification.

Why this matters for you:

**Ordinary language models learn by brute force.** English is messy and full of
irregularities, so a model has to see trillions of words before it works out
the patterns. There is no shortcut, because there is no underlying rule set to
find.

**Sanskrit has a rule set, and it is written down.** Much of what a model would
otherwise have to infer from enormous amounts of data is already available as
explicit rules you can run as code.

This changes the trade. You can lean on **depth instead of scale**. You can
check the model's output against the rules. You can generate training data from
the rules. You can build a hybrid system where the neural model handles
meaning and nuance while ordinary Python code enforces the grammar.

That is the subject of [Step 15](15-panini-neurosymbolic.md), and it is the
part of this book with the least prior work behind it.

:::{warning} An honest caution
The Ashtadhyayi is a formal system, but it is not a computer program, and
scholars still disagree about how several rules interact. Existing software
implementations are partial. Treat "Sanskrit is code" as a useful direction to
push in, not as a solved fact.
:::

---

## How to read this book

The book is one long ladder. It runs from **Step 0** to **Step 25**. Each step
assumes you did the step before it.

Every step follows the same shape:

- **Goal** — one line telling you what you are trying to achieve.
- **Why this step matters** — the reasoning, in plain words.
- **What you do** — the actual work, as a numbered list.
- **Where people usually get stuck** — the common mistakes.
- **You are ready to move on when** — a clear finish line.

Do not skip ahead. [Step 2](02-break-it.md), [Step 6](06-collect-data.md), and
[Step 10](10-evaluation.md) look boring and are the three most important
chapters in the book.

:::{tip} If you are short on time
The fastest useful path is Step 0, Step 1, Step 3, Step 4, Step 6, then jump
straight to Step 11. That gets you a working Sanskrit model in about two weeks.
Come back for the rest afterwards. See
[Should you build from scratch at all?](appendix/scratch-or-finetune.md)
:::

---

## The full ladder at a glance

**Part 1 — Understand the machine (Steps 0 to 2)**
Build a transformer by hand and break it on purpose.

**Part 2 — Feed the machine (Steps 3 to 7)**
Tokenizers and data. This is where most of the real work lives.

**Part 3 — Train it properly (Steps 8 to 10)**
Modern design, a real training run, and honest testing.

**Part 4 — Make it useful (Steps 11 to 15)**
Adapt a large model, teach it to follow instructions, teach it to reason, and
add Panini's rules back in.

**Part 5 — Make it small and fast (Steps 16 to 19)**
Mixture of Experts, distillation, quantization, serving, and RAG.

**Part 6 — Go beyond text (Steps 20 to 24)**
Speech, images, video, agents, and medical data.

**Part 7 — Ship it (Step 25)**
Safety, licences, and release.

---

## A note on hardware

You can start this book on a free Colab notebook. You will not get past Step 5
on one.

The book assumes you will eventually have access to one reasonably strong
machine. That might be a rented cloud GPU by the hour, a desktop with a 24 GB
card, or a workstation such as an NVIDIA DGX Spark with 128 GB of unified
memory.

The [hardware appendix](appendix/hardware.md) covers what each tier can and
cannot do, what things cost, and specific notes for the DGX Spark, which is an
unusually good fit for the path this book takes.

---

## Let us begin

Start with [Step 0 — Get your workspace ready](00-setup.md).
