---
title: Who else is working on this
short_title: The landscape
---

# Who else is working on this

An honest look at the competition, and why there is still room for you.

:::{note} This page dates quickly
Check current sources before relying on any specific claim here.
:::

---

## The short version

There is real activity around Indic languages, and some dedicated Sanskrit
work. But there is no mature, high-performing, specialised classical Sanskrit
model that scholars and students actually use for deep work.

The gap for a strong, accessible, **local-first** model is real.

---

## The large efforts

**Bhashini** — the Indian government's national language technology mission.
Covers 22-plus scheduled languages with translation, voice, and multilingual
models. Sanskrit gets some support inside the broader stack, but deep classical
work is not the focus.

**AI4Bharat** (IIT Madras) and related academic groups — strong translation
models including Sanskrit, multilingual Indic models, benchmarks, and
digitisation. They include Sanskrit as one language among many.

**Institutional digitisation projects** — including a major collaboration in
Chennai between IIT Madras and Madras Sanskrit College, working through more
than 110,000 rare manuscripts to build a native Sanskrit model. Real and
valuable, but weighted toward preservation and digitisation rather than a
polished usable model.

**BharatGen** and other indigenous platforms — broad Indic coverage.

---

## The smaller efforts

- **Paramanu-Sanskrit** (around 139 million parameters) and similar small
  from-scratch models
- **ByT5-Sanskrit**, focused on sandhi splitting and syntax parsing
- Llama-family fine-tunes adapted for Sanskrit generation
- Assorted cleaned Sanskrit datasets on public platforms

---

## Why this is good news for you

**They are doing the expensive part.** Digitisation, benchmark creation, and
corpus cleaning are slow, costly, and mostly published openly. You do not have
to repeat any of it.

**They are optimising for breadth.** A model covering 22 languages cannot go
deep on one. That is not a criticism, it is what their goal requires.

**Nobody is doing depth on classical Sanskrit.** Precise sandhi handling,
faithful translation of shastra, metre-aware generation, philosophical analysis
that does not fabricate — this is unclaimed.

**Nobody is doing neuro-symbolic seriously.** The Panini angle in
[Step 15](../15-panini-neurosymbolic.md) has very little competition, largely
because it requires both engineering and domain knowledge, and few people have
both.

---

## The honest risks

**Data is still the main limit.** GRETIL, SARIT, and DCS are real resources, but
cleaning, sandhi handling, and scholarly validation take genuine work.

**You will need scholarly help for evaluation.** In philosophical and religious
material, a fluent wrong answer is worse than no answer. You cannot judge this
alone unless you are already a Sanskritist.

**Competition exists at the broad Indic level.** If your plan is "a general
Indic model", you will lose. If your plan is "the best tool for splitting
compounds in classical texts", you can win.

**Your model will be superseded.** Plan for it. This is why
[Step 25](../25-release.md) says your evaluation set and tokenizer may outlast
your weights.
