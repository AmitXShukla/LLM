---
title: "Step 20 — Speech"
short_title: "20. Speech"
---

# Step 20 — Speech

:::{note} Chapter status
Outline. To be expanded.
:::

**Goal:** understand speech models, and see why this may be the highest-impact
area of all.

---

## Why this step matters

Many people who speak South Asian languages read them less easily than they
speak them. Voice is the natural interface. And speech recognition quality for
most of these languages is poor.

This is where a small team can do the most good.

For Sanskrit there is something more specific. Sanskrit is a **recited**
language. Chanting follows strict rules of metre and pronunciation that have
been preserved orally for thousands of years. That structure is both a challenge
and an unusual opportunity: you have rule-based ground truth for what the audio
should sound like.

---

## What to cover

1. **Speech recognition.** Encoder-decoder models that turn audio into text. The
   Whisper family is the usual starting point.

2. **Self-supervised speech representations.** Models trained on unlabelled
   audio, which matters a lot when labelled audio is scarce.

3. **Discrete audio tokens.** The key idea for you: audio can be turned into a
   sequence of discrete tokens using a neural codec. Once it is tokens,
   **everything you learned about text applies directly** — same architecture,
   same training, same tricks.

4. **Text to speech.** Generating Sanskrit recitation with correct metre.

5. **End-to-end speech models.** Audio in, audio out, with no text step.

6. **Sanskrit-specific opportunities:**
   - Recitation checking — does this chant follow the correct metre and
     pronunciation?
   - Aligning existing recordings with existing texts to build training data
   - Pronunciation teaching tools

---

:::{seealso} Related
- [Step 15](15-panini-neurosymbolic.md) — metre rules as ground truth for audio
- [Step 24](24-medical.md) — the same techniques on heart sounds
:::
