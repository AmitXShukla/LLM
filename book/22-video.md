---
title: "Step 22 — Video"
short_title: "22. Video"
---

# Step 22 — Video

:::{note} Chapter status
Outline. To be expanded. Consider skipping this one entirely.
:::

**Goal:** understand what makes video hard.

---

## Why this step matters

Video is images plus time, which mostly means video is images plus an enormous
number of tokens.

Everything you learned still applies. The problem is purely scale.

---

## What to cover

1. **Frame sampling.** You cannot use every frame. Deciding which ones to keep
   is most of the engineering.
2. **Temporal tokenization.** Turning a sequence of frames into tokens without
   producing a million of them.
3. **The context length problem.** This connects directly to
   [Step 19](19-long-context-rag.md).
4. **Video-language models.** The same projector pattern as
   [Step 21](21-vision.md), with a time dimension added.

---

## An honest note

This is expensive and it is the least relevant part of the book for a Sanskrit
or Urdu project.

Possible uses: recitation videos where mouth position matters, teaching
material, ritual documentation. Genuine, but narrow.

Do it last, if at all.
