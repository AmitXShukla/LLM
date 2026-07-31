---
title: "Step 19 — Long text and RAG"
short_title: "19. Long context and RAG"
---

# Step 19 — Long text and RAG

:::{note} Chapter status
Outline. To be expanded.
:::

**Goal:** let the model work with a whole text instead of a paragraph.

---

## Why this step matters

Sanskrit texts are long, highly structured, and full of cross-references. A
commentary refers to a verse, which refers to an earlier chapter. A model that
only sees 2,000 tokens cannot do this work.

There are two different tools here, and people confuse them:

- **Long context** — make the model itself able to read more at once.
- **RAG** — retrieve the relevant passages and put only those in the context.

You usually want both.

---

## Part A — Long context

1. **Understand why a short-trained model fails on long text.** It has never
   seen those position values before, so it is being asked to do something
   genuinely unfamiliar.

2. **Learn position interpolation and its improved versions** (NTK-aware
   scaling, YaRN). The idea is to squeeze the position values into the range the
   model already knows, so nothing looks unfamiliar.

3. **Do a short continued-pretraining run on long documents** after extending
   the context. Interpolation alone gets you part of the way.

4. **Be sceptical of needle-in-a-haystack tests.** Finding one planted fact in a
   long document is much easier than understanding the document. Build tests
   that need information from several places at once.

5. **Watch the KV cache grow.** Long context is mostly a memory problem, not a
   modelling problem.

## Part B — RAG

RAG stands for Retrieval-Augmented Generation. Instead of making the model read
everything, you search your corpus for the relevant passages and put only those
in the prompt.

1. **This is unusually effective for classical texts.** Your entire corpus may
   be small enough that retrieval is fast and accurate, and some individual
   works fit in context whole.

2. **Chunk by structure, not by character count.** Split at verse and chapter
   boundaries, not every 512 characters. Structure is meaningful here in a way
   it is not for web text.

3. **You need a Sanskrit embedding model.** General multilingual embedding
   models have the same weaknesses as general tokenizers. Test before trusting.

4. **Retrieval quality usually matters more than model size.** A good retriever
   plus a small model beats a large model with bad retrieval, and costs far
   less.

5. **Always cite the source passage in the output.** For scholarly use this is
   not optional. A claim without a citation is not useful to a researcher.

---

## You are ready to move on when

Your system can answer a question that requires information from two different
parts of a long text, and it shows you where it got each part.

---

:::{seealso} Related
- [Step 8](08-modern-architecture.md) — RoPE, which makes extension possible
- [Step 18](18-quantization-serving.md) — the KV cache
:::
