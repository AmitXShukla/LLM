---
title: Should you build from scratch at all?
short_title: Scratch or fine-tune?
---

# Should you build from scratch at all?

An honest look at a real disagreement, because you deserve to see both sides
rather than be told there is one answer.

---

## The disagreement

Ask three experienced people whether a competent engineer should build a small
transformer from scratch before fine-tuning a large one, and you will get three
different answers.

### "Skip it. Go straight to fine-tuning."

The argument: if you build a GPT-2 replica, you are implementing an **outdated
architecture**. Modern models use RoPE, RMSNorm, SwiGLU, and grouped-query
attention. You will touch none of them.

Meanwhile fine-tuning a modern model forces you to engage with the things that
actually matter now:

- Its tokenizer, and how it mangles your language
- FlashAttention, gradient checkpointing, and the real memory bottleneck
- LoRA target modules — `q_proj`, `gate_proj`, and the rest — which requires
  knowing what each projection matrix does
- Distributed training, precision, and serving

**The conclusion:** you get your architectural refresher in the first few hours
of reading a modern model's config, and you get a useful model at the end.

### "Do it. Fine-tuning skips the part you wanted."

The argument: fine-tuning teaches you a **workflow**, not a mechanism. The
transformer stays a box you call into. Attention, the backward pass, why the
loss drops — none of it gets exercised.

And for a low-resource language specifically, the tokenizer and data pipeline
work is where the real pain lives. **Feeling that pain yourself on day one is
worth more than any tutorial**, and the pipeline you build is directly reusable
later.

**The conclusion:** build it, but treat it as education, not as a product.

### "Time-box it. Two days, no more."

The middle position, and the one this book takes.

---

## What this book does, and why

**Steps 1 and 2 are time-boxed to two days.** Build it, break it, learn what
each part does, move on. Do not tune it. Do not try to make it good.

**Step 8 immediately rebuilds it with the modern architecture.** This is the
part that answers the "you will learn an outdated design" objection directly. If
you stopped at Step 2, that objection would be completely correct.

**Steps 3 to 7 are where the real value is anyway.** Tokenization and data.
These transfer perfectly to the fine-tuning path — the tokenizer you build in
Step 4 is grafted onto a large model in Step 11.

**Step 11 is where the useful model actually gets built.**

So the honest summary: **the from-scratch build is worth two days and no more,
and its main value is the tokenizer and data work, not the model.**

---

## If you are genuinely short on time

Do this compressed version:

1. **Step 0** — set up properly. Do not skip this one.
2. **Steps 3 to 5** — spend four to six hours on tokenizers using an existing
   minimal repository. This is roughly 60 to 70 percent of the total learning
   benefit for a low-resource language.
3. **Step 6** — do the data audit. You must know your token count before you can
   plan anything.
4. **Step 11** — jump straight to adapting a base model.

Come back for Steps 1, 2, 8, 9, and 10 later. They will make more sense once you
have hit a real problem that needs them.

---

## The one thing everyone agrees on

Every version of this argument agrees on one point:

**Tokenization is the central technical problem for Sanskrit, not a side
detail.**

Existing tokenizers shred Devanagari — conjuncts, sandhi, compounds — into
nonsense, and that is a major reason off-the-shelf models are poor at Sanskrit.

Whatever path you take, do not skip [Steps 3 to 5](../03-tokenizers.md).
