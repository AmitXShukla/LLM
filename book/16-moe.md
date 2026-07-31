---
title: "Step 16 — Mixture of Experts"
short_title: "16. Mixture of Experts"
---

# Step 16 — Mixture of Experts

:::{note} Chapter status
Outline. To be expanded.
:::

**Goal:** understand the architecture that most large models now use.

---

## Why this step matters

A Mixture of Experts (MoE) model replaces one big feed-forward layer with many
smaller **expert** layers, plus a small **router** that picks only two or three
of them for each token.

Think of a company with fifty specialists. For any one job you only call in two
of them. You get the knowledge of fifty people at the cost of consulting two.

:::{important} This is a pretraining decision
MoE is decided **before** you train. It is not something you add afterwards.

It sits here in the book because it is an efficiency idea, but in a real project
you would decide it back at [Step 8](08-modern-architecture.md).
:::

---

## What you do

1. **Understand total versus active parameters.** A model may have 30 billion
   parameters in total but use only 3 billion for any given token. Both numbers
   matter, for different reasons: total decides your memory, active decides your
   speed.

2. **Understand routing.** A small network looks at each token and picks the top
   few experts for it.

3. **Understand load balancing.** Left alone, the router sends everything to a
   few favourite experts and the rest never learn anything. An auxiliary loss —
   or newer loss-free balancing methods — prevents this.

4. **Understand shared experts.** Some designs keep one expert that every token
   always uses, for the general knowledge, while the routed experts specialise.

5. **Understand the honest cost.** MoE saves compute but **not memory**. Every
   expert has to sit in memory even though most are idle at any moment. Serving
   is also more complex, because experts may live on different devices.

6. **Know when not to use it.** For a single-GPU model, MoE is usually the
   wrong choice. Say so plainly in your write-up.

7. **Learn about upcycling.** You can turn an already-trained dense model into
   an MoE by copying its feed-forward layer into several experts and then
   training the router. Cheaper than starting over.

---

## You are ready to move on when

You can explain in two sentences why MoE helps and what it costs.

---

:::{seealso} Related
- [Step 8](08-modern-architecture.md) — where this decision really belongs
- [Step 18](18-quantization-serving.md) — why the memory cost matters
:::
