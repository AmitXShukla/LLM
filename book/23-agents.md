---
title: "Step 23 — Agents and tools"
short_title: "23. Agents and tools"
---

# Step 23 — Agents and tools

:::{note} Chapter status
Outline. To be expanded.
:::

**Goal:** let the model use tools and take actions instead of just producing
text.

---

## Why this step matters

An **agent** is a model that can call tools, look at the results, and decide
what to do next. It thinks, acts, observes, and repeats.

For Sanskrit this is a natural fit, because you built the tools in
[Step 15](15-panini-neurosymbolic.md). A model that can call a sandhi splitter,
a dictionary, and a metre checker is far more capable than one that has to know
everything itself.

---

## What to cover

1. **Function calling.** Getting the model to produce valid, structured tool
   calls reliably. Constrained decoding forces the output to match a schema.

2. **The think-act-observe loop.** The basic agent pattern.

3. **Multi-turn reinforcement learning.** Note that plain GRPO from
   [Step 13](13-preference-tuning.md) assumes a single turn, which breaks here.
   Multi-turn extensions exist and are an active research area.

4. **Memory and context management** across many steps.

5. **Evaluating agents on task completion**, not on whether the response reads
   nicely. This is a genuinely different kind of evaluation.

---

## A concrete Sanskrit example

An agent that helps a student work through a difficult verse:

```
1. Split the sandhi              → calls your Step 15 splitter
2. Look up each word             → calls a dictionary
3. Identify grammatical forms    → calls your parser
4. Check the metre               → calls your metre checker
5. Propose a translation         → the neural model's own work
6. Verify the translation is
   consistent with the parse     → back to the tools
```

Steps 1 to 4 and step 6 are deterministic and checkable. Only step 5 needs the
neural model. That division is the whole point.

---

:::{seealso} Related
- [Step 15](15-panini-neurosymbolic.md) — the tools this agent calls
:::
