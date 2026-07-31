---
title: "Step 13 — Teach it what a good answer looks like"
short_title: "13. Preference tuning"
---

# Step 13 — Teach it what a good answer looks like

**Goal:** move the model from "correct" to "actually preferred by people".

---

## Why this step matters

[Step 12](12-instruction-tuning.md) taught the model one right answer per
question. But usually many answers are correct, and some are much better than
others.

Preference training teaches the model which one people would choose.

---

## What you do

### 1. Understand the older approach first (RLHF)

Reinforcement Learning from Human Feedback works like this:

1. Collect pairs of answers.
2. Ask humans which one is better.
3. Train a separate **reward model** to predict those human choices.
4. Use reinforcement learning to make your model maximise that reward.

It works. It is also complex, expensive, and has a lot of moving parts.

### 2. Learn DPO

Direct Preference Optimization has a neat insight: **you can skip the separate
reward model entirely.**

Take your pairs of "better answer, worse answer" and train the model directly
to raise the probability of the better one and lower the probability of the
worse one. That is it — one loss function, no reward model, no reinforcement
learning loop.

Much simpler, much cheaper, and it works well. Start here.

There is a family of related methods — IPO, KTO, ORPO, SimPO — each patching a
different weakness. Read them for the weaknesses, not the names.

### 3. Learn GRPO

Group Relative Policy Optimization works differently:

1. For each question, generate a **group** of answers, maybe eight or sixteen.
2. Score them all.
3. Push the model toward the ones that scored above the group's average.

Simple, and it needs no value network. It has become the default in reasoning
work, and you will use it directly in [Step 14](14-reasoning.md).

### 4. Watch for reward hacking

The model will find ways to score well without being better.

The most common: it learns that longer answers score higher, so answers get
longer and emptier. Also watch for **sycophancy**, where the model learns to
agree with the user because agreement was preferred in the training data.

For a Sanskrit model doing translation or philosophical analysis, sycophancy is
genuinely dangerous. A model that agrees with a wrong reading because the user
suggested it is worse than useless.

### 5. Build preference data for a language with no annotator pool

This is honestly hard for Sanskrit. Practical options:

- Use scholars where you can afford them, even for a small set
- Use **verifiable tasks** where correctness is automatically checkable — see
  [Step 14](14-reasoning.md) and [Step 15](15-panini-neurosymbolic.md)
- Use a strong multilingual model as a judge, while treating its output with
  suspicion and spot-checking by hand

The verifiable route is the most promising for Sanskrit, and it is the reason
Step 15 exists.

---

## Where people usually get stuck

**The loss goes down beautifully and the model gets worse.**

Always sample and read actual outputs during preference training. The loss
number lies here more than anywhere else in this book.

---

## You are ready to move on when

Human reviewers prefer your Step 13 model over your Step 12 model, on real
examples, without knowing which is which.

---

:::{seealso} Related
- [Step 14](14-reasoning.md) — GRPO applied to reasoning
- [Step 15](15-panini-neurosymbolic.md) — where your verifiable rewards come from
:::

---

## 🧑‍💻 Runnable code for this step

Preference tuning teaches the model to prefer the *better* of two answers.
**DPO** does this with no reward model and no RL loop — just a `chosen` vs.
`rejected` pair:

```python
# one line of prefs.jsonl
{"prompt": "Explain 'karma' briefly.",
 "chosen":   "Karma is the principle that intentional actions carry moral consequences...",
 "rejected": "karma is a hindu thing about luck idk"}
```

```python
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(output_dir="./dpo-adapter", beta=0.1,   # beta = how hard to push vs. reference
                learning_rate=5e-6, num_train_epochs=1, bf16=True, report_to="none")
DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok).train()
```

:::{important} Order matters ➡️
Do **SFT first, then DPO.** DPO assumes a model that already gives reasonable
answers; running it on a raw base usually disappoints. Note DPO's very small
learning rate (`5e-6`) compared to SFT.
:::
