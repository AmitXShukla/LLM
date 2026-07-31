---
title: "Step 17 — Distillation"
short_title: "17. Distillation"
---

# Step 17 — Distillation

:::{note} Chapter status
Outline. To be expanded.
:::

**Goal:** get a small model to behave like a large one.

---

## Why this step matters

Distillation means a large **teacher** model trains a small **student** model.
It is often the cheapest way to get good quality at small size.

For your project it has a specific use: take a large multilingual model that
knows a little Sanskrit among a hundred other things, and concentrate that
knowledge into a small model that only does Sanskrit. That is exactly the
"depth over breadth" argument from the [introduction](intro.md), made concrete.

---

## What you do

### 1. Understand the three kinds

**Logit distillation.** The student learns to match the teacher's full
probability distribution, not just its final choice. This carries much more
information: "70 percent this word, 20 percent that one, 10 percent a third"
teaches far more than "this word".

**Sequence distillation.** The teacher writes out full answers, and the student
trains on them as ordinary text. Simple and effective.

**On-policy distillation.** The student produces its own answers, and the
teacher scores or corrects them. Usually the strongest of the three, because
the student gets feedback on its own actual mistakes rather than on problems it
never had.

### 2. Try reasoning distillation

Have a large reasoning model write out its full thinking for your Sanskrit
tasks. Train your small model on those traces.

This is how most small reasoning models are made, and it works surprisingly
well.

### 3. Distil a multilingual teacher into a Sanskrit-only student

Directly your project's thesis. Measure whether the small specialist beats the
large generalist on your Sanskrit tests. It often does.

### 4. Read the licence

Many model licences restrict using their outputs to train other models. Check
before you build on it, not after you have released something.

---

## You are ready to move on when

You have a small model that scores close to a much larger one on your Sanskrit
tests.

---

:::{seealso} Related
- [Step 15](15-panini-neurosymbolic.md) — rule-checked traces make better teachers
- [Step 18](18-quantization-serving.md) — the other way to get small
:::

---

## 🧑‍💻 Runnable code for this step

Distillation = train a small **student** on a big **teacher's** outputs. For
reasoning, that means SFT on the teacher's `<think>` traces (keep only the ones
the verifier says are correct):

```python
# one line of a distillation set — the teacher's full reasoning + answer
{"prompt": "A dose is 5 mg/kg for a 12 kg child. Total dose?",
 "completion": "<think>5 mg/kg × 12 kg = 60 mg. Units: mg/kg × kg = mg.</think>\n\\boxed{60 mg}"}
```

Then it's just SFT (Step 12). See [`code/step-14-reasoning/distill_dataset_example.jsonl`](https://github.com/AmitXShukla/LLM/tree/main/code/step-14-reasoning).

:::{tip} The pragmatic default 💡
For most teams and most hardware, **distillation beats DIY reinforcement
learning**: cheaper, stable, and often *better* for small models. DeepSeek showed
distilling R1 into small Qwen/Llama models beat running RL on them directly.
:::

The same idea shrinks a big ECG or X-ray model into a compact one for a wearable
or edge device — a "teacher → student" transfer that keeps most of the accuracy at
a fraction of the size.
