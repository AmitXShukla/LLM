---
title: "Step 14 — Teach it to reason"
short_title: "14. Reasoning"
---

# Step 14 — Teach it to reason

**Goal:** get the model to think through a problem before answering.

---

## Why this step matters

The key idea behind current reasoning models is simpler than it sounds.

For some tasks you can **check** the answer automatically. Maths has a right
answer. Code either passes the tests or it does not. A sandhi split is either
correct or it is not.

When you can check the answer, **you do not need a reward model at all. The
checker is the reward.**

This is called **RLVR**: Reinforcement Learning with Verifiable Rewards. Let the
model generate many attempts, keep what verifies as correct, and push the model
toward that behaviour.

The remarkable finding is what happens next. Models trained this way start
writing longer reasoning, checking their own work, and correcting their own
mistakes — without ever being explicitly taught to do any of that.

---

## Why Sanskrit is unusually well suited to this

Most languages have very few automatically checkable tasks. Sanskrit has many,
because so much of its grammar is rule-based.

This is a real structural advantage, and almost nobody has exploited it.

---

## What you do

### 1. Build a set of verifiable tasks in Sanskrit

This is where all your earlier language work pays off:

| Task | How it is checked |
|---|---|
| Sandhi splitting | Re-join the parts and compare to the original |
| Compound analysis | Compare against a reference parse |
| Metre validation | Fully rule-based, so fully automatic |
| Grammatical parsing | Compare against an annotated corpus |
| Declension and conjugation | Generate the form from the rules and compare |
| Translation round-trip | Translate out and back, compare meaning |

[Step 15](15-panini-neurosymbolic.md) is entirely about building these
checkers.

### 2. Set up the training loop

This is real engineering. You need **fast batched text generation running
inside your training loop**, because RLVR generates many attempts per question.

- Generation: vLLM or SGLang
- The RL part: `trl`, `verl`, or `OpenRLHF`

Budget real time for the plumbing. This is the most infrastructure-heavy step
in the book.

### 3. Start with GRPO

You met it in [Step 13](13-preference-tuning.md). Generate a group of answers,
score them with your checker, push toward the above-average ones.

Then read about the variants — DAPO, Dr. GRPO, GSPO, and others. Read them to
understand which specific failure each one is fixing, not to memorise a list.

### 4. Add a length penalty

Longer thinking is not free and not always better. You are trading tokens for
accuracy.

Measure that trade explicitly. A model that gets 2 percent more accurate while
producing three times as much text may be a worse product.

### 5. Decide which language the model should think in

This is an open research question and a genuinely interesting one for your
project.

Should a Sanskrit model reason **in Sanskrit**, or reason in English and answer
in Sanskrit?

Arguments both ways. Reasoning in English uses the base model's strongest
ability. Reasoning in Sanskrit keeps the whole chain in one language and may
handle grammatical reasoning better.

Test both. There is no accepted answer, so whatever you find is a contribution.

---

## Where people usually get stuck

**Underestimating the infrastructure.** RLVR needs generation and training
running together efficiently. If your generation is slow, your training is
slow, and the whole thing becomes impractical.

Get generation fast first, then start training.

---

## You are ready to move on when

Your model shows visible step-by-step working before its answer, and scores
better than the Step 13 model on your verifiable tasks.

---

:::{seealso} Related
- [Step 13](13-preference-tuning.md) — GRPO basics
- [Step 15](15-panini-neurosymbolic.md) — building the checkers
:::
