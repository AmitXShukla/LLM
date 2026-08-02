---
title: "Step 2 — Break your model on purpose"
short_title: "2. Break it"
---

# Step 2 — Break your model on purpose

**Goal:** understand what every part of your Step 1 model does — by removing it,
one piece at a time, and watching exactly what breaks.

---

## Why this step matters

In [Step 1](01-build-transformer.md) you wrote a tiny transformer by hand, and
you finished when you could explain every line out loud. This step is where you
*prove* those explanations to yourself.

You cannot truly understand a part by reading about it. You understand it by
deleting it and seeing what falls over. Remove the causal mask and the model
learns to cheat. Remove the residual connections and a deep model refuses to
learn at all. Each break turns a line you *believe* is important into a line you
*know* is important.

This takes one afternoon and is worth more than a month of tutorials. It is also
the step people skip, because it feels like going backwards. It is not. Every
hour here saves you a day of confused debugging in
[Step 9](09-training-run.md).

:::{important} 🔬 The one rule of this chapter
**Change one thing, run it, write down what happened — then put it back.**
Two changes at once and you learn nothing, because you can't tell which change
caused which effect. Keep a known-good copy of your Step 1 code and always
return to it between experiments.
:::

---

## What you do

You will run five experiments against the *exact* model from Step 1 — the same
`Head`, `Block`, `Config`, and `get_batch`. For each one: make the single change
shown, train for a few hundred steps, read the loss and a sample, and write one
short note in your own words.

Here is the whole chapter at a glance:

| # | What you break | What you'll see | What it teaches |
|---|---|---|---|
| 1 | The causal mask 🔓 | Loss crashes to ~0, samples are noise | The model cheated — it saw the answer |
| 2 | Pre-norm → post-norm 🔄 | Training gets shaky, worse when deep | Why modern models normalize *first* |
| 3 | The residual connections 🛣️ | A deep model won't learn at all | Residuals are the gradient's highway |
| 4 | The learning rate (10× too high) 🌡️ | Loss spikes up, or turns to `NaN` | What a loss spike looks like |
| 5 | The data size (tiny + long) 🧠 | Train loss → 0, val loss ↑, samples copied | Memorisation (overfitting) |

The rest of this chapter is the code for each one.

---

## 🧑‍💻 Break it with me — the exact changes

Each experiment is a *tiny* edit to the Step 1 code. The lines below are copied
straight from [`code/step-01-tiny-transformer/train_sanskrit_gpt.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-01-tiny-transformer), so you can find them in seconds.

:::{note} 📊 About the loss logs below
The numbers shown for each experiment are from a **full training run on a real
corpus** — that's when the effects are unmistakable. A ready-to-run harness,
[`code/step-02-break-it/ablations.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-02-break-it), wires all five experiments behind flags so you can reproduce them on your
own text (`--corpus your_file.txt`). On a few hundred steps of toy data the
effects are only *directional*; give each one a real corpus and enough
iterations to see the full picture.
:::

### 🔓 Experiment 1 — Remove the causal mask

This is the most important bug in this entire field. In `Head.forward`, Step 1
had this line, which hides the future before the softmax:

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # <-- Step 1: hide the future
wei = F.softmax(wei, dim=-1)
```

Break it by deleting that one line:

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
# wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # <-- BROKEN: future is visible
wei = F.softmax(wei, dim=-1)
```

**What you'll see** — the loss looks *incredible*, then the samples are garbage:

```text
iter    0: train 4.71
iter  200: train 0.08     ← too good to be true
iter  400: train 0.02
sample: ेैं् प ्ा ंे ्  ...   ← pure noise
```

:::{important} 🧠 The lesson you never forget
To predict the next akshara, the model is now allowed to *look at* the next
akshara. Copying is trivial; predicting is not. It learned to copy, so it learned
nothing. **Any time a loss looks too good to be true, check for leakage first.**
This single experiment will save you days over your career.
:::

### 🔄 Experiment 2 — Post-norm instead of pre-norm

Step 1 used **pre-norm**: the `LayerNorm` sits *inside* the `x + (...)`, so a
clean copy of `x` runs untouched from input to output.

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))   # Step 1: pre-norm
    x = x + self.ff(self.ln2(x))
    return x
```

Break it by switching to **post-norm** — the original 2017 design, where the norm
happens *after* the add:

```python
def forward(self, x):
    x = self.ln1(x + self.sa(x))   # BROKEN-ish: post-norm
    x = self.ln2(x + self.ff(x))
    return x
```

**What you'll see** — training is jumpier, and it gets clearly worse as you add
blocks (try bumping `n_layer` to 8).

:::{note} 📎 Why pre-norm wins
Pre-norm keeps an unchanged path from input to output; every block adds a small
correction onto it. Post-norm re-normalizes that path at every block, so the
signal gets disturbed over and over. You just re-derived, by hand, why nearly
every modern model normalizes first.
:::

### 🛣️ Experiment 3 — Remove the residual connections

Now remove the `x +` — the "add the input back" from Step 1's block:

```python
def forward(self, x):
    x = self.sa(self.ln1(x))   # BROKEN: no residual (dropped the `x +`)
    x = self.ff(self.ln2(x))
    return x
```

Set `n_layer = 10` so the model is genuinely deep, and train.

**What you'll see** — the loss barely moves. A deep model refuses to learn.

:::{note} 🛣️ Why residuals are non-negotiable
Without the side road, the learning signal (the gradient) has to squeeze back
through *every* block on its way to the early layers. It fades to almost nothing
before it arrives. The `x +` gives that signal a clean highway straight back —
it is the single trick that makes deep networks trainable at all.
:::

### 🌡️ Experiment 4 — Learning rate 10× too high

One number, in `Config`:

```python
class Config:
    ...
    learning_rate = 3e-4   # Step 1: a calm, sensible value
```

Break it:

```python
class Config:
    ...
    learning_rate = 3e-3   # BROKEN: 10× too high
```

**What you'll see** — instead of falling smoothly, the loss jumps *up*, or turns
into `NaN` and never recovers:

```text
iter    0: train 4.71
iter  100: train 3.10
iter  200: train 7.88     ← a loss spike
iter  300: train nan
```

:::{tip} 🌡️ Why this one matters later
You will meet a *real* loss spike in [Step 9](09-training-run.md). Because you
caused one here on purpose, you'll recognise it in one glance — "learning rate
too high" — instead of losing a day guessing.
:::

### 🧠 Experiment 5 — Overfit on purpose

Keep the model exactly as Step 1, but feed it only a *tiny* slice of text and
train for a long time. Point `get_batch` at, say, the first 100 KB of your
corpus and raise `max_iters`.

Watch **both** losses — Step 1's `estimate_loss` already prints train and val:

```text
iter    0: train 4.70  val 4.70
iter 2000: train 0.40  val 3.10   ← the gap opens
iter 5000: train 0.05  val 3.60   ← train keeps falling, val climbs
sample: (a verse copied word-for-word from your file)
```

:::{warning} 🧠 Memorisation, and why Sanskrit makes it worse
Training loss near zero while validation loss *rises* is **overfitting**: the
model stopped learning patterns and started memorising your exact file. This is
especially dangerous for Sanskrit — your data is small, and the *same* verses
appear across many sources, so duplicates sneak in and inflate how good things
look. You'll fix this properly in [Step 7](07-clean-data.md). The skill here is
simply *seeing* the train/val gap open.
:::

:::{tip} ⚙️ Make them flags, not hand-edits
Editing code back and forth is error-prone (it's easy to forget to restore the
mask!). Wire each experiment behind a flag — `--no-mask`, `--post-norm`,
`--no-residual`, `--lr`, `--tiny` — so you can run all five in an afternoon
without ever losing your good model. A ready-made version is in
[`code/step-02-break-it/ablations.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-02-break-it).
:::

---

## What you should see

Put side by side, the five failures each have a *distinct signature*. Learning to
read these signatures is the real skill:

- **Loss too low, output noise** → leakage (you're seeing the answer).
- **Loss jumpy, worse when deep** → a normalization or depth problem.
- **Loss flat on a deep model** → the gradient can't reach the early layers.
- **Loss spikes up or `NaN`** → learning rate too high.
- **Train down, val up** → memorisation.

---

## Where people usually get stuck

**Changing two things at once.** Then a result appears and you can't say which
change caused it. One change, one run, one note.

**Forgetting to revert.** Especially the causal mask. Before every new
experiment, restore your known-good Step 1 model. (This is exactly why the flag
version above is worth the ten minutes to set up.)

**Not writing it down.** The five notes *are* the output of this step — not the
training runs. If you skip the notes, you did the experiments and kept none of
the understanding.

---

## You are ready to move on when

You have five short notes, one per experiment, in your own words — and your model
is back to the correct Step 1 version.

A good test: hand the notes to another engineer. If they learn something from
your five sentences, you did this properly.

---

:::{seealso} Related
- [Step 1](01-build-transformer.md) — the model you are breaking
- [Step 7](07-clean-data.md) — where overfitting gets fixed for real
- [Step 9](09-training-run.md) — where these failures show up in a real training run
:::
