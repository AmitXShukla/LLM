---
title: "Step 2 — Break your model on purpose"
short_title: "2. Break it"
---

# Step 2 — Break your model on purpose 🔬

**Goal:** understand what every part of your Step 1 model actually does — by
removing it, one piece at a time, and watching exactly what breaks.

---

## Why this step matters

In [Step 1](01-build-transformer.md) you wrote a tiny transformer by hand and
finished when you could explain every line out loud. This step is where you
*prove* those explanations to yourself.

You can't truly understand a part by reading about it. You understand it by
deleting it and seeing what falls over. Remove the causal mask and the model
learns to cheat. Remove the residual connections and a deep model refuses to
learn at all. Each break turns a line you *believe* matters into a line you
*know* matters.

This takes one afternoon and is worth more than a month of tutorials. It's also
the step people skip, because it feels like going backwards. It isn't. Every hour
here saves you a day of confused debugging in [Step 9](09-training-run.md).

:::{tip} 🖥️ Run the experiments as you read
Each experiment is a one-line change to your Step 1 code. A ready-made harness
wires all five behind flags, so you never have to edit by hand (or forget to put
the mask back):
```bash
pip install torch regex
python ablations.py --no-mask        # 1: remove the causal mask
python ablations.py --post-norm      # 2: normalize after the block
python ablations.py --no-residual --n-layer 10   # 3: drop residuals
python ablations.py --lr 3e-3        # 4: learning rate 10× too high
python ablations.py --tiny           # 5: overfit tiny data
```
Full file: [`code/step-02-break-it/ablations.py`](https://github.com/AmitXShukla/LLM/blob/main/code/step-02-break-it/ablations.py).
:::

:::{important} 🔬 The one rule of this chapter
**Change one thing → run → write down what happened → put it back.** Two changes
at once and you learn nothing, because you can't tell which change caused which
effect. Keep a known-good copy of your Step 1 code and return to it between
experiments.
:::

```{mermaid}
flowchart LR
    A["edit ONE line"] --> B["run"] --> C["read train &amp; val loss"] --> D["write one note"] --> E["put the line back"] --> A
```

---

## The shape of this chapter 🗺️

Five experiments, against the *exact* model from Step 1 — the same `Head`,
`Block`, `Config`, and `get_batch`. Each one breaks a different part and leaves a
different fingerprint:

| # | What you break | What you'll see | What it teaches |
|---|---|---|---|
| 1 | The causal mask 🔓 | Loss crashes to ~0, samples are noise | The model cheated — it saw the answer |
| 2 | Pre-norm → post-norm 🔄 | Training gets shaky, worse when deep | Why modern models normalize *first* |
| 3 | The residual connections 🛣️ | A deep model won't learn at all | Residuals are the gradient's highway |
| 4 | The learning rate (10× too high) 🌡️ | Loss spikes up, or turns to `NaN` | What a loss spike looks like |
| 5 | The data size (tiny + long) 🧠 | Train loss → 0, val loss ↑, samples copied | Memorisation (overfitting) |

:::{note} 📊 About the loss logs below
The numbers shown for each experiment are what you see on a **full training run on
a real corpus** — that's when the effects are unmistakable. On a few hundred steps
of toy data they're only *directional*. Give each one a real corpus (the
`--corpus` flag) to see the full picture.
:::

The rest of this chapter is the five experiments, in the same order.

---

## 1 · Remove the causal mask 🔓

This is the most important bug in the entire field. In `Head.forward`, Step 1 had
this line — it hides the future before the softmax:

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # <-- Step 1: hide the future
wei = F.softmax(wei, dim=-1)
```

**Break it** by deleting that one line:

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
# wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # <-- BROKEN: future visible
wei = F.softmax(wei, dim=-1)
```

Now every position can peek at the answer sitting one step ahead:

```{mermaid}
flowchart LR
    subgraph with_mask["with mask ✓"]
    A["position 5"] --> B["sees only 0–5"]
    end
    subgraph no_mask["no mask ✗"]
    C["position 5"] --> D["sees position 6 = the answer"]
    end
```

**What you'll see** — the loss looks *incredible*, then the samples are garbage:

```text
iter    0: train 4.71
iter  200: train 0.08     <- too good to be true
iter  400: train 0.02
sample: े ैं ् प ्ा ...    <- pure noise
```

:::{important} 🧠 The lesson you never forget
To predict the next akshara, the model is now allowed to *look at* the next
akshara. Copying is trivial; predicting is not. It learned to copy, so it learned
nothing. **Any time a loss looks too good to be true, check for leakage first.**
This one experiment will save you days over your career.
:::

---

## 2 · Normalize after, not before 🔄

Step 1 used **pre-norm**: the `LayerNorm` sits *inside* the `x + (...)`, so a clean
copy of `x` runs untouched from input to output.

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))   # Step 1: pre-norm
    x = x + self.ff(self.ln2(x))
    return x
```

**Break it** by switching to **post-norm** — the original 2017 design, where the
norm happens *after* the add:

```python
def forward(self, x):
    x = self.ln1(x + self.sa(x))   # post-norm
    x = self.ln2(x + self.ff(x))
    return x
```

**What you'll see** — training is jumpier, and it gets clearly worse as you stack
more blocks (try `n_layer = 8`).

:::{note} 📎 Why pre-norm wins
Pre-norm keeps one unchanged path from input to output; every block just adds a
small correction onto it. Post-norm re-normalizes that path at every block, so the
signal gets disturbed over and over. You just re-derived, by hand, why nearly
every modern model normalizes first.
:::

---

## 3 · Remove the residual connections 🛣️

Now remove the `x +` — the "add the input back" from Step 1's block:

```python
def forward(self, x):
    x = self.sa(self.ln1(x))   # BROKEN: no residual (dropped the `x +`)
    x = self.ff(self.ln2(x))
    return x
```

Set `n_layer = 10` so the model is genuinely deep, and train.

**What you'll see** — the loss barely moves. A deep model refuses to learn:

```text
iter    0: train 4.71
iter 1000: train 4.55
iter 3000: train 4.51     <- barely moves
```

```{mermaid}
flowchart LR
    subgraph with_res["with residual ✓"]
    G1["gradient"] --> H["straight highway to the early layers"]
    end
    subgraph no_res["no residual ✗"]
    G2["gradient"] --> B1["block"] --> B2["block"] --> B3["…fades to nothing"]
    end
```

:::{note} 🛣️ Why residuals are non-negotiable
Without the side road, the learning signal (the gradient) has to squeeze back
through *every* block on its way to the early layers. It fades to almost nothing
before it arrives. The `x +` gives it a clean highway straight back — it's the
single trick that makes deep networks trainable at all.
:::

---

## 4 · Learning rate 10× too high 🌡️

One number, in `Config`:

```python
class Config:
    ...
    learning_rate = 3e-4   # Step 1: a calm, sensible value
```

**Break it:**

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
iter  200: train 7.88     <- a loss spike
iter  300: train nan
```

:::{tip} 🌡️ Why this one matters later
You'll meet a *real* loss spike in [Step 9](09-training-run.md). Because you caused
one here on purpose, you'll recognise it in one glance — "learning rate too high" —
instead of losing a day guessing.
:::

---

## 5 · Overfit on purpose 🧠

Keep the model exactly as Step 1, but feed it only a *tiny* slice of text and train
for a long time. Point the data at, say, the first 100 KB and raise `max_iters`.

Watch **both** losses — Step 1's `estimate_loss` already prints train and val:

```text
iter    0: train 4.70  val 4.70
iter 2000: train 0.40  val 3.10     <- the gap opens
iter 5000: train 0.05  val 3.60     <- train falls, val CLIMBS
sample: (a verse copied word-for-word from your file)
```

Training loss near zero while validation loss *rises* is the classic fingerprint:

```{mermaid}
flowchart LR
    T["train loss"] --> D["falls toward 0 ↓"]
    V["val loss"] --> U["climbs back up ↑"]
    D --> M["the gap = memorisation"]
    U --> M
```

:::{warning} 🧠 Memorisation, and why Sanskrit makes it worse
Train loss falling while validation loss *rises* is **overfitting**: the model
stopped learning patterns and started memorising your exact file. This is
especially dangerous for Sanskrit — your data is small, and the *same* verses
appear across many sources, so duplicates sneak in and inflate how good things
look. You'll fix this properly in [Step 7](07-clean-data.md). The skill here is
simply *seeing* the train/val gap open.
:::

---

## What you should see ▶️

Put side by side, the five failures each have a *distinct signature*. Learning to
read these fingerprints is the real skill of this chapter:

- 🔓 **Loss too low + output noise** → leakage (you're seeing the answer).
- 🔄 **Loss jumpy, worse when deep** → a normalization or depth problem.
- 🛣️ **Loss flat on a deep model** → the gradient can't reach the early layers.
- 🌡️ **Loss spikes up or `NaN`** → learning rate too high.
- 🧠 **Train down, val up** → memorisation.

---

## Where people usually get stuck

**Changing two things at once.** Then a result appears and you can't say which
change caused it. One change, one run, one note.

**Forgetting to revert** — especially the causal mask. Before every new
experiment, restore your known-good Step 1 model. (This is exactly why the flag
harness above is worth the ten minutes to set up.)

**Not writing it down.** The five notes *are* the output of this step — not the
training runs. Skip the notes and you did the experiments but kept none of the
understanding.

---

## You are ready to move on when

You have five short notes, one per experiment, in your own words — and your model
is back to the correct Step 1 version.

A good test: hand the notes to another engineer. If they learn something from your
five sentences, you did this properly. Then head to
[Step 3](03-tokenizers.md), where we finally fix the real problem — how we chop
Sanskrit and Urdu into pieces.

---

:::{seealso} 📚 Related
- [Step 1](01-build-transformer.md) — the model you are breaking
- [Step 7](07-clean-data.md) — where overfitting gets fixed for real
- [Step 9](09-training-run.md) — where these failures show up in a real training run
- 📄 The flag harness: [`code/step-02-break-it/ablations.py`](https://github.com/AmitXShukla/LLM/blob/main/code/step-02-break-it/ablations.py)
:::
