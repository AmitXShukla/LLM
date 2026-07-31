---
title: "Step 1 — Build a tiny Sanskrit transformer"
short_title: "1. Tiny transformer"
---

# Step 1 — Build a tiny Sanskrit transformer by hand

**Goal:** write a small character-level language model from a blank file, with
no `transformers` import.

---

## Why this step matters

A language model does one thing: it looks at some text and guesses the next
piece. Everything else — all the size, all the tricks, all the products built
on top — sits on that one idea.

You need to feel this in your hands before anything else makes sense.

A **character-level** model means the model works with single letters instead
of words. It is the simplest possible version. It is also slow and weak, and
you will replace it in [Step 3](03-tokenizers.md). That is fine. Right now you
are learning, not building a product.

:::{important} Time-box this
Give this step two days. No more. The goal is understanding, not a good model.
If you find yourself tuning hyperparameters to squeeze out a better loss, stop
— you have finished the useful part and started the useless part.
:::

---

## What you do

### 1. Get a small Sanskrit text file

A few megabytes is plenty. The Ramayana, the Bhagavad Gita, or a Sanskrit
Wikipedia dump all work. See the
[corpora appendix](appendix/corpora.md) for places to get text.

Use *your own* text if you have some. That matters, because it forces you to
meet Sanskrit's real problems on day one instead of reading about them later.

### 2. Build a character vocabulary

List every unique character in your file. Give each one a number. Write a
function that turns text into numbers, and one that turns numbers back into
text.

Look at how many unique characters you got. Devanagari will give you more than
you expect, because vowel signs, the virama, and conjunct forms all count
separately. This is your first small hint that tokenization is going to be the
real problem.

### 3. Write a data loader

Pick a random point in the text. Take the next 128 characters as the input.
Take the same 128 characters shifted along by one as the answer.

That shift is the entire training task. The model sees `त`, and must predict
`त्`. Then it sees `तत्`, and must predict the space. And so on.

### 4. Write single-head self-attention by hand

For each position, the model produces three vectors: a **query**, a **key**,
and a **value**.

Here is the intuition. Each character asks a question — that is the query.
Every earlier character offers a label describing what it is — that is the key.
The model compares the question against all the labels, and mixes together the
actual content — the values — of whichever earlier characters matched best.

That is it. Attention is a weighted average, where the weights are learned by
matching questions against labels.

### 5. Add the causal mask

The model must not look at future characters.

If it can see the answer, it will simply copy it. It will look like it is
learning brilliantly and it will have learned nothing. The mask covers the
future so this cannot happen.

You will prove this to yourself in [Step 2](02-break-it.md).

### 6. Add multiple heads

Run several attention operations side by side and join the results together.
Different heads learn to look for different things: one might track word
boundaries, another might track long-range agreement.

### 7. Build the full block

One block is:

1. Normalize
2. Attention
3. Add the input back in
4. Normalize
5. A small feed-forward network
6. Add the input back in again

Steps 3 and 6 are called **residual connections**. They give information a side
road around the block. This is what lets you stack many blocks without the
learning signal fading away.

### 8. Stack four to six blocks, and add a training loop

Use the AdamW optimizer and a cosine learning rate schedule. Keep it simple.

### 9. Write a sampling function

You need to be able to read what your model produces. Add temperature and
top-k sampling so you can control how adventurous it is.

### 10. Train it, and read the output

---

## What you should see

The loss goes down. The samples look like Sanskrit at a glance. They have the
right letters, roughly the right word lengths, and roughly the right rhythm.

They mean nothing at all.

**This is the correct result.** Your model is tiny and your data is tiny. It has
learned what Sanskrit *looks like*, not what Sanskrit *says*. The rest of this
book is about closing that gap.

If your samples look like meaningless but plausible Sanskrit, you have
succeeded. Do not try to fix it. It is not broken.

---

## Where people usually get stuck

**Shape errors in the attention code.** Print the shape of every tensor on
every line until it works. Everyone does this, including people who have
written transformers before. It is not a sign that you are bad at this.

**Trying to make the model good.** It cannot be good. Move on.

---

## You are ready to move on when

You can explain, out loud, what every line of your model does and why it is
there.

Not "this is the attention layer." More like: "this line divides by the square
root of the head dimension, because without it the dot products get large when
the dimension is large, and then softmax turns into a hard maximum and the
gradients vanish."

---

:::{seealso} Related
- [Step 2](02-break-it.md) — break what you just built
- [Should you build from scratch at all?](appendix/scratch-or-finetune.md) — the honest debate
- [Things worth reading](appendix/reading.md) — nanoGPT and other starting points
:::

---

## 🧑‍💻 Build it with me — the complete, annotated model

The steps above told you *what* to do. Now here is the actual, runnable code,
broken into pieces with the *why* next to each one. The full file is
[`code/step-01-tiny-transformer/train_sanskrit_gpt.py`](https://github.com/AmitXShukla/LLM/tree/main/code/step-01-tiny-transformer) — run `python train_sanskrit_gpt.py --smoke` for a 30-second sanity run.

:::{note} 🧭 One idea to hold onto
The model has exactly one job: **given some aksharas, predict the next one.** A
chatbot, a translator, a reasoning model — all of them are elaborations of this
one trick. Nail this and the rest of the book compounds.
:::

### 📥 The data is the task (`get_batch`)

We turn the whole corpus into one long list of token IDs. Training data is just
pairs of windows: `x` is a window of aksharas, and `y` is the *same* window
shifted right by one. So for every position, the "correct answer" is literally
the next akshara. That shift-by-one is the entire supervision signal — no labels,
no annotation. The text labels itself. This is why it is called *self*-supervised.

```python
def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i:i + cfg.block_size]         for i in ix])
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])  # answer = input shifted by 1
    return x.to(device), y.to(device)
```

### 🎛️ The knobs (`Config`)

These defaults are a sensible weekend starting point. Scale them **up** on a real
GPU once your corpus is bigger.

```python
class Config:
    block_size = 128   # context length: how many aksharas the model sees at once
    n_embd     = 256   # width of the model (embedding dimension)
    n_head     = 4     # number of attention heads (n_embd must divide by this)
    n_layer    = 4     # number of transformer blocks stacked on top of each other
    dropout    = 0.1
    batch_size = 32
    max_iters  = 3000
    learning_rate = 3e-4
```

### 🔎 Attention — the one genuinely new idea (`Head`)

This is the heart. Everything else is supporting cast. Each token produces three
vectors:

- **query** — "what am I looking for?"
- **key** — "what do I contain / advertise?"
- **value** — "what will I actually contribute if you pick me?"

To decide how much position *i* should listen to position *j*, we take the dot
product of *i*'s query with *j*'s key. Big dot product = "these two are relevant."
Softmax turns those affinities into weights that sum to 1, and we use them to take
a **weighted average of the value vectors**.

```python
class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # lower-triangular mask so position t can only attend to <= t (causal)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5     # scaled affinities (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # can't see the future
        wei = F.softmax(wei, dim=-1)                            # weights sum to 1
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v                                          # weighted sum of values
```

:::{important} 🧠 Two small details that matter a lot
**The `* head_size**-0.5` scaling.** Without it, dot products grow large as
vectors widen, softmax saturates into a near one-hot spike, and gradients die.
Dividing by √(head_size) keeps things healthy — this is the "scaled" in *scaled
dot-product attention*.

**The causal mask.** A language model must predict the future from the past only —
it can't peek at the answer. We set the future entries to `-inf` *before* softmax,
so they become 0 after. Position 5 can see 0–5, never 6+.
:::

### 👀 Many heads at once (`MultiHeadAttention`)

One head learns one kind of relationship; several run in parallel so the model can
track several at once. Maybe one head learns to bind a vowel sign to its
consonant, another learns that a *danda* (`।`) ends a clause. We never assign
these roles — the heads discover them.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))
```

### 🧮 The little thinker (`FeedForward`)

Attention moves information *between* positions. The MLP then lets each position
*think* about what it just received, on its own. Widen to 4×, apply a
non-linearity, shrink back.

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x):
        return self.net(x)
```

:::{tip} 🔁 The rhythm of a transformer
**Attention = communication. MLP = computation.** A block alternates the two.
That is the whole pattern, repeated `n_layer` times.
:::

### 🧱 One block (`Block`) — residuals, LayerNorm, pre-norm

```python
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.sa  = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff  = FeedForward(cfg)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # communicate, then add back (residual)
        x = x + self.ff(self.ln2(x))   # compute, then add back (residual)
        return x
```

- **`x + ...` (residual).** Each sub-layer *adds a correction* instead of
  replacing `x`. This gives gradients a clean highway back to the early layers —
  it's what makes deep networks trainable at all.
- **LayerNorm.** Re-centres and re-scales each vector so numbers stay stable
  through many layers.
- **Pre-norm.** We normalise *before* each sub-layer (the LayerNorms sit *inside*
  the `x + (...)`). It trains more smoothly than the original post-norm design.

### 🏗️ The whole model (`SanskritGPT`)

Embeddings at the bottom, a stack of blocks in the middle, a prediction over the
vocabulary at the top. Note the **two** embeddings: one for *what* each akshara is,
one for *where* it sits — because attention is order-blind unless we hand it
position.

```python
class SanskritGPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.token_embedding    = nn.Embedding(vocab_size, cfg.n_embd)     # what
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd) # where
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)                   # score per akshara

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos                 # meaning + position
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss
```

:::{note} 📉 What the loss means
Cross-entropy is, intuitively, *"how surprised was the model by the right
answer?"* Confident and correct → low loss. Confident and wrong → high loss.
Training is nothing but nudging the weights to be less surprised by real Sanskrit.
A sanity check: an untrained model's loss should be about `ln(vocab_size)` — pure
guessing. Watching it fall below that is your proof learning is happening. ✅
:::

### 🎲 Making it talk (`generate`)

To sample, we feed the context, look only at the logits for the *last* position,
and pick a next token. Three knobs shape the output: **temperature** (lower = safer
and more repetitive, higher = more adventurous), **top-k** (only consider the k
most likely tokens), and `multinomial` (sample, so output varies each run).

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.cfg.block_size:]      # never feed more than block_size
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature        # focus on the last step
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

That's the entire model — a few dozen lines. Training it (the loop, the
train/val split, watching the loss) is [Step 9](09-training-run.md). The tokenizer
that feeds it — and the Sanskrit "gotcha" that makes this project interesting — is
[Step 4](04-sanskrit-tokenizer.md).

:::{seealso} 📚 Follow-along resources
- 🧠 The line-by-line teaching notes: [`docs/notes/weekend1-tiny-transformer-teaching.md`](https://github.com/AmitXShukla/LLM/tree/main/docs/notes/weekend1-tiny-transformer-teaching.md)
- ✍️ The story of building it: [`docs/notes/weekend1-blog-tiny-sanskrit-gpt.md`](https://github.com/AmitXShukla/LLM/tree/main/docs/notes/weekend1-blog-tiny-sanskrit-gpt.md)
:::
