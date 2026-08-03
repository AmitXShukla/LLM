---
title: "Step 1 — Build a tiny Sanskrit transformer, by hand"
short_title: "1. Build a transformer"
---

# Step 1 — Build a tiny Sanskrit transformer, by hand 🧠

**Goal:** write a small GPT from a blank file — no `transformers` import — and
understand *every single line*, from the `Config` at the top to the `main()`
that ties it all together and trains it on Sanskrit.

---

## Why this step matters

A language model does exactly one thing: **it looks at some text and guesses the
next little piece.** Everything else in this book — tokenizers, fine-tuning,
reasoning, deployment — is built on top of that one idea. So before we touch any
big libraries, we build the whole thing ourselves, in about 250 lines, and read
it line by line.

We work at the level of the **akshara** — a Sanskrit syllable — instead of the
word. It's the simplest possible version: slow, weak, and you'll replace it
later. That's the point. Time-box this to a weekend, and finish when you can
explain any line out loud.

:::{tip} 🖥️ Read this chapter with the code open
Everything below is the real file,
[`code/step-01-tiny-transformer/train_sanskrit_gpt.py`](https://github.com/AmitXShukla/LLM/blob/main/code/step-01-tiny-transformer/train_sanskrit_gpt.py).
Run it as you read:
```bash
pip install torch regex
python train_sanskrit_gpt.py --smoke     # ~30-second sanity run
python train_sanskrit_gpt.py             # the real (still tiny) run
```
:::

---

## The shape of the whole file 🗺️

The file reads bottom-up (small pieces first), but it *runs* top-down: `main()`
loads text, turns it into numbers, builds the model, trains it, and samples from
it. Here's the whole pipeline on one page:

```{mermaid}
flowchart LR
    A["📄 corpus.txt"] --> B["🔤 tokenizer<br/>text → numbers"]
    B --> C["get_batch<br/>make x, y pairs"]
    C --> D["🧠 SanskritGPT<br/>embeddings → blocks → head"]
    D --> E["😲 loss<br/>how surprised?"]
    E --> F["🎚️ optimizer<br/>nudge the weights"]
    F -->|repeat| C
    D --> G["🎲 generate<br/>write Sanskrit"]
```

We'll walk it in the order that's easiest to learn:
**the knobs → the data → the model (embeddings, attention, blocks) → training →
generating → and finally `main()`, the conductor.**

---

## 0 · The knobs — `Config` ⚙️

Every choice you can tune lives in one small class at the top. Nothing is hidden.

```python
class Config:
    block_size = 128      # context length: how many aksharas the model sees at once
    n_embd     = 256      # width of the model (embedding dimension)
    n_head     = 4        # number of attention heads (n_embd must divide by this)
    n_layer    = 4        # number of transformer blocks stacked on top of each other
    dropout    = 0.1
    # training
    batch_size    = 32
    max_iters     = 3000
    eval_interval = 250
    eval_iters    = 50
    learning_rate = 3e-4
```

- **`block_size`** — how far back the model can look. 128 aksharas of context.
- **`n_embd`** — how many numbers represent each akshara. Wider = more capacity.
- **`n_head`** — how many attention "heads" run in parallel. `n_embd` must divide
  evenly by this (256 ÷ 4 = 64 numbers per head).
- **`n_layer`** — how many transformer blocks we stack. More = deeper = smarter
  (and slower).
- **`dropout`** — randomly ignore 10% of connections while training, so the model
  can't over-rely on any one path.
- The **training** block sets how big each batch is, how long we train, how often
  we check ourselves, and how big each learning step is.

:::{note} 🍼 Sensible weekend defaults
These are deliberately small so the file runs on a laptop. On a real GPU you scale
them *up* — more layers, wider `n_embd`, longer `block_size` — once your real
corpus is in place.
:::

---

## 1 · The data *is* the task — `get_batch` 📥

Here's the trick that makes training possible with **no hand-labelling at all**.
We take a window of aksharas as the input `x`, and *the same window shifted right
by one* as the answer `y`. So at every position, the correct answer is simply the
next akshara.

```python
def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])  # inputs shifted by 1
    return x.to(device), y.to(device)
```

```{mermaid}
flowchart LR
    subgraph one window
    X["x:  ध  र्  म  क्  षे"] --> Y["y:  र्  म  क्  षे  त्"]
    end
    X -. "predict →" .-> Y
```

- `torch.randint` picks `batch_size` random starting points in the text.
- `x` is `block_size` aksharas; `y` is the same, shifted by one.
- Both come back with shape `(batch_size, block_size)` — a batch of `B` sequences,
  each `T` aksharas long.

:::{important} 🔑 This shift-by-one is the whole supervision signal
No labels, no annotation — the text teaches itself. This is why it's called
*self*-supervised learning, and it's exactly how the biggest models are trained
too. Everything downstream is just this idea, scaled up.
:::

---

## 2 · Turning an akshara into numbers — embeddings 🔢

A model does maths, so first each akshara ID becomes a vector of numbers. We use
**two** embeddings — one for *what* the akshara is, one for *where* it sits —
because a transformer has no built-in sense of order. (These live inside the model
class we build next.)

```python
self.token_embedding    = nn.Embedding(vocab_size, cfg.n_embd)      # what
self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)  # where
```

Think of each akshara as landing on a point in a big "meaning space." During
training, aksharas that behave similarly drift toward similar points — the model
*learns* the geometry. We then simply **add** the two vectors: `meaning + position`.

---

## 3 · Attention — the one genuinely new idea — `Head` 🔎

This is the heart of the transformer, and the only truly new concept. Each
position produces three vectors:

- a **query** — "what am I looking for?"
- a **key** — "what do I offer?"
- a **value** — "what do I actually pass on if you pick me?"

```python
class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # affinities, scaled
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # hide the future
        wei = F.softmax(wei, dim=-1)         # weights that sum to 1
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v                       # weighted sum of values
```

Line by line in `forward`: turn each akshara into a query and a key; multiply
every query by every key (`q @ k`) to get an **affinity** — how relevant is each
earlier akshara to me? Then hide the future, softmax the affinities into weights
that add up to 1, and use them to take a **weighted average of the values**. That
weighted average *is* attention.

```{mermaid}
flowchart LR
    Q["query: what I want"] --> M(("· dot ·"))
    K["key: what I offer"] --> M
    M --> S["softmax → weights"]
    V["value: my content"] --> W((" weighted<br/>sum "))
    S --> W
    W --> O["what this position learns"]
```

:::{important} 🧠 Two small lines that matter enormously
**The `* k.shape[-1] ** -0.5` scaling.** As vectors get wider, raw dot products
grow large, softmax turns into a near one-hot spike, and gradients die. Dividing
by √(head_size) keeps things healthy — this is the "scaled" in *scaled
dot-product attention*.

**The causal mask (`tril` + `masked_fill`).** A language model must predict the
future from the past only — it can't peek at the answer. We set future entries to
`-inf` *before* softmax, so they become 0 after. Position 5 can see 0–5, never 6+.
:::

---

## 4 · Many heads at once — `MultiHeadAttention` 👀

One head learns one kind of relationship. We run several in parallel so the model
can track several at once — maybe one head follows vowel signs, another follows
word boundaries marked by a *danda* (`।`). We never assign these jobs; the heads
discover them.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # glue heads side by side
        return self.dropout(self.proj(out))                   # mix them back together
```

Run every head, concatenate their outputs side by side, then a `proj` layer mixes
them back into the model's width.

---

## 5 · A place to think — `FeedForward` 🧮

Attention moves information *between* positions. The feed-forward MLP then lets
each position *think* about what it just gathered — on its own. Widen to 4×, apply
a non-linearity, shrink back.

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
That's the whole pattern, repeated `n_layer` times.
:::

---

## 6 · One block — `Block` 🧱

A block bundles attention and the MLP with two supporting tricks: **residual
connections** and **LayerNorm**.

```python
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.sa  = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff  = FeedForward(cfg)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # communicate, then add back
        x = x + self.ff(self.ln2(x))   # compute, then add back
        return x
```

- **`x + ...` (residual).** Each sub-layer adds a *correction* to `x` instead of
  replacing it. This gives gradients a clean highway back to the early layers —
  it's what makes deep networks trainable at all.
- **LayerNorm.** Re-centres and re-scales each vector so the numbers stay stable
  through many layers.
- **Pre-norm.** We normalise *before* each sub-layer (the `ln1`/`ln2` sit *inside*
  the `x + (...)`). It trains more smoothly than the original 2017 post-norm design
  — something you'll *prove* to yourself in [Step 2](02-break-it.md).

```{mermaid}
flowchart TD
    I["x in"] --> N1["LayerNorm"] --> SA["attention"] --> A1(("＋"))
    I --> A1
    A1 --> N2["LayerNorm"] --> FF["feed-forward"] --> A2(("＋"))
    A1 --> A2 --> O["x out"]
```

---

## 7 · The whole model + the loss — `SanskritGPT` 🏗️

Now we stack everything: embeddings at the bottom, `n_layer` blocks in the middle,
and a final layer (`lm_head`) that scores every possible next akshara.

```python
class SanskritGPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)   # a score per akshara

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos                 # meaning + position
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)      # (B, T, vocab)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss
```

```{mermaid}
flowchart TD
    ID["aksharas (ids)"] --> TE["token embedding"]
    ID --> PE["position embedding"]
    TE --> ADD(("＋"))
    PE --> ADD --> BL["N × Block"]
    BL --> LNF["LayerNorm"] --> LMH["lm_head → score per akshara"]
    LMH --> LG["logits"]
```

`forward` runs the model. If we pass `targets`, it also computes the **loss** with
cross-entropy.

:::{note} 📉 What the loss means
Cross-entropy is, intuitively, *"how surprised was the model by the right
answer?"* Confident and correct → low loss; confident and wrong → high loss.
Training is nothing but nudging the weights to be less surprised by real Sanskrit.
A great sanity check: an untrained model's loss should be about
`ln(vocab_size)` — pure guessing. Watching it fall below that is your proof that
learning is happening. ✅
:::

---

## 8 · Making it write — `generate` 🎲

To produce text, we feed the context, look only at the logits for the *last*
position, and sample the next akshara — then loop.

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.cfg.block_size:]     # never feed more than block_size
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature       # focus on the last step
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

- **`@torch.no_grad()`** — we're not training, so skip gradient bookkeeping.
- **`temperature`** divides the logits: below 1.0 makes the model safer and more
  repetitive; above 1.0 makes it more adventurous.
- **`top_k`** keeps only the *k* most likely aksharas before sampling — it trims
  the long tail of nonsense.
- **`multinomial`** *samples* (rather than always taking the top pick), so output
  varies each run.

---

## 9 · Measuring honestly — `estimate_loss` 🧪

During training we periodically check the loss on a held-out **validation** split,
so we can tell real learning from memorising.

```python
@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out
```

`model.eval()` and `model.train()` flip dropout off and back on; averaging over
`eval_iters` batches gives a stable number instead of a noisy single reading.

---

## 10 · The conductor — `main()` 🎬

This is the part the model itself doesn't show you, and it's where everything
comes together. We'll take it in five small beats.

### a) Command-line options

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--tokenizer", default="grapheme", choices=["grapheme", "codepoint"])
    ap.add_argument("--smoke", action="store_true", help="tiny/fast run to check wiring")
    ap.add_argument("--prompt", default="विद्या", help="seed text for the final sample")
    args = ap.parse_args()
```

Four switches: which text file, which tokenizer (whole-akshara `grapheme` vs the
naive `codepoint` — see [Step 4](04-sanskrit-tokenizer.md)), a `--smoke` flag for a
fast wiring check, and a `--prompt` to seed the final sample.

### b) Config, smoke mode, device, seed

```python
    cfg = Config()
    if args.smoke:                       # shrink everything for a quick sanity check
        cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer = 32, 64, 2, 2
        cfg.max_iters, cfg.eval_interval, cfg.eval_iters = 200, 50, 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
```

`--smoke` shrinks the model and shortens training so you can confirm the whole
thing *runs* in ~30 seconds before committing to a real run. We pick the GPU if
there is one, and set a fixed random seed so runs are reproducible.

### c) Text → tokens → train/val split

```python
    text = Path(args.corpus).read_text(encoding="utf-8")
    tok = build_tokenizer(text, args.tokenizer)
    print(f"device={device}  tokenizer={tok.name}  vocab_size={tok.vocab_size}  "
          f"corpus_tokens={len(tok.encode(text)):,}")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
```

Read the corpus, build the tokenizer from it, and encode the whole thing into one
long list of IDs. Then hold back the last **10% for validation** — the model never
trains on it, so it's an honest test of whether it's learning patterns or just
memorising.

### d) Build the model, count it, pick the optimizer

```python
    model = SanskritGPT(cfg, tok.vocab_size).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
```

Create the model, move it to the device, and print how many parameters it has (a
good gut-check on size). **AdamW** is the standard, reliable optimizer — it turns
gradients into smart weight updates.

### e) The training loop — four lines, repeated

```python
    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, splits, cfg, device)
            print(f"iter {it:>5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch(splits["train"], cfg, device)   # 1. a batch of aksharas
        _, loss = model(xb, yb)                            # 2. forward: measure surprise
        optimizer.zero_grad(set_to_none=True)              # 3. clear old gradients
        loss.backward()                                    # 4. backward: assign blame
        optimizer.step()                                   #    nudge every weight
```

If you understand these four steps, you understand how *every* neural network is
trained:

```{mermaid}
flowchart LR
    A["get_batch"] --> B["forward → loss"]
    B --> C["loss.backward()<br/>gradients"]
    C --> D["optimizer.step()<br/>nudge weights"]
    D -->|repeat| A
```

`loss.backward()` is autograd computing how each weight contributed to the error
(the chain rule, automated); `optimizer.step()` moves each weight a little in the
right direction. Every `eval_interval` steps we print train *and* val loss so we
can watch progress.

### f) Sample, then save

```python
    print("\n----- sample -----")
    start = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    if start.numel() == 0:                       # prompt had no in-vocab tokens
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=200, temperature=0.8, top_k=20)
    print(tok.decode(out[0].tolist()))

    torch.save({"model": model.state_dict(), "vocab": tok.units, "cfg": vars(cfg)},
               "sanskrit_gpt.pt")
    print("\nsaved checkpoint -> sanskrit_gpt.pt")
```

We seed generation with your `--prompt` (falling back to a blank start if the
prompt has no known aksharas), generate 200 new aksharas, decode them back to
text, and print. Finally we **save a checkpoint** — the weights, the vocabulary,
and the config — so you can reload this exact model later without retraining.

:::{note} 🐍 The `if __name__ == "__main__":` line
The file ends with `if __name__ == "__main__": main()`. That just means "if you
*run* this file directly, start `main()`" — but if another file *imports* it (to
reuse `SanskritGPT`), nothing runs automatically. Standard Python housekeeping.
:::

---

## What you should see ▶️

```bash
python train_sanskrit_gpt.py --smoke
```

```text
device=cpu  tokenizer=grapheme  vocab_size=112  corpus_tokens=3,214
parameters: 0.21 M
iter     0: train loss 4.7213, val loss 4.7248
iter   200: train loss 2.61xx, val loss 2.9xxx
----- sample -----
धर्मे च विद्यायाः ... (looks like Sanskrit, means nothing yet)
```

The samples look like Sanskrit — right shapes, real aksharas — but they don't
*mean* anything. **That is the correct result.** The model learned what Sanskrit
looks like, not what it says; closing that gap is the rest of the book.

:::{important} ✅ Read the output charitably
Every "word" it makes is a *valid* akshara — it literally cannot produce an orphan
vowel sign — because of the grapheme tokenizer (see [Step 4](04-sanskrit-tokenizer.md)).
Nonsense-but-valid means your pipeline works; it just needs more data to become
nonsense-that-means-something.
:::

---

## Where people usually get stuck

**Shape errors.** Print the shape of every tensor (`print(x.shape)`) until the
mismatch is obvious. Everyone does this — it's not a sign you're bad at it.

**`n_embd` not divisible by `n_head`.** 256 ÷ 4 works; 256 ÷ 5 doesn't. The head
size must come out whole.

**Trying to make the output *good*.** It can't be, on this little data. Resist the
urge to fix it — that's [Step 3](03-tokenizers.md) onward.

---

## You are ready to move on when

You can point at any line — the √head_size scaling, the causal mask, the `x +`
residual, `cross_entropy`, `optimizer.step()`, the 90/10 split, the checkpoint
save — and say, in one sentence, what it does and why it's there.

Then head to [Step 2](02-break-it.md), where you'll delete these lines one at a
time and watch exactly what breaks.

---

:::{seealso} 📚 Go deeper
- 🧠 Line-by-line teaching notes: [`docs/notes/weekend1-tiny-transformer-teaching.md`](https://github.com/AmitXShukla/LLM/blob/main/docs/notes/weekend1-tiny-transformer-teaching.md)
- ✍️ The story of building it: [`docs/notes/weekend1-blog-tiny-sanskrit-gpt.md`](https://github.com/AmitXShukla/LLM/blob/main/docs/notes/weekend1-blog-tiny-sanskrit-gpt.md)
- 🏃 How to run it: [`code/step-01-tiny-transformer/README.md`](https://github.com/AmitXShukla/LLM/blob/main/code/step-01-tiny-transformer/README.md)
- 📄 The full file: [`code/step-01-tiny-transformer/train_sanskrit_gpt.py`](https://github.com/AmitXShukla/LLM/blob/main/code/step-01-tiny-transformer/train_sanskrit_gpt.py)
:::
