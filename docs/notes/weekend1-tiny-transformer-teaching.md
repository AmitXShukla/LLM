# TEACHING.md — How this tiny Sanskrit GPT actually works

Read this with `train_sanskrit_gpt.py` open beside you. Every section names the
class or function it explains. By the end you'll understand a transformer not as
a diagram you memorised, but as code you could rewrite from memory.

The model has exactly one job: **given some aksharas, predict the next one.** A
chatbot, a translator, a reasoning model — all of them are elaborations of this
one trick. Nail this and the rest of your LLM journey is just scaling and
fine-tuning.

---

## 0. The data shape (`get_batch`)

We turn the whole corpus into one long list of token IDs. Training data is just
pairs of windows: `x` is `block_size` aksharas, and `y` is the *same* window
shifted right by one. So for every position, the "correct answer" is literally
the next akshara. That shift-by-one is the entire supervision signal — no labels,
no annotation. The text labels itself. This is why it's called *self*-supervised.

```python
x = data[i      : i+block_size]
y = data[i+1    : i+block_size+1]   # the answer key is just "everything, one step later"
```

---

## 1. Embeddings — turning a symbol into a vector (`token_embedding`, `position_embedding`)

A neural net can't do math on the symbol `क`. So we give every vocabulary item a
learnable vector (`token_embedding`). At the start these vectors are random; over
training, aksharas that behave similarly drift to similar vectors. That's the
model *learning what each syllable means* from usage alone.

There's a second, subtler embedding. Attention (next section) is **order-blind**
— shuffle the inputs and it computes the same thing. But word order obviously
matters. So we add a `position_embedding`: a learned vector for "slot 0", "slot
1", and so on. The model's input is `token meaning + position`. Now it knows both
*what* is there and *where*.

---

## 2. Attention — the one genuinely new idea (`Head`)

This is the heart. Everything else is supporting cast.

Each token produces three vectors:
- **query** — "what am I looking for?"
- **key** — "what do I contain / advertise?"
- **value** — "what will I actually contribute if you pick me?"

To decide how much position *i* should listen to position *j*, we take the dot
product of *i*'s query with *j*'s key. Big dot product = "these two are
relevant to each other." Do this for all pairs and you get an attention matrix
`wei` of shape `(T, T)`.

```python
wei = q @ k.transpose(-2,-1) * head_size**-0.5
```

Two details that look small but matter:

- **The `* head_size**-0.5` scaling.** Without it, dot products grow large as the
  vectors get wider, softmax saturates into a near one-hot spike, and gradients
  die. Dividing by √(head_size) keeps the numbers in a healthy range. This is the
  "scaled" in *scaled dot-product attention*.

- **The causal mask.** A language model must predict the future from the past
  only — it can't be allowed to peek at the answer. We force every position to
  ignore later positions by setting those entries to `-inf` before softmax:

  ```python
  wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
  wei = F.softmax(wei, dim=-1)   # -inf -> 0 after softmax
  ```

  `tril` is a lower-triangular matrix of 1s. Position 5 can see 0–5, never 6+.

Then softmax turns affinities into weights that sum to 1, and we use them to take
a **weighted average of the value vectors**. Each position walks away with a
custom blend of information pulled from the positions it cared about.

```python
out = wei @ v
```

That's attention. Re-read it until `q@k.T -> mask -> softmax -> @v` feels obvious.

---

## 3. Multiple heads (`MultiHeadAttention`)

One head learns one kind of relationship. We run several in parallel so the model
can track several at once — perhaps one head learns to bind a vowel sign to its
consonant, another learns that a *danda* (। ॥) tends to end a clause. We don't
assign these roles; the heads specialise on their own. Their outputs are
concatenated and passed through a projection back to the model width.

---

## 4. The feed-forward MLP (`FeedForward`)

Attention *moves information between* positions. The MLP then lets each position
*think* about what it just received, independently. It widens to 4× and back
(`n_embd -> 4*n_embd -> n_embd`) with a GELU non-linearity in between. The 4× is
convention; the non-linearity is what lets the network represent more than linear
relationships.

Rule of thumb: **attention = communication, MLP = computation.** A transformer
block alternates the two.

---

## 5. Residuals, LayerNorm, and pre-norm (`Block`)

```python
x = x + self.sa(self.ln1(x))
x = x + self.ff(self.ln2(x))
```

- **The `x + ...` (residual connection).** Instead of replacing `x`, each
  sub-layer *adds a correction* to it. This gives gradients a clean highway back
  to the early layers, which is what makes deep networks trainable at all.
- **LayerNorm.** Re-centres and re-scales each vector to keep the numbers stable
  as they flow through many layers. Stability = faster, less fragile training.
- **Pre-norm** (normalising *before* the sub-layer, as we do) trains more
  smoothly than the original post-norm design. It's why the LayerNorms sit inside
  the `x + (...)` rather than outside.

Stack `n_layer` of these blocks and you have the body of the model.

---

## 6. The output head and the loss (`lm_head`, `cross_entropy`)

`lm_head` is a single linear layer projecting each position's vector to one score
per vocabulary item — the *logits*. Softmax would turn them into probabilities,
but `F.cross_entropy` does that internally, so we feed it raw logits plus the
correct next-token IDs.

Cross-entropy is, intuitively, "how surprised was the model by the right answer?"
Confident and correct → low loss. Confident and wrong → high loss. Training is
nothing but nudging the weights to be less surprised by real Sanskrit. A useful
sanity number: an untrained model's loss should be about `ln(vocab_size)` (pure
guessing). Watching it fall below that is your proof learning is happening.

---

## 7. The training loop (`main`)

Four lines, repeated thousands of times:

```python
_, loss = model(xb, yb)          # forward: predict, measure surprise
optimizer.zero_grad()            # clear last step's gradients
loss.backward()                  # backward: blame each weight for the error
optimizer.step()                 # nudge every weight a little in the right direction
```

`loss.backward()` is autograd computing the gradient of the loss w.r.t. every
parameter (the chain rule, automated). `AdamW` is the optimiser that turns those
gradients into smart weight updates. That's it — that's "training a neural net."

We periodically call `estimate_loss` on a held-out **val** split. If train loss
keeps dropping while val loss climbs, you're *overfitting* — memorising the
corpus instead of learning the language. With our 20-line toy corpus that happens
almost immediately, which is itself the lesson: **the model is data-starved, not
brain-starved.** That's the bridge to the real project.

---

## 8. Generating text (`generate`)

To sample, we feed the prompt, look only at the logits for the *last* position,
and pick a next token. Three knobs shape the output:

- **temperature** divides the logits. `< 1.0` makes the model more
  conservative/repetitive; `> 1.0` more adventurous/chaotic. 0.8 is a calm
  default.
- **top-k** restricts sampling to the k most likely tokens, cutting off the long
  tail of nonsense.
- **`torch.multinomial`** then samples from what remains — so output varies run
  to run, instead of always taking the single most likely token.

We append the new token, slide the context window (never longer than
`block_size`), and repeat.

---

## 9. Why the *grapheme* tokenizer matters here specifically

Because the model emits one **token** at a time, the tokenizer decides what a
"step" even is. With the grapheme tokenizer, every emitted token is a complete,
valid akshara — the model literally *cannot* produce an orphan vowel sign or a
dangling virama, because no such unit exists in its vocabulary. With the
code-point tokenizer it can, and early in training it will. You made invalid
states unrepresentable by choosing the right unit. That principle — pick the
representation that bakes in your constraints — is worth far more than this one
model.

---

## Where to go next (in rough order of value-for-effort)

1. **Feed it real data.** Everything above is starving on 20 verses. This is the
   single biggest lever. See the data section in `BLOG.md`.
2. **Scale the model** on the DGX Spark: longer `block_size`, bigger `n_embd`,
   more `n_layer`. Watch val loss to find the point where more data, not more
   model, becomes the bottleneck.
3. **Swap to subword (BPE) tokenization** once your vocab of whole aksharas gets
   unwieldy — it's the bridge to how production models tokenize.
4. **Stop pretraining from scratch and start fine-tuning** a real multilingual
   base model. You'll now understand exactly what the libraries are doing under
   the hood — which was the whole point of this weekend.
