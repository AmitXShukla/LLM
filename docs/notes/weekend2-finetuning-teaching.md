# TEACHING_finetune.md — Weekend 2, one concept at a time

Read this next to the three scripts. Weekend 1 taught you how a transformer works
*inside*. Weekend 2 teaches you how to *adapt a giant one someone else trained*.
Almost everything here rests on ideas you already have — I'll keep pointing back
to weekend 1 so it compounds instead of feeling brand new.

---

## 0. The one-sentence difference

- **Weekend 1 = pretraining.** Start from random weights, feed raw text, learn
  the language from scratch. You did this on 20 verses (and it starved).
- **Weekend 2 = fine-tuning.** Start from a model that *already* knows language,
  and teach it a *behaviour* using a small set of examples.

You are no longer teaching Sanskrit. You're teaching "when asked to translate,
translate." The heavy lifting was already paid for by whoever pretrained the base.

---

## 1. Base vs Instruct models

Two flavours of pretrained model you'll download:

- A **base** model (e.g. `sarvam-1`) only knows how to *continue* text — exactly
  like your weekend-1 model, just vastly bigger. Ask it a question and it might
  continue with *more questions*.
- An **instruct** model (e.g. `Qwen2.5-1.5B-Instruct`) has already been
  fine-tuned to follow instructions and chat.

You can SFT either. Fine-tuning an *instruct* model is the gentler start (it
already has manners; you're nudging its domain). Fine-tuning a *base* model gives
you more control but needs more data. Our default is an instruct model for a fast,
forgiving first run.

---

## 2. The data changed shape (`01_make_dataset.py`)

Weekend 1's data was one long string. Weekend 2's data is **pairs**:
`{"prompt": ..., "completion": ...}`. Keeping them as two fields (not one glued
string) is what lets the trainer score the model on the *answer only* — see §6.
That's why step 1 writes JSONL in exactly this shape. The dataset *is* the
lesson plan; the model learns whatever behaviours your pairs demonstrate, and
nothing else.

---

## 3. Tokenization, again — and why it still bites (callback to weekend 1)

Remember watching Devanagari shatter into code points? Real models have the same
problem at a different level. Each model ships its own **tokenizer**, and
English-centric ones chop Sanskrit into many tiny pieces — a high "fertility"
(tokens per word). More tokens per verse means shorter effective context, slower
training, and wasted capacity.

This is a concrete reason to prefer an **Indic-aware base**: Sarvam's tokenizer,
for instance, was built for Indic scripts and needs far fewer tokens per Sanskrit
word than a generic one. The weekend-1 lesson ("pick the right unit") reappears
here as "pick the right base for your script." Same principle, bigger stakes.

---

## 4. LoRA — the whole reason this fits on one machine (`02`, SECTION 2)

Fine-tuning *all* of a 1.5B model means storing, for every one of 1.5 billion
weights, a gradient and optimizer state too — roughly 4× the model in memory.
That's how full fine-tuning eats clusters.

**LoRA's trick:** freeze every original weight. Beside each big weight matrix `W`,
add two skinny matrices `A` (d×r) and `B` (r×d) where the **rank `r`** is small
(say 16). Train only `A` and `B`. Their product `B·A` is a low-rank "correction"
added to `W`:

```
output = (W + B·A) · x        # W frozen; only A, B learn
```

Why does something so small work? Because adapting *behaviour* doesn't require
rewriting what the model knows — it only needs to steer it in a few directions.
Low-rank is enough to capture that steer. You end up training well under 1% of the
parameters. `trainer.model.print_trainable_parameters()` in the script prints the
exact ratio so you can see it with your own eyes.

The knobs:
- **`r`** — adapter capacity. Bigger = more expressive, more memory. 8–32 typical.
- **`lora_alpha`** — scaling; the effective strength is roughly `alpha / r`.
- **`target_modules`** — which matrices get adapters. We use `"all-linear"` so it
  works across Qwen/Gemma/Llama/Sarvam without you memorising each family's layer
  names. (The commented-out explicit list shows what it expands to.)

---

## 5. Quantization & QLoRA (`02`, SECTION 4)

The frozen base never gets updated — so why store it in full precision? **QLoRA**
squashes the base to **4-bit** (format `nf4`, tuned for weight distributions;
`double_quant` squeezes a bit more), while the small LoRA adapters stay in bf16.
Memory for the base drops ~4×. On your DGX Spark's 128GB unified memory, this is
the lever that lets you climb from a 1.5B toy to a 7B, 13B, even ~70B base later.

One hardware-specific rule baked into the code: **compute in bf16, not fp16.** The
GB10/Blackwell + fp16 combination has documented numerical issues; bf16 is the
recommended precision on this box. (This is why `bnb_4bit_compute_dtype` and the
`SFTConfig(bf16=...)` both say bfloat16.)

---

## 6. The training objective = weekend 1's loss, masked (`02`, SECTION 3)

Here's the satisfying part. The SFT loss is the **exact same token-level
cross-entropy** you used in weekend 1 — "how surprised was the model by the next
token?" The only addition: with `completion_only_loss=True`, the trainer **masks
the prompt tokens**, so the model is scored only on producing the *completion*.

Intuition: we don't want the model rewarded for parroting the question back. We
want it good at generating the answer. Masking the prompt focuses all the learning
signal there. Everything else — predict, measure surprise, backprop, step — is the
loop you already wrote by hand.

---

## 7. Chat templates (`03`, `build_prompt`)

Instruct models are trained with an exact special-token layout marking user vs
assistant turns. That layout is the **chat template**, stored in the tokenizer.
At inference we call `tokenizer.apply_chat_template(...)` so our prompt matches
what training looked like. Mismatch here is the #1 cause of "why is my fine-tuned
model acting dumb?" — the weights are fine, the wrapper is wrong.

---

## 8. The Trainer = your weekend-1 loop, industrialised (`02`, SECTION 5)

`SFTTrainer` isn't magic. It tokenizes, applies LoRA, masks prompts, then runs the
predict → loss → `backward()` → `optimizer.step()` loop from weekend 1, plus the
boring-but-vital extras: gradient accumulation (simulate a big batch on small
memory), gradient checkpointing (trade compute for memory), learning-rate warmup
and cosine decay, logging, checkpointing. You *could* write it by hand — you
basically did — but you'd re-derive a hundred people's bug fixes.

---

## 9. What you should watch, and what can go wrong

- **Trainable %**: should print as a fraction of a percent. If it says ~100%,
  your LoRA didn't attach — check `peft_config`.
- **Loss trend**: should fall then flatten. On 15 examples it'll overfit almost
  instantly (same lesson as weekend 1 — *feed it more data*). A few hundred good
  pairs is a real starting point.
- **Catastrophic forgetting**: fine-tune too hard and a model can lose skills it
  had. LoRA helps (the base is frozen), but keep learning rate and epochs modest,
  and mix in general examples if you see its other abilities degrade.
- **`--compare` looks identical**: with a tiny dataset the tuned model may barely
  differ. That's honest feedback about data size, not a bug.

---

## 10. Where weekend 3+ goes (the map from here)

You now have the base skill (SFT + LoRA). The path onward, in value order:

1. **Evaluation.** Before improving, *measure*. Hold out some pairs; check
   translation/explanation quality. You can't steer what you don't score.
2. **More & better data.** Still the bottleneck. Your OCR'd corpus → drafted
   pairs → hand-checked. This dwarfs every other lever.
3. **DPO (preference tuning).** After SFT, show the model pairs of *better* vs
   *worse* answers to sharpen quality and tone. It's the practical, single-GPU
   cousin of RLHF.
4. **Reasoning via verifiable rewards.** Sanskrit's rule-based grammar (Panini)
   means correctness — sandhi, segmentation, meter — is *checkable by code*. That
   makes it a natural fit for RL-with-verifiable-rewards: the checker is your
   reward function. This is genuinely a place your "ancient" language is *easier*
   than English.
5. **Agents & RAG — mostly not training.** Wrap your model with retrieval over
   your corpus and tools. Your DGX Spark ships with agent tooling; this is
   configuration, not gradient descent.

Do §1–§2 before reaching for §3–§5. The unglamorous parts (measure, get data) are
where the real gains hide — exactly as they were in weekend 1.
