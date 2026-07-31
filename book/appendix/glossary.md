---
title: Glossary
short_title: Glossary
---

# Glossary

Every technical term used in this book, in plain words.

---

**Activation function** — a small maths function applied between layers that
lets a network learn non-straight-line patterns. SwiGLU is the common one now.

**Adapter** — a small set of extra weights added to a frozen model. LoRA
produces adapters.

**Attention** — the mechanism by which each token looks at earlier tokens and
decides which ones matter for predicting the next one.

**Base model** — a model trained only to predict the next token. It has not yet
been taught to follow instructions.

**bf16** — a 16-bit number format that keeps a wide range of magnitudes at the
cost of precision. The default for language model training.

**BPE (Byte Pair Encoding)** — a tokenizer method that repeatedly joins the most
common pair of neighbouring pieces into one new token.

**Calibration data** — a small sample of text used during quantization to decide
which weights matter most.

**Catastrophic forgetting** — when teaching a model something new makes it forget
things it used to know.

**Causal mask** — a block that stops the model from seeing future tokens, so it
cannot cheat by copying the answer.

**Checkpoint** — a saved copy of the model's weights partway through training.

**Chat template** — the special markup that marks where the user's turn ends and
the assistant's begins.

**Context window** — how many tokens the model can look at in one go.

**Continued pretraining (CPT)** — carrying on the pretraining of an
already-trained model, usually on new data such as a new language.

**Contamination** — when test data has accidentally ended up in the training
data, making evaluation scores meaningless.

**Distillation** — training a small model to copy a larger one.

**DPO (Direct Preference Optimization)** — training on pairs of better and worse
answers, without needing a separate reward model.

**Embedding** — the vector of numbers that represents one token.

**Fertility** — how many tokens one average word becomes. Lower is better.

**Fine-tuning** — further training of an existing model on new data.

**FlashAttention** — a way of computing attention that never writes the full
attention matrix to memory. Same result, much less memory.

**FSDP (Fully Sharded Data Parallel)** — splitting the model itself across
several GPUs so a large model fits.

**GGUF** — a file format for quantized models, used by llama.cpp for local
running.

**GQA (Grouped-Query Attention)** — several query heads share one set of keys and
values, which saves a lot of memory when serving.

**Gradient accumulation** — running several small batches and adding up their
gradients before updating, to imitate a large batch.

**Gradient checkpointing** — throwing away intermediate values during the forward
pass and recomputing them later, to save memory.

**GRPO (Group Relative Policy Optimization)** — generate several answers, score
them, and push the model toward the ones that scored above average.

**Head** — one independent attention operation. Models run several in parallel.

**KV cache** — stored keys and values from earlier tokens, kept during generation
so they are not recomputed.

**LoRA (Low-Rank Adaptation)** — training small add-on matrices instead of
updating the whole model.

**MoE (Mixture of Experts)** — many small expert layers with a router that uses
only a few of them per token.

**muP** — a way of setting hyperparameters on a small model so they transfer
correctly to a large one.

**Neuro-symbolic** — combining a neural model with explicit rule-based code.

**Normalization** — rescaling values so they stay in a sensible range. RMSNorm is
the common one now.

**Overfitting** — when a model learns your specific training file instead of
general patterns. Also called memorisation.

**Panini** — the grammarian who codified Sanskrit in the *Ashtadhyayi*, around
four thousand rules that form a near-formal system.

**Perplexity** — a measure of how surprised a model is by some text. Lower is
better.

**Pre-tokenization** — the rough splitting of text, usually by a regular
expression, that happens before the tokenizer's main algorithm runs.

**Pretraining** — the first, large training run on raw text.

**Projector** — the small network that maps image or audio vectors into a
language model's space.

**QLoRA** — LoRA applied on top of a quantized model. The cheapest way to
fine-tune.

**Quantization** — storing weights with fewer bits to save memory.

**RAG (Retrieval-Augmented Generation)** — searching your documents and putting
the relevant passages into the prompt, instead of expecting the model to know
everything.

**Replay data** — general data mixed into specialised training to stop the model
forgetting what it already knew.

**Residual connection** — a side road that lets information skip past a block.
Essential for training deep networks.

**Reward hacking** — when a model finds a way to score well without actually
being better.

**RLHF (Reinforcement Learning from Human Feedback)** — training a reward model
on human preferences, then optimising against it.

**RLVR (Reinforcement Learning with Verifiable Rewards)** — using an automatic
checker as the reward, for tasks where correctness can be verified.

**RMSNorm** — a simpler, faster form of normalization that skips subtracting the
mean.

**RoPE (Rotary Position Embeddings)** — encoding position by rotating query and
key vectors, so the model learns about distance between tokens.

**Router** — the small network in an MoE model that decides which experts handle
which token.

**Samasa** — a Sanskrit compound word, formed by joining several words.

**Sandhi** — the Sanskrit rules by which sounds change where words join.

**SFT (Supervised Fine-Tuning)** — training on example question and answer pairs.

**SwiGLU** — a gated activation function, standard in modern models.

**Sycophancy** — when a model learns to agree with the user rather than be
correct.

**Token** — one piece of text, as cut by the tokenizer.

**Tokenizer** — the tool that cuts text into tokens.

**Unicode normalization** — converting text so that visually identical strings
are stored identically. NFC is the usual form.

**Virama** — the Devanagari mark that removes a consonant's inherent vowel, used
to form conjunct letters.

**ZWNJ (zero-width non-joiner)** — an invisible character that stops two letters
from joining. Common and inconsistent in Urdu text.
