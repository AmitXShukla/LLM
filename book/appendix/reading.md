---
title: Things worth reading
short_title: Reading list
---

# Things worth reading

A short list. Depth over breadth.

For papers and PDFs, see the `docs/` folder in the repository.

---

## Start here

**Andrej Karpathy** — the best free starting point that exists.

- `micrograd` — an autograd engine in about 100 lines. Do this first if
  `loss.backward()` feels like magic.
- `nanoGPT` — the cleanest minimal GPT implementation. This is the basis for
  [Step 1](../01-build-transformer.md).
- The videos "Let's build GPT" and "Let's build the GPT Tokenizer".

**Sebastian Raschka, *Build a Large Language Model (From Scratch)*** — the best
modern practical book on exactly the material in Part 1, with a working
repository.

**`modded-nanogpt`** by Keller Jordan — a public competition to train a
GPT-2-sized model as fast as possible. The community has driven it from 45
minutes down to under 90 seconds on an 8-GPU machine.

Reading its history of record-breaking changes is a compressed tour of every
real improvement since 2022: the Muon optimizer, FlashAttention-3, low-precision
matrix multiplication, multi-token prediction. Better than most survey papers.

**`LitGPT`** — more batteries included than nanoGPT, still readable.

---

## Papers

**Attention Is All You Need** (Vaswani et al., 2017) — read it once, as history.

**The technical reports from the major open model families.** The OLMo and
Nemotron reports are the most useful for a learner, because they publish the
most detail about data, recipes, and what did not work.

**The DeepSeek-R1 report** — the clearest public description of the RLVR and
GRPO approach in [Step 14](../14-reasoning.md).

---

## For low-resource and Indic languages

**IndicGenBench** — benchmarks across many Indic languages, and the source of
the token fertility comparisons cited in [Step 3](../03-tokenizers.md).

**Indic tokenizer papers** — several recent papers on pre-tokenization rules and
vocabulary design for Indic scripts. These are directly actionable for
[Steps 4 and 5](../04-sanskrit-tokenizer.md).

**Papers on Sanskrit computational linguistics** — sandhi splitting,
morphological analysis, and the ongoing work on formalising the Ashtadhyayi. See
[Step 15](../15-panini-neurosymbolic.md).

---

## Tools you will meet

| Tool | What it is for |
|---|---|
| `transformers` | Loading and running models |
| `tokenizers` | Building tokenizers ([Steps 3 to 5](../03-tokenizers.md)) |
| `peft` | LoRA and QLoRA ([Steps 11 to 12](../11-adapt-base-model.md)) |
| `trl` | Instruction tuning and preference training |
| Unsloth | Faster, lower-memory fine-tuning |
| Axolotl | Configuration-driven fine-tuning |
| `verl`, OpenRLHF | Reinforcement learning at scale ([Step 14](../14-reasoning.md)) |
| vLLM, SGLang | Fast serving ([Step 18](../18-quantization-serving.md)) |
| llama.cpp | Local running, GGUF format |
| MONAI | Medical imaging ([Step 24](../24-medical.md)) |

---

## A note on staying current

This field moves fast enough that any list goes stale within months.

Two habits that work better than any reading list:

1. **Read the technical reports** when open models are released. They contain
   more practical detail than most papers.
2. **Follow one or two repositories** where people are actively competing on a
   measurable task. Watching what wins teaches you more than reading about what
   might.
