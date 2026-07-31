---
title: "Step 18 — Make it small and serve it"
short_title: "18. Quantization and serving"
---

# Step 18 — Make it small and serve it

**Goal:** get your model running fast enough and cheap enough for real use.

---

## Why this step matters

A model that only runs on a rented data-centre GPU is a demo. A model that runs
on a laptop is a product.

For a Sanskrit model this matters more than usual. A lot of the value is in
running **locally and privately** — sacred texts, personal study, a scholar's
unpublished work. If it needs a cloud, you have lost part of the point.

---

## What you do

### 1. Understand quantization

Model weights are normally 16-bit numbers. Quantization stores them in 8, 4, or
even fewer bits.

It is like keeping fewer decimal places. Much smaller, slightly less accurate.

### 2. Use post-training quantization

GPTQ and AWQ are the common methods. They need no retraining — just a small
**calibration dataset** that they use to work out which weights matter most.

**Use Sanskrit text for calibration, not English.** The calibration data decides
what the compression protects. If you calibrate on English, you protect English.

### 3. Test the quality after quantizing, in Sanskrit

:::{warning} This matters more than most people realise
Quantization damages low-resource-language performance noticeably more than
English performance.

A model that loses 1 percent on English benchmarks may lose far more on
Sanskrit. The rare, script-specific parts of the model are exactly the parts
that compression discards first.

If you only test in English, you will ship a broken model and not know.
:::

### 4. Use GGUF and llama.cpp for local running

This is what gets your model running on a normal laptop, or on Apple Silicon.
It is the format most local tools expect.

### 5. Understand the KV cache

When generating text, the model saves its keys and values for earlier tokens so
it does not have to recompute them for every new token.

This cache is often the biggest memory user during serving, and it grows with
context length.

This is why grouped-query attention from [Step 8](08-modern-architecture.md)
mattered: fewer key and value sets means a much smaller cache.

### 6. Serve with vLLM or SGLang

They handle **continuous batching** (starting new requests as old ones finish,
instead of waiting for the whole batch) and **paged attention** (managing the KV
cache like an operating system manages memory).

Together these give a very large throughput improvement over a naive
generation loop.

### 7. Measure three numbers, not one

- **Throughput** — total tokens per second across all users
- **Time to first token** — how long before anything appears
- **Time between tokens** — how fast the text flows once it starts

Users feel the second and third. Your dashboard usually shows only the first.

---

## Where people usually get stuck

Quantizing, checking an English benchmark, seeing a small drop, and shipping —
without ever testing the actual target language.

---

## You are ready to move on when

Your model runs on hardware a normal person owns, and you have measured its
quality after compression **in Sanskrit**.

---

:::{seealso} Related
- [Step 8](08-modern-architecture.md) — why GQA mattered
- [Step 25](25-release.md) — publishing quantized versions
:::

---

## 🖥️ Serving privately on your own NVIDIA hardware

For healthcare especially, the data often *cannot leave the building* — so you
serve on your own GPUs. Know what each layer of the NVIDIA stack is **for**:

```{mermaid}
flowchart TD
    NIM["🟢 NIM — OpenAI-compatible container<br/>pull → run → serve in hours"]
    TRTLLM["⚙️ TensorRT-LLM — LLM optimizer<br/>FP8/FP4, paged KV, LoRA serving"]
    TRITON["📦 Triton — multi-framework server<br/>your ECG / X-ray / ONNX models"]
    DYNAMO["🌐 Dynamo — distributed serving<br/>huge & reasoning models, disaggregated"]
    NIM --> TRTLLM
    NIM --> TRITON
    DYNAMO --> TRTLLM
```

| You want to… | Use |
|--------------|-----|
| Serve a standard LLM privately, fast | 🟢 **NIM** (or open-source **vLLM**) |
| Serve custom / non-LLM models (ECG, X-ray) | 📦 **Triton** with a TensorRT engine |
| Serve huge or **reasoning** models across many GPUs | 🌐 **Dynamo** |
| Squeeze max single-model speed | ⚙️ **TensorRT-LLM** + quantization |

```bash
# A NIM exposes an OpenAI-compatible endpoint — your app code barely changes.
docker run --gpus all -p 8000:8000 -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

:::{important} Why on-prem for healthcare 🔒
PHI (patient data) usually can't go to a public API. Serving on your own NVIDIA
hardware keeps data in the building, gives predictable latency/cost, and supports
the audit trail regulators expect.
:::

:::{note} Speed levers, in order
**(1) quantize** (FP8/FP4/INT4) → **(2) batch** (continuous batching + paged KV
cache) → **(3) speculate** (draft-then-verify) → **(4) disaggregate** prefill and
decode for reasoning workloads.
:::

:::{seealso} Full deployment chapter
📘 [`docs/reports/fine-tuning-foundation-models.pdf`](https://github.com/AmitXShukla/LLM/tree/main/docs/reports/fine-tuning-foundation-models.pdf) — Part VI covers this in depth, plus on-prem architecture and MLOps.
:::
