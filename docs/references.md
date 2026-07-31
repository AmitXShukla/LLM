# Reference index

An index of the papers, reports, and resources behind this book.

For anything you cannot legally commit as a PDF, put the link and a short
summary here instead. A clear two-sentence summary is often more useful to a
reader than the full paper.

---

## Format

```
### Short title
- **Authors / team:**
- **Year:**
- **Link:**
- **Local file:** docs/papers/... (or "link only")
- **Why it matters for this book:** one or two plain sentences.
- **Related step:** Step N
```

---

## Foundations

### Attention Is All You Need
- **Authors:** Vaswani et al.
- **Year:** 2017
- **Link:** https://arxiv.org/abs/1706.03762
- **Local file:** link only
- **Why it matters:** The paper that introduced the transformer. Read it once,
  as history. Almost nothing in it survives unchanged in modern models.
- **Related step:** Step 1, Step 8

---

## Tokenization and Indic languages

*(Add entries here as you collect them. See Step 3, Step 4, and Step 5.)*

---

## Training and optimization

*(See Step 9.)*

---

## Post-training and reasoning

*(See Steps 12 to 14.)*

---

## Sanskrit grammar and computational linguistics

*(See Step 15.)*

---

## Corpora and datasets

*(See Step 6 and the corpora appendix.)*

---

## 📦 Companion materials (generated for this book)

These are original notes, code walkthroughs, and a full course PDF produced
alongside the book. They live in `docs/` and are linked from the README. All are
freely redistributable (authored for this project).

### Fine-Tuning Foundation Models — full course (58-page PDF) 📘
- **Local file:** [`docs/reports/fine-tuning-foundation-models.pdf`](reports/fine-tuning-foundation-models.pdf)
- **Why it matters:** a self-contained companion course covering language fine-tuning (Sanskrit, Tamil, Telugu, Urdu), healthcare multimodal models (ECG, X-ray, video), reasoning models, and private NVIDIA deployment — with runnable code and an interview question bank.
- **Related steps:** 11–24.

### Weekend 1 — Tiny transformer, teaching notes 🧠
- **Local file:** [`docs/notes/weekend1-tiny-transformer-teaching.md`](notes/weekend1-tiny-transformer-teaching.md)
- **Why it matters:** concept-by-concept walkthrough of the from-scratch model, mapped line-by-line to the code in `code/step-01-tiny-transformer/`.
- **Related step:** Steps 1–2.

### Weekend 1 — Blog draft ✍️
- **Local file:** [`docs/notes/weekend1-blog-tiny-sanskrit-gpt.md`](notes/weekend1-blog-tiny-sanskrit-gpt.md)
- **Why it matters:** a publish-ready narrative of building the tiny Sanskrit GPT and the tokenization "gotcha"; good raw material for a post or video.
- **Related step:** Steps 1–4.

### Weekend 2 — Fine-tuning teaching notes 🚀
- **Local file:** [`docs/notes/weekend2-finetuning-teaching.md`](notes/weekend2-finetuning-teaching.md)
- **Why it matters:** explains base vs. instruct, LoRA/QLoRA, SFT completion-only loss, chat templates, and the road to DPO/reasoning — mapped to `code/step-11-adapt-base-model/`.
- **Related steps:** 11–14.

### Fine-tuning architecture diagram 🗺️
- **Local file:** [`docs/notes/finetuning-architecture-diagram.md`](notes/finetuning-architecture-diagram.md)
- **Why it matters:** Mermaid diagrams of the full fine-tuning pipeline, the LoRA concept, and where SFT sits in the post-training roadmap.
- **Related steps:** 11–14.

### GPU primer (CuTile) 🖥️
- **Local file:** [`docs/notes/gpu-primer-cutile.md`](notes/gpu-primer-cutile.md)
- **Why it matters:** a beginner-friendly explainer of GPU programming and NVIDIA CuTile — background for readers new to why GPUs matter for all of this.
- **Related step:** Appendix (hardware).

---

## 🔑 Key papers & tools referenced in the book

Link-only (check each source's licence before committing any PDF):

- **DeepSeek-R1** (Guo et al., 2025) — reasoning via RLVR + GRPO. arXiv:2501.12948. → Steps 14, 17.
- **GRPO / DeepSeekMath** (Shao et al., 2024) — the value-free RL algorithm behind reasoning models. → Step 14.
- **QLoRA** (Dettmers et al., 2023) — 4-bit base + LoRA adapters. → Step 11.
- **LoRA** (Hu et al., 2021) — low-rank adaptation. → Step 11.
- **DPO** (Rafailov et al., 2023) — preference tuning without a reward model. → Step 13.
- **MedGemma** (Google, 2026) — open medical vision-language models (arXiv:2507.05201). → Step 24.
- **PTB-XL / MIT-BIH** — ECG arrhythmia benchmark datasets. → Step 24.
- **NVIDIA NIM / TensorRT-LLM / Triton / Dynamo** — the private inference stack. → Step 18.
- **Sarvam-1** — Indic-tuned base model with efficient Devanagari tokenizer. → Steps 3–5, 11.
