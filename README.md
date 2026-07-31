# Build Your Own Language Model

### A step-by-step book for engineers, using Sanskrit, Urdu, Tamil, Telgu

[![Deploy Book](https://github.com/AmitXShukla/LLM/actions/workflows/deploy.yml/badge.svg)](https://github.com/AmitXShukla/LLM/actions/workflows/deploy.yml)

[![Content: CC BY 4.0](https://img.shields.io/badge/content-CC--BY--4.0-lightgrey.svg)](LICENSE)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE-CODE)

**Read the book online: <https://amitxshukla.github.io/LLM/>**

---

## What this book is

Most engineers today only fine-tune models that other people built. This book goes one level deeper.

You start by writing a tiny transformer by hand, line by line. You break it on purpose, so you learn what each part does. Then you build a proper tokenizer for Sanskrit, collect and clean a real corpus, and adapt a large open model into a genuine Sanskrit specialist that runs on your own machine.

The whole book is written in simple English. Every technical word is explained the first time it appears. There is a glossary at the end.

## Who it is for

You know some Python. You have touched PyTorch, or you are willing to learn it as you go. You do not need a PhD, and you do not need a GPU cluster.

## Why Sanskrit and Urdu

Sanskrit teaches you **structure and scarcity**: sandhi, long compound words, and the hard fact that there is not much clean text. Urdu teaches you **mess and script**: right-to-left writing, unwritten short vowels, and half the internet's Urdu written in English letters.

Together they cover almost every problem you will meet in a real low-resource language project.

---

## Chapters

> 💡 Many chapters now end with a **🧑‍💻 Runnable code** section containing tested,
> copy-paste code, colored callouts, and diagrams. The full code lives in
> [`code/`](code/); see the table below the chapter list.

### Part 1 — Understand the machine
| Step | Chapter | Status |
|---|---|---|
| 0 | [Get your workspace ready](book/00-setup.md) | Draft |
| 1 | [Build a tiny Sanskrit transformer](book/01-build-transformer.md) | Draft |
| 2 | [Break your model on purpose](book/02-break-it.md) | Draft |

### Part 2 — Feed the machine
| Step | Chapter | Status |
|---|---|---|
| 3 | [Understand tokenizers](book/03-tokenizers.md) | Draft |
| 4 | [Build a Sanskrit tokenizer](book/04-sanskrit-tokenizer.md) | Draft |
| 5 | [Build an Urdu tokenizer](book/05-urdu-tokenizer.md) | Draft |
| 6 | [Collect your data](book/06-collect-data.md) | Draft |
| 7 | [Clean your data](book/07-clean-data.md) | Draft |

### Part 3 — Train it properly
| Step | Chapter | Status |
|---|---|---|
| 8 | [Rebuild with the modern design](book/08-modern-architecture.md) | Draft |
| 9 | [Run a real training job](book/09-training-run.md) | Draft |
| 10 | [Test it honestly](book/10-evaluation.md) | Draft |

### Part 4 — Make it useful
| Step | Chapter | Status |
|---|---|---|
| 11 | [Adapt a large open model](book/11-adapt-base-model.md) | Draft |
| 12 | [Teach it to follow instructions](book/12-instruction-tuning.md) | Draft |
| 13 | [Teach it what a good answer looks like](book/13-preference-tuning.md) | Draft |
| 14 | [Teach it to reason](book/14-reasoning.md) | Draft |
| 15 | [Panini: add the rules back in](book/15-panini-neurosymbolic.md) | Draft |

### Part 5 — Make it small and fast
| Step | Chapter | Status |
|---|---|---|
| 16 | [Mixture of Experts](book/16-moe.md) | Outline |
| 17 | [Distillation](book/17-distillation.md) | Outline |
| 18 | [Make it small and serve it](book/18-quantization-serving.md) | Draft |
| 19 | [Long text and RAG](book/19-long-context-rag.md) | Outline |

### Part 6 — Go beyond text
| Step | Chapter | Status |
|---|---|---|
| 20 | [Speech](book/20-speech.md) | Outline |
| 21 | [Images and manuscript OCR](book/21-vision.md) | Outline |
| 22 | [Video](book/22-video.md) | Outline |
| 23 | [Agents and tools](book/23-agents.md) | Outline |
| 24 | [Medical images and heart sounds](book/24-medical.md) | Outline |

### Part 7 — Ship it
| Step | Chapter | Status |
|---|---|---|
| 25 | [Release your model responsibly](book/25-release.md) | Draft |

### Appendix
- [Glossary](book/appendix/glossary.md)
- [Hardware notes (including DGX Spark)](book/appendix/hardware.md)
- [Where to find Sanskrit and Urdu text](book/appendix/corpora.md)
- [Who else is working on this](book/appendix/landscape.md)
- [Should you build from scratch at all?](book/appendix/scratch-or-finetune.md)
- [Things worth reading](book/appendix/reading.md)

---

## 🧑‍💻 Runnable code

Every core step now ships with **tested, runnable code** in [`code/`](code/).
Each folder is self-contained with its own README (what it does, how to run,
hardware, and time).

| Folder | What it runs | Used in |
|---|---|---|
| 🧠 [`code/step-01-tiny-transformer/`](code/step-01-tiny-transformer/) | A ~250-line GPT from scratch that babbles Sanskrit | Steps 1–2, 9 |
| 🔤 [`code/step-04-sanskrit-tokenizer/`](code/step-04-sanskrit-tokenizer/) | Code-point vs. grapheme (akshara) tokenizers | Step 4 |
| 🧹 [`code/step-06-data-audit/`](code/step-06-data-audit/) | PDFs → clean corpus, with an OCR health check | Steps 6–7 |
| 🚀 [`code/step-11-adapt-base-model/`](code/step-11-adapt-base-model/) | LoRA/QLoRA fine-tuning: dataset → train → chat | Steps 11–13 |
| 🧩 [`code/step-14-reasoning/`](code/step-14-reasoning/) | Verifiable rewards + GRPO, and a distillation set | Steps 14, 17 |
| ❤️ [`code/step-24-medical-ecg/`](code/step-24-medical-ecg/) | 1D-CNN heartbeat → arrhythmia, imbalance-aware | Step 24 |
| 🦴 [`code/step-24-medical-xray/`](code/step-24-medical-xray/) | X-ray → fracture by transfer learning | Steps 21, 24 |
| 🎥 [`code/step-22-video/`](code/step-22-video/) | Per-frame CNN + temporal Transformer for behaviour | Step 22 |

```bash
# quickest taste — see the Sanskrit tokenization "gotcha" in 5 seconds
pip install regex && python code/step-04-sanskrit-tokenizer/devanagari_tokenizer.py
```

## 📚 Companion materials & downloads

Longer-form notes, a full course PDF, and diagrams live in [`docs/`](docs/) and
are indexed in [`docs/references.md`](docs/references.md).

| Resource | Format | What it is |
|---|---|---|
| 📘 [Fine-Tuning Foundation Models](docs/reports/fine-tuning-foundation-models.pdf) | PDF (58 pp) | Full companion course: language, healthcare multimodal (ECG/X-ray/video), reasoning, private NVIDIA deployment, and an interview Q&A bank |
| 🧠 [Tiny-transformer teaching notes](docs/notes/weekend1-tiny-transformer-teaching.md) | Markdown | Every concept in the from-scratch model, mapped to the code |
| ✍️ [Blog: I built a tiny Sanskrit GPT](docs/notes/weekend1-blog-tiny-sanskrit-gpt.md) | Markdown | Publish-ready narrative of the tokenization gotcha |
| 🚀 [Fine-tuning teaching notes](docs/notes/weekend2-finetuning-teaching.md) | Markdown | LoRA/QLoRA, SFT, DPO, and the road to reasoning |
| 🗺️ [Fine-tuning architecture](docs/notes/finetuning-architecture-diagram.md) | Markdown (Mermaid) | The full pipeline + LoRA concept + roadmap diagrams |
| 🖥️ [GPU primer (CuTile)](docs/notes/gpu-primer-cutile.md) | Markdown | Beginner explainer of GPU programming |

---

## Build the book locally

The book is written in [MyST Markdown](https://mystmd.org). You need Node.js 20 or newer.

```bash
# install the MyST command line tool, once
npm install -g mystmd

# start a live preview at http://localhost:3000
myst start

# build the static site into ./_build/html
myst build --html
```

To build a PDF you also need a LaTeX installation:

```bash
myst build --pdf
```

## How publishing works

Every push to `main` triggers the [`deploy.yml`](.github/workflows/deploy.yml) workflow. It builds the site with MyST and publishes it to GitHub Pages.

To turn it on the first time:

1. Go to **Settings → Pages** in this repository.
2. Under **Source**, choose **GitHub Actions**.
3. Push to `main`.

## Repository layout

```
.
├── book/            # the chapters, one Markdown file per step
│   ├── intro.md     # the long introduction
│   ├── appendix/    # glossary, hardware, corpora, reading list
│   └── images/      # figures used in chapters
├── docs/            # reference PDFs, papers, and notes (not built into the book)
├── code/            # runnable example code for each chapter
├── myst.yml         # book config and table of contents
└── .github/workflows/deploy.yml
```

**Adding a new chapter:** create the Markdown file in `book/`, then add one line for it in the `toc:` section of `myst.yml`.

---

## Contributing

Corrections and additions are welcome. The most useful contributions are:

- Tokenizer fertility measurements for languages not covered here
- Evaluation sets written by native speakers
- Reports of what failed for you and why

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and open an issue before sending a large pull request.

---

## Author

**Amit Shukla**

- Website: <https://AmitXShukla.github.io/LLM/>
- GitHub: [@AmitXShukla](https://github.com/AmitXShukla)
- YouTube: <https://youtube.com/@amit.shukla>
- Contact: X.com/@ashuklax

If this book helped you, a star on the repository is genuinely appreciated.

## Citing this book

See [CITATION.cff](CITATION.cff), or use:

> Amit Shukla. *Build Your Own Language Model: A step-by-step book for engineers, using Sanskrit and Urdu.* 2026. https://github.com/AmitXShukla/llm

## Licence

Two licences, because a book is two things at once:

- **The writing** (everything in `book/` and `docs/`) is licensed under
  [Creative Commons Attribution 4.0 International](LICENSE) (CC BY 4.0).
  You may share and adapt it, including commercially, as long as you give credit.
- **The code** (everything in `code/`, and all code samples inside chapters) is
  licensed under the [MIT Licence](LICENSE-CODE). Use it freely.

Reference PDFs placed in `docs/` remain under the licence of their original authors. Do not commit anything you do not have the right to redistribute.
