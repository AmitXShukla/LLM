# Fine-Tuning Architecture — Process, Concepts & Tools

A visual map of weekend 2: how raw Sanskrit text becomes a fine-tuned assistant,
and which tool does each job. Diagrams are Mermaid — they render on GitHub,
Obsidian, and most Markdown viewers.

---

## 1 · End-to-end pipeline

```mermaid
flowchart TD
    subgraph DATA["1 · Data preparation"]
        A["PDFs / OCR'd corpus<br/>(from weekend 1)"] --> B["Build instruction pairs<br/>prompt + completion<br/>01_make_dataset.py"]
        B --> C[("sanskrit_sft.jsonl")]
    end

    subgraph PREP["2 · Model preparation"]
        D["Pretrained BASE model<br/>Qwen / Gemma-3 / Sarvam-1"] --> E{"Quantize to 4-bit?<br/>(QLoRA)"}
        E -->|yes| F["4-bit frozen base<br/>BitsAndBytesConfig · nf4"]
        E -->|no| G["bf16 frozen base"]
        F --> H["Attach LoRA adapters<br/>freeze base · train A,B only<br/>peft · LoraConfig"]
        G --> H
    end

    subgraph TRAIN["3 · Supervised fine-tuning"]
        H --> I["SFTTrainer<br/>trl · completion-only loss"]
        C --> I
        I --> J["loop: predict → loss → backward → step"]
        J --> K[("LoRA adapter<br/>~tens of MB")]
    end

    subgraph USE["4 · Inference"]
        K --> L["base + adapter<br/>peft · PeftModel"]
        D -.reuse frozen base.-> L
        L --> M["apply chat template<br/>+ generate<br/>03_chat.py"]
        K --> N["optional: merge_and_unload<br/>→ single standalone model"]
    end

    HW["Substrate: NVIDIA DGX Spark · GB10 Blackwell · 128GB unified memory · bf16 (not fp16)"]
    HW -.-> PREP
    HW -.-> TRAIN
```

---

## 2 · The LoRA concept (why it fits on one machine)

Freeze the giant original weight `W`; train only two skinny matrices `A` and `B`.
Their low-rank product is a small "nudge" added to the frozen weight.

```mermaid
flowchart LR
    X(["input x"]) --> W["W · x<br/>FROZEN original weight"]
    X --> A["A · x<br/>down-project to rank r<br/>TRAINABLE"]
    A --> Bp["B · (A·x)<br/>up-project back<br/>TRAINABLE"]
    W --> P(("＋"))
    Bp --> P
    P --> Y(["output = (W + B·A)·x"])
```

`r` = rank (adapter capacity) · `lora_alpha` = strength scale · only `A`,`B` get
gradients ⇒ **well under 1% of parameters trained**.

---

## 3 · Where SFT sits in the bigger roadmap

```mermaid
flowchart LR
    P["Pretraining<br/>weekend 1 · from scratch"] --> S["SFT + LoRA<br/>weekend 2 · YOU ARE HERE"]
    S --> E["Evaluation<br/>held-out pairs"]
    E --> D["DPO<br/>preference tuning"]
    D --> R["RLVR<br/>Panini grammar = verifier"]
    R --> AG["Agents + RAG<br/>retrieval over your corpus"]
```

---

## Tools used

| Tool | Role in the pipeline |
|------|----------------------|
| **Hugging Face `datasets`** | Loads the `prompt`/`completion` JSONL into a training-ready dataset. |
| **`transformers`** | `AutoModelForCausalLM` / `AutoTokenizer` — loads the pretrained base and its tokenizer; `BitsAndBytesConfig` declares 4-bit. |
| **`bitsandbytes`** | Actually performs the 4-bit (nf4) quantization for QLoRA. Only needed with `--4bit`. |
| **`peft`** | `LoraConfig` defines the adapters; `PeftModel` attaches them at inference; `merge_and_unload()` bakes them in. |
| **`trl`** | `SFTTrainer` + `SFTConfig` — runs supervised fine-tuning with completion-only loss. |
| **`accelerate`** | Device placement and the training-loop plumbing under the Trainer. |
| **`torch`** | The tensor/autograd engine everything sits on (use the DGX Spark container build). |

---

### Legend
`[( … )]` = data/artifact on disk · `{ … }` = decision · dotted arrow = reuse /
runs-on · solid arrow = data or control flow.
