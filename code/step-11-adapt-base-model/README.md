# Step 11 — Adapt a real base model with LoRA/QLoRA 🚀

Graduate from the toy to a real billions-of-parameters model, adapted cheaply on
one machine. Three staged scripts:

| file | what it teaches |
|------|-----------------|
| `01_make_dataset.py` | Build the instruction dataset (prompt/completion JSONL). Your bottleneck. |
| `02_finetune_lora.py` | LoRA / QLoRA supervised fine-tuning. Has a safe `--dry-run`. |
| `03_chat.py` | Chat with the result; `--compare` shows base vs. tuned side by side. |

## Run it
```bash
pip install -r requirements.txt
python 01_make_dataset.py
python 02_finetune_lora.py --dry-run     # validates setup, downloads/trains nothing
python 02_finetune_lora.py --4bit        # the real QLoRA run
python 03_chat.py --adapter ./sanskrit-lora-adapter --compare
```
- **Hardware:** a GPU (your DGX Spark is ideal). Use bf16, never fp16.
- **Time:** minutes on the tiny sample set; scale the dataset for real quality.
