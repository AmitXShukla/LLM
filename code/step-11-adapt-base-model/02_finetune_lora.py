"""
02_finetune_lora.py  —  STEP 2 of weekend 2
===========================================
This is the heart of weekend 2. We take a REAL pretrained model (billions of
parameters, trained by someone else on the whole internet) and teach it our
Sanskrit-assistant behaviour — cheaply, on one machine — using LoRA.

Read the sections top to bottom. Each one is a concept. The full narrative is in
TEACHING_finetune.md; this file is the concept "in the wild."

The five ideas you'll meet:
  1. A pretrained BASE model — we don't start from noise like weekend 1.
  2. QUANTIZATION (optional) — squash the frozen base to 4-bit to save memory.
  3. LoRA — freeze the giant base, train tiny "adapter" matrices instead.
  4. SFT with COMPLETION-ONLY loss — same next-token cross-entropy as weekend 1,
     but scored only on the answer.
  5. The Trainer — the loop from weekend 1, industrialised.

Try it safely first:
    python 02_finetune_lora.py --dry-run          # builds everything, downloads/ trains nothing
Then for real (on your DGX Spark):
    python 02_finetune_lora.py                     # LoRA, bf16
    python 02_finetune_lora.py --4bit              # QLoRA (4-bit base) — the GB10 sweet spot
"""

import argparse
import torch


def parse_args():
    p = argparse.ArgumentParser()
    # A small, ungated, multilingual model is the fast default for LEARNING.
    # For real Sanskrit quality, see the model table in README_finetune.md
    # (google/gemma-3-4b-it is a strong, Sanskrit-proven choice).
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="sanskrit_sft.jsonl")
    p.add_argument("--out", default="./sanskrit-lora-adapter")
    p.add_argument("--4bit", dest="fourbit", action="store_true",
                   help="QLoRA: load the base in 4-bit (needs bitsandbytes; ideal on GB10)")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--dry-run", action="store_true",
                   help="build configs + load data, but DON'T download the model or train")
    return p.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()

    # =======================================================================
    # SECTION 1 — The dataset (same file from step 1).
    # We load the JSONL into a Hugging Face `datasets.Dataset`. Because each row
    # has "prompt" and "completion" columns, TRL knows to compute loss on the
    # completion only (see SECTION 5).
    # =======================================================================
    from datasets import load_dataset
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"[data] loaded {len(ds)} examples with columns {ds.column_names}")

    # =======================================================================
    # SECTION 2 — LoRA config. THE key idea of efficient fine-tuning.
    #
    # A 1.5B model has 1.5 billion weights. Fine-tuning ALL of them needs huge
    # memory (weights + gradients + optimizer state ≈ 4x the model). LoRA's trick:
    # FREEZE every original weight, and next to each big weight matrix W, insert
    # two tiny matrices A (d×r) and B (r×d) with a small rank r. We only train
    # A and B. Their product BA is a low-rank "nudge" added to W. You end up
    # training well under 1% of the parameters, yet it's enough to teach new
    # behaviour — because behaviour lives in small directions, not in relearning
    # the whole language.
    #
    #   r          = rank = how much capacity the adapter has (8/16/32 typical)
    #   lora_alpha = a scaling knob; alpha/r sets the effective strength
    #   target_modules = which weight matrices get an adapter. "all-linear" is a
    #                    robust choice that works across Qwen/Gemma/Llama/Sarvam
    #                    without you memorising each family's layer names.
    # =======================================================================
    from peft import LoraConfig
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        # Explicit alternative, for understanding what "all-linear" expands to on
        # a Llama/Qwen-style model:
        # target_modules=["q_proj","k_proj","v_proj","o_proj",
        #                 "gate_proj","up_proj","down_proj"],
    )

    # =======================================================================
    # SECTION 3 — Training config (SFTConfig).
    #
    # completion_only_loss=True  -> score the answer, not the question (crucial).
    # bf16 (NOT fp16) on your DGX Spark: the GB10/Blackwell + FP16 combo has known
    #   numerical issues; bf16 is the recommended precision on this hardware.
    # gradient_checkpointing trades a little compute for a lot of memory headroom.
    # batch_size * grad_accum = your effective batch size.
    # =======================================================================
    from trl import SFTConfig
    sft = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,     # effective batch = 2 * 8 = 16
        learning_rate=2e-4,                # LoRA likes a higher LR than full fine-tuning
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        max_length=1024,
        completion_only_loss=True,         # <-- the important one
        gradient_checkpointing=True,
        bf16=use_cuda,                     # bf16 on GPU (Blackwell), fp32 on CPU
        report_to="none",
    )

    # =======================================================================
    # SECTION 4 — (optional) 4-bit quantization for QLoRA.
    #
    # The frozen base doesn't need full precision — we're not updating it. So we
    # can store it in 4-bit, cutting its memory ~4x. The LoRA adapters stay in
    # bf16. On the GB10's 128GB unified memory this is what lets you reach for
    # much bigger bases (up to ~70B) later. "nf4" is a 4-bit format tuned for
    # neural-net weight distributions; double-quant squeezes a little more.
    # =======================================================================
    quant_config = None
    if args.fourbit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,   # bf16, per the hardware note above
        )

    # ---- The dry-run stops here: everything above is built and validated, but
    # ---- we never touch the network or the GPU. Great for checking your setup.
    if args.dry_run:
        print("[dry-run] LoraConfig, SFTConfig"
              + (", BitsAndBytesConfig" if quant_config else "")
              + " built OK.")
        print(f"[dry-run] would fine-tune '{args.model}' "
              f"({'QLoRA 4-bit' if args.fourbit else 'LoRA bf16'}) "
              f"on {len(ds)} examples for {args.epochs} epochs.")
        print("[dry-run] remove --dry-run to download the model and train.")
        return

    # =======================================================================
    # SECTION 5 — Load the base model + tokenizer, then train.
    #
    # The tokenizer is the model's own — and here's a callback to weekend 1:
    # its handling of Devanagari is exactly the "fertility" problem you felt.
    # Indic-tuned tokenizers (e.g. Sarvam's) split Sanskrit into far fewer tokens
    # than English-centric ones, which is a real reason to prefer an Indic base.
    #
    # We hand the base model, the LoRA config, and the SFTConfig to SFTTrainer.
    # It applies LoRA, tokenizes, masks the prompt tokens (completion-only loss),
    # and runs the same predict -> loss -> backprop -> step loop you wrote by
    # hand in weekend 1 — just wrapped in a battle-tested Trainer.
    # =======================================================================
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds,
        peft_config=lora,               # <- turns this into a LoRA run
        processing_class=tokenizer,     # (this arg used to be called `tokenizer`)
    )

    # How little are we actually training? This line proves the LoRA claim.
    trainer.model.print_trainable_parameters()

    trainer.train()

    trainer.save_model(args.out)        # saves ONLY the small adapter (~tens of MB)
    tokenizer.save_pretrained(args.out)
    print(f"\nDone. LoRA adapter saved to {args.out}")
    print("Next: python 03_chat.py --adapter", args.out)


if __name__ == "__main__":
    main()
