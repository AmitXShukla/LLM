"""
03_chat.py  —  STEP 3 of weekend 2
==================================
You trained an adapter. Now let's talk to it — and, more instructively, let's
see the model BEFORE vs AFTER your fine-tuning, side by side.

Key concept here: an adapter is NOT a whole model. It's the small set of LoRA
matrices from step 2. At load time we take the original base model and "attach"
the adapter on top. So loading = base + adapter. (You can also permanently merge
them into one model; see the note at the bottom.)

Run:
    python 03_chat.py --adapter ./sanskrit-lora-adapter
    python 03_chat.py --adapter ./sanskrit-lora-adapter --compare   # base vs tuned
    python 03_chat.py --adapter ./sanskrit-lora-adapter --ask "Translate: सत्यं वद।"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="must match the --model you fine-tuned in step 2")
    p.add_argument("--adapter", default="./sanskrit-lora-adapter")
    p.add_argument("--ask", default=None, help="one-shot question; omit for a chat loop")
    p.add_argument("--compare", action="store_true",
                   help="show base-model answer next to fine-tuned answer")
    return p.parse_args()


def build_prompt(tokenizer, user_text):
    """Wrap the user's text in the model's chat format.

    Every instruct model has a 'chat template' — the exact special-token layout
    it was trained to expect (who's the user, who's the assistant, where the turn
    ends). Using the tokenizer's own template means our prompt at inference time
    matches what training looked like. Get this wrong and even a good model acts
    confused.
    """
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate(model, tokenizer, user_text, device):
    prompt = build_prompt(tokenizer, user_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,     # a touch of randomness; lower = safer/more repetitive
        top_p=0.9,           # nucleus sampling: consider only the top 90% mass
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    # slice off the prompt so we only show the freshly generated answer
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The base model on its own...
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map=device
    )

    # ...and the same base with your adapter attached on top.
    from peft import PeftModel
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map=device),
        args.adapter,
    )

    def answer(text):
        if args.compare:
            print("\n" + "=" * 60)
            print("Q:", text)
            print("-" * 60)
            print("BASE  :", generate(base, tokenizer, text, device))
            print("TUNED :", generate(tuned, tokenizer, text, device))
        else:
            print("\nA:", generate(tuned, tokenizer, text, device))

    if args.ask:
        answer(args.ask)
        return

    print("Chat with your Sanskrit model. Type 'quit' to exit.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if text:
            answer(text)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# NOTE — merging, for later.
# To ship a single standalone model (no separate adapter at load time), merge:
#
#     from peft import PeftModel
#     from transformers import AutoModelForCausalLM
#     base  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
#     merged = PeftModel.from_pretrained(base, "./sanskrit-lora-adapter").merge_and_unload()
#     merged.save_pretrained("./sanskrit-merged")
#
# Merge only when you're happy with the adapter — keeping them separate while you
# iterate lets you train several adapters against one base cheaply.
# ---------------------------------------------------------------------------
