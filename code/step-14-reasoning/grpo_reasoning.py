"""
grpo_reasoning.py — teach a model to *think* with verifiable rewards (RLVR + GRPO)
=================================================================================
This is the engine behind reasoning models like DeepSeek-R1. The idea in one line:
if you can CHECK whether an answer is correct with a plain function, you don't
need human ratings or a reward model — the checker IS the reward.

Sanskrit is a beautiful fit for this because Panini's grammar makes many things
(sandhi, segmentation, meter) verifiable by rule. Here we use a simple math-style
example so it runs anywhere; swap `reward_fn` for a Panini/grammar checker to make
a Sanskrit reasoner.

Two ways to get reasoning into a model:
  A) DISTILLATION  — SFT on a strong reasoner's <think> traces. Cheap, stable,
     best for small models. (See distill_dataset_example.jsonl.)
  B) RLVR + GRPO   — the model explores, a verifier rewards correct answers, and
     reasoning *emerges*. That's this file.

Run (needs a GPU): python grpo_reasoning.py
"""

import re
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig


# ---------------------------------------------------------------------------
# THE HEART OF RLVR: a deterministic reward. No reward model, no human labels.
# We reward two things:
#   (1) format  — did it put its thinking in <think>...</think> ?  (+0.1)
#   (2) answer  — is the boxed final answer correct?               (+1.0)
# A model will exploit any gap here ("reward hacking"), so keep verifiers tight.
# ---------------------------------------------------------------------------
def reward_fn(completions, ground_truth, **kwargs):
    rewards = []
    for text, truth in zip(completions, ground_truth):
        r = 0.0
        if re.search(r"<think>.*?</think>", text, re.DOTALL):
            r += 0.1
        m = re.search(r"\\boxed\{([^}]*)\}", text)
        if m and m.group(1).strip() == str(truth).strip():
            r += 1.0
        rewards.append(r)
    return rewards


def main():
    # Dataset: prompts with a checkable answer. Columns: "prompt", "ground_truth".
    ds = load_dataset("json", data_files="reasoning_prompts.jsonl", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-3B-Instruct",
        reward_funcs=reward_fn,
        train_dataset=ds,
        args=GRPOConfig(
            output_dir="./sanskrit-reasoner",
            num_generations=8,            # GRPO samples a GROUP of answers per prompt
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-6,           # RL uses a very small LR
            bf16=True,                    # bf16 on Blackwell, never fp16
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model("./sanskrit-reasoner")
    print("Saved reasoning model to ./sanskrit-reasoner")


if __name__ == "__main__":
    main()
