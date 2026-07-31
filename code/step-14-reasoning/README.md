# Step 14 — Teach it to reason (RLVR + GRPO) 🧩

The engine behind "thinking" models. If you can *check* an answer with a plain
function, that checker becomes the reward — no human ratings, no reward model.

- `grpo_reasoning.py` — the reward function + `GRPOTrainer` setup.
- `reasoning_prompts.jsonl` — prompts with checkable answers (for RLVR).
- `distill_dataset_example.jsonl` — the cheaper alternative: SFT on `<think>` traces.

## Run it
```bash
pip install trl datasets
python grpo_reasoning.py           # needs a GPU
```
- **Hardware:** a GPU. **Time:** long (RL is expensive) — distillation is the cheaper default.

Swap the math-style reward for a **Panini grammar checker** to make a Sanskrit
reasoner whose correctness is verifiable by rule (see Chapter 15).
