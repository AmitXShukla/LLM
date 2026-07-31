# Step 1 — Tiny Sanskrit transformer (from scratch) 🧠

A ~250-line GPT written by hand — no `transformers` import — that learns to
continue Sanskrit one *akshara* (syllable) at a time. This is the "feel it in
your hands" model from Chapter 1.

## Run it
```bash
pip install torch regex
python train_sanskrit_gpt.py --smoke     # 30-second sanity run on the sample corpus
python train_sanskrit_gpt.py             # the real (still tiny) run
python train_sanskrit_gpt.py --tokenizer codepoint   # see the naive way, for contrast
```

- **Hardware:** runs on a laptop CPU for the sample corpus; instant on any GPU.
- **Time:** seconds (smoke) to a few minutes.
- **Data:** `data/sample_corpus.txt` (public-domain verses). Drop in your own text to feel Sanskrit's real problems on day one.

The model babbles at first — that is expected. Every "word" it makes is a *valid*
akshara, because of the tokenizer choice (see Step 4).
