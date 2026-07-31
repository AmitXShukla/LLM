# Step 4 — Devanagari tokenizer 🔤

The most important file in the whole project. Shows *why* a "character" is a lie
in Devanagari: a syllable you see is often several Unicode code points. Contains
two tokenizers (naive code-point vs. grapheme/akshara) so you can see the gap.

## Run it
```bash
pip install regex
python devanagari_tokenizer.py     # prints a side-by-side of both tokenizers
```
- **Hardware:** any CPU. **Time:** instant.

You'll watch `स्कृ`, `ज्ञा`, `श्री` stay whole under grapheme splitting and
shatter under code-point splitting — the central Sanskrit tokenization lesson.
