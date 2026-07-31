# I Built a Tiny GPT That Speaks Sanskrit in a Weekend — Here's What Broke

> *A from-scratch transformer, ~250 lines of Python, and one Unicode lesson that
> humbled me. Drop-in ready for Medium / HackerNoon. Replace the* `[👉 ...]`
> *markers with your own outputs and screenshots.*

---

I've understood transformers "at a high level" for a while now. Attention,
embeddings, the usual diagram with the arrows. But I'd never *built* one. And
there's a specific kind of knowledge you only get from typing out the code
yourself and watching it break.

So I gave myself a weekend and a constraint: build a GPT small enough to
understand completely, but on a language that would actually *fight back* —
Sanskrit. I have a pile of Devanagari text and an NVIDIA DGX Spark sitting on my
desk, so why not.

Here's the story of what I built, the one bug that taught me the most, and the
unglamorous problem that turned out to matter more than the neural network.

---

## The goal: predict the next *syllable*

A GPT does exactly one thing: **given some text, predict what comes next.**
Everything else — chat, translation, reasoning — is built on top of that single
reflex. So my whole model is just: feed it some Sanskrit, ask it to guess the
next unit, and correct it a few thousand times.

The interesting word there is *unit*. In English, "predict the next character" is
unambiguous. In Sanskrit, it's a trap.

---

## The gotcha: a "character" is a lie

The classic way to build a character-level model is one line:

```python
chars = sorted(set(text))   # the entire vocabulary
```

This is perfect for English and a disaster for Devanagari. Here's why. What your
eye reads as one syllable is often several Unicode code points stacked together:

```
कि   = क + ि          (2 code points, 1 syllable)
क्ष  = क + ्  + ष      (3 code points, 1 cluster)
श्री = श + ्  + र + ी   (4 code points, 1 cluster)
```

So `set(text)` doesn't give you Sanskrit characters. It gives you orthographic
shrapnel — bare consonants, floating vowel signs, lone viramas. Your model is
then forced to *re-learn Devanagari spelling from scratch* before it can learn
anything about the language, and it can happily generate nonsense like a vowel
sign attached to nothing.

The fix is to split on **grapheme clusters** — the chunks humans actually
perceive as characters — using the `regex` module's `\X`:

```python
import regex
tokens = regex.findall(r"\X", text)   # one akshara-ish unit per token
```

I put both tokenizers side by side and ran the same sentence through each:

```
Sample:  संस्कृतम् ज्ञानम् श्रीगणेशाय नमः

code-point :  स | ं | स | ् | क | ृ | त | म | ् | ... (32 tokens)
grapheme   :  सं | स्कृ | त | म् | ज्ञा | न | म् | श्री | ... (17 tokens)
```

Look at `स्कृ`, `ज्ञा`, `श्री` — whole and intact under grapheme splitting,
shattered under code-point splitting. And the measured trade-off on my toy
corpus was striking:

| tokenizer | vocab size | sequence length |
|-----------|-----------:|----------------:|
| code-point | 50 | 845 tokens |
| grapheme (akshara) | 160 | **476 tokens** |

The grapheme tokenizer has a bigger vocabulary but produces **44% shorter
sequences** — meaning the model sees far more real context in the same window,
and every token it emits is a *valid* on-screen syllable. You can't generate an
orphan vowel sign if no such token exists. I made invalid output unrepresentable
just by choosing the right unit. That single decision is the most important one
in the whole project.

> One honest caveat worth knowing: a Unicode grapheme cluster is *close to* a
> Sanskrit *akshara* but not identical — they were defined by different people for
> different goals. For a weekend model, `\X` is the pragmatic sweet spot.

`[👉 Screenshot idea: paste your own terminal output of` `python devanagari_tokenizer.py` `here.]`

---

## The model, in plain words

With tokenization sorted, the network itself is almost anticlimactic. It's a
small stack of transformer blocks, and each block does two things:

- **Attention** — every syllable looks back at the earlier ones and pulls in
  information from whichever it finds relevant. (One head might learn to bind a
  vowel sign to its consonant; another might learn that a *danda* `।` ends a
  clause. Nobody tells them to — they specialise on their own.)
- **A small MLP** — each position then "thinks" about what it just gathered.

Attention is communication; the MLP is computation. Stack four of those, add
embeddings at the bottom and a prediction layer at the top, and that's the entire
model. I wrote the attention math out by hand instead of calling a library, so I
could actually *see* the `query · key → mask → softmax → weighted values` dance.
If you want the line-by-line, it's all in my `TEACHING.md`.

`[👉 Optional: drop in the attention code snippet from train_sanskrit_gpt.py here.]`

---

## Training, and the first wobbly words

Training is a four-line loop repeated a few thousand times: predict, measure how
surprised the model was by the right answer, compute gradients, nudge the
weights. Watching the loss fall is weirdly addictive.

On my tiny starter corpus (twenty famous verses), here's a sample after a short
run:

```
[👉 PASTE YOUR OWN GENERATED SAMPLE HERE — run:
    python train_sanskrit_gpt.py
 and copy the text under "----- sample -----"]
```

Is it gibberish? Mostly, yes — and that's the point I want to make. Every "word"
it produced was built from **valid aksharas**, no broken syllables, because of
the tokenizer choice. The output is nonsense not because the model is broken, but
because it's *starving*. Twenty verses is nothing. Which brings me to the part
nobody warns you about.

`[👉 Screenshot idea: your loss curve, train vs val. Note where val loss turns
back up — that's overfitting on the tiny corpus.]`

---

## The real bottleneck wasn't the AI. It was the PDFs.

I assumed the neural network would be the hard part. It wasn't. The hard part was
getting *clean Devanagari text out of a PDF*. Three failure modes, roughly in
order of how often they bit me:

1. **Scanned images.** Half my "PDFs" were just photographs of pages. Text
   extraction returns almost nothing. The only fix is OCR.
2. **Legacy non-Unicode fonts.** Older Sanskrit typesetting fonts store glyphs at
   ASCII positions, so extraction hands you Latin-looking garbage instead of
   देवता. You need a font-specific re-mapper or, again, OCR.
3. **Reordered vowel signs.** Even clean Unicode PDFs sometimes emit marks in the
   wrong order. Normalising to NFC (`unicodedata.normalize("NFC", text)`) quietly
   fixes most of these.

So my data script doesn't pretend. It *measures* how much real Devanagari it
extracted from each file and refuses the bad ones, printing an OCR recipe
instead:

```
sample_corpus.txt   devanagari=100.0%  chars_kept=845   [OK]
old_scan.pdf        devanagari=  2.3%  chars_kept= 11    [⚠ LOW → needs OCR]
```

If you take one practical thing from this post: **budget most of your time for
data, not modelling.** The transformer is a solved, copy-pasteable artifact. A
clean, deduplicated, properly-encoded Sanskrit corpus is the actual moat — and
it's the thing the big multilingual models are *worst* at, because they treat
Sanskrit as a rounding error inside a hundred other languages.

`[👉 Screenshot idea: your own prepare_data.py output across your real PDFs.]`

---

## What this weekend actually bought me

I didn't build anything anyone needs to use. The output babbles. But I now
understand — in my hands, not just in a diagram — what an embedding is, what
attention computes, why the causal mask exists, what "loss went down" really
means, and why Indic NLP is genuinely harder than English NLP.

That last point is the strategic one. The off-the-shelf models are bad at
Sanskrit largely because of tokenization and data scarcity — exactly the two
things a focused individual can fix better than a big generalist team. The compute
isn't the moat. The care is.

Next, I'm going to stop pretraining toys and start fine-tuning a real
multilingual base model on a properly OCR'd corpus — but now I'll understand
exactly what every library call is doing under the hood. Which was the whole
point.

---

### Run it yourself

The full project — both tokenizers, the data pipeline, and the ~250-line model —
is here: `[👉 your repo link]`. It runs on a laptop CPU for the toy corpus and
flies on a DGX Spark.

```bash
pip install -r requirements.txt
python devanagari_tokenizer.py        # see the gotcha for yourself
python train_sanskrit_gpt.py --smoke  # 30-second sanity run
```

If you found the tokenization bit useful, that's the part I'd love feedback on —
how are *you* handling aksharas in your Indic NLP work?

`[👉 CTA: like / follow / subscribe — and tell me which ancient language I should
try next.]`
