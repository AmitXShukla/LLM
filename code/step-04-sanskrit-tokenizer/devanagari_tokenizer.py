"""
devanagari_tokenizer.py
=======================
This is the most important file in the whole project.

If you remember one thing about building anything for Sanskrit (or Hindi, or
almost any Indic script), let it be this: **a "character" is not what you think
it is.**

In English, the unit you see on the page (a letter) and the unit the computer
stores (a Unicode code point) are basically the same thing. "cat" is 3 glyphs
and 3 code points. Life is simple.

Devanagari breaks that one-to-one assumption. What your eye reads as a single
syllable (an *akshara*) is very often built from SEVERAL code points stacked
together:

    कि   = क (U+0915 consonant KA) + ि (U+093F vowel sign I)      -> 2 code points, 1 syllable
    क्ष  = क (KA) + ् (U+094D VIRAMA) + ष (SHA)                   -> 3 code points, 1 cluster
    श्री = श + ् + र + ी                                          -> 4 code points, 1 cluster

The naive nanoGPT move is `chars = sorted(set(text))`, which splits text into
code points. On English that's a perfect character-level tokenizer. On Sanskrit
it shreds every syllable into orthographic atoms and forces your tiny model to
re-learn the rules of Devanagari spelling from scratch — a waste of its limited
capacity, and a recipe for generating *invalid* text (stray vowel signs with no
consonant to attach to).

This file gives you BOTH tokenizers so you can see the difference with your own
eyes (run `python devanagari_tokenizer.py`), then defaults you to the right one.
"""

import regex  # NOTE: third-party `regex`, not the stdlib `re`. We need its \X support.


# ---------------------------------------------------------------------------
# Tokenizer 1: the naive, "wrong-for-Devanagari" one. Kept on purpose.
# ---------------------------------------------------------------------------
class CharTokenizer:
    """Code-point level tokenizer.

    This is exactly the classic nanoGPT tokenizer. It is correct and great for
    English. We keep it so you can *measure* how badly it treats Sanskrit.
    """

    name = "codepoint"

    def __init__(self, text: str):
        self.units = sorted(set(text))                 # each "unit" is one code point
        self.stoi = {u: i for i, u in enumerate(self.units)}
        self.itos = {i: u for u, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.units)

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# Tokenizer 2: the one you actually want. Grapheme clusters (~= aksharas).
# ---------------------------------------------------------------------------
class GraphemeTokenizer:
    r"""Grapheme-cluster tokenizer.

    `regex`'s `\X` matches a full "extended grapheme cluster" — the chunk a human
    perceives as one character. Recent Unicode versions even group Devanagari
    conjuncts (consonant + virama + consonant) into a single cluster, so क्ष
    tends to come out as ONE token instead of three.

    Caveat worth knowing (and worth a paragraph in your blog): a Unicode grapheme
    cluster is *close to* a Sanskrit akshara but not identical. The linguistic
    akshara and Unicode's clustering rules were designed by different people for
    different reasons. For a weekend model, `\X` is the pragmatic sweet spot.
    """

    name = "grapheme"

    def __init__(self, text: str):
        self.units = sorted(set(self._split(text)))    # each "unit" is one grapheme cluster
        self.stoi = {u: i for i, u in enumerate(self.units)}
        self.itos = {i: u for u, i in self.stoi.items()}

    @staticmethod
    def _split(text: str):
        # \X = one extended grapheme cluster. This single regex is the whole trick.
        return regex.findall(r"\X", text)

    @property
    def vocab_size(self) -> int:
        return len(self.units)

    def encode(self, s: str):
        out = []
        for g in self._split(s):
            if g in self.stoi:
                out.append(self.stoi[g])
            # Unknown clusters (e.g. a syllable never seen in training) are skipped.
            # A real system would add an <unk> token; we keep it simple here.
        return out

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def build_tokenizer(text: str, kind: str = "grapheme"):
    """Factory so train.py can switch with one flag."""
    if kind == "grapheme":
        return GraphemeTokenizer(text)
    if kind == "codepoint":
        return CharTokenizer(text)
    raise ValueError(f"unknown tokenizer kind: {kind!r} (use 'grapheme' or 'codepoint')")


# ---------------------------------------------------------------------------
# Run this file directly to SEE the gotcha. This is your first blog screenshot.
# ---------------------------------------------------------------------------
def demo():
    sample = "संस्कृतम् ज्ञानम् श्रीगणेशाय नमः"
    print("Sample text:\n   ", sample, "\n")

    cp = CharTokenizer(sample)
    gr = GraphemeTokenizer(sample)

    cp_units = list(sample)                  # what code-point splitting produces
    gr_units = GraphemeTokenizer._split(sample)

    print(f"Code-point tokens ({len(cp_units)} of them, vocab={cp.vocab_size}):")
    print("   ", " | ".join(cp_units), "\n")

    print(f"Grapheme tokens  ({len(gr_units)} of them, vocab={gr.vocab_size}):")
    print("   ", " | ".join(gr_units), "\n")

    print("Notice:")
    print(" - 'कि'-style syllables stay whole under grapheme splitting but shatter "
          "under code-point splitting.")
    print(" - Grapheme tokens are FEWER (shorter sequences => the model sees more "
          "real context per step).")
    print(" - Every grapheme token is a *valid* on-screen unit, so generated text "
          "can't contain orphan vowel signs.")

    # Prove round-tripping works
    ids = gr.encode(sample)
    assert gr.decode(ids) == sample, "grapheme round-trip failed!"
    print("\nRound-trip (encode->decode) is lossless for the grapheme tokenizer. ✓")


if __name__ == "__main__":
    demo()
