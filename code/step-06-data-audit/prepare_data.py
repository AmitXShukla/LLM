"""
prepare_data.py
===============
Turns whatever you throw into the ./data folder (PDFs and/or .txt files) into a
single clean corpus.txt that train_sanskrit_gpt.py can eat.

This script is short, but it is where your project will actually get stuck — not
in the neural network. Getting clean Devanagari *out of a PDF* is the real
bottleneck, and this file is honest about it: it measures how much real
Devanagari it found and tells you when a PDF needs OCR instead.

Three things that go wrong with Sanskrit PDFs, in order of how often they bite:

  1. SCANNED IMAGES. The "PDF" is just photos of pages. Text extraction returns
     almost nothing. Fix: OCR (see the printed advice at the end).

  2. LEGACY (non-Unicode) FONTS. Old typesetting fonts (Shusha, Yogesh, DVB-TT…)
     store glyphs at ASCII positions, so extraction returns Latin-looking garbage
     like "Ÿ‚ŸàdÊ" instead of देवता. Fix: a font-specific re-mapper, or OCR.

  3. BROKEN LIGATURES / REORDERING. Even good Unicode PDFs sometimes emit vowel
     signs in the wrong visual order. NFC normalization (which we do) fixes most.
"""

import sys
import glob
import unicodedata
from pathlib import Path

DEV_START, DEV_END = 0x0900, 0x097F          # the Devanagari Unicode block
DANDA, DOUBLE_DANDA = "\u0964", "\u0965"      # । and ॥ , the Sanskrit "full stops"


def is_devanagari(ch: str) -> bool:
    return DEV_START <= ord(ch) <= DEV_END


def devanagari_ratio(text: str) -> float:
    """Fraction of non-space characters that are actual Devanagari. Our health check."""
    meaningful = [c for c in text if not c.isspace()]
    if not meaningful:
        return 0.0
    dev = sum(1 for c in meaningful if is_devanagari(c))
    return dev / len(meaningful)


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("Please `pip install pypdf` to read PDFs (or feed .txt files instead).")
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def clean(text: str) -> str:
    """Keep Devanagari + dandas + whitespace; drop everything else.

    NFC normalization is the important line: it composes characters into the
    canonical stacked form, so 'क' + 'ि' is stored consistently every time. Skip
    this and your tokenizer will think two visually-identical syllables are
    different tokens.
    """
    text = unicodedata.normalize("NFC", text)
    kept = []
    for ch in text:
        if is_devanagari(ch) or ch in (DANDA, DOUBLE_DANDA) or ch.isspace():
            kept.append(ch)
    cleaned = "".join(kept)
    # collapse runs of blank lines / spaces
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines) + "\n"


def main():
    data_dir = Path("data")
    out_path = Path("corpus.txt")
    sources = sorted(glob.glob(str(data_dir / "*.pdf"))) + \
              sorted(glob.glob(str(data_dir / "*.txt")))

    if not sources:
        sys.exit("No .pdf or .txt files found in ./data — drop your files there first.")

    chunks, needs_ocr = [], []
    print(f"Scanning {len(sources)} file(s) in ./data …\n")

    for src in sources:
        src = Path(src)
        raw = extract_pdf_text(src) if src.suffix.lower() == ".pdf" else src.read_text(encoding="utf-8", errors="ignore")
        ratio = devanagari_ratio(raw)
        cleaned = clean(raw)
        verdict = "OK" if ratio >= 0.30 else "⚠ LOW"
        print(f"  {src.name:<30} devanagari={ratio:5.1%}  chars_kept={len(cleaned):>7}  [{verdict}]")
        if ratio < 0.30:
            needs_ocr.append(src.name)
        else:
            chunks.append(cleaned)

    corpus = "\n".join(chunks)
    out_path.write_text(corpus, encoding="utf-8")

    print(f"\nWrote {out_path}  ({len(corpus):,} characters from {len(chunks)} usable file(s)).")

    if needs_ocr:
        print("\n" + "=" * 70)
        print("These files looked like scanned images or legacy fonts (very little")
        print("real Devanagari was extracted):")
        for n in needs_ocr:
            print(f"   - {n}")
        print("\nTo rescue them, OCR the pages. Quick recipe with Tesseract:")
        print("   sudo apt install tesseract-ocr tesseract-ocr-san tesseract-ocr-hin")
        print("   # convert PDF pages to images, then:")
        print("   tesseract page.png out -l san+hin")
        print("Then save the resulting text as a .txt in ./data and re-run this.")
        print("=" * 70)


if __name__ == "__main__":
    main()
