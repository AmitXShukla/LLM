# Step 6 — Data audit: PDFs → clean corpus 🧹

Turns PDFs / text files in `./data` into one clean `corpus.txt`, and — crucially —
*measures* how much real Devanagari it extracted, flagging scanned or legacy-font
PDFs that need OCR. This is where real projects get stuck (not the neural net).

## Run it
```bash
pip install pypdf regex
python prepare_data.py             # reads ./data, writes corpus.txt, reports health
```
- **Hardware:** any CPU. **Time:** seconds per file.

If a file reports low Devanagari %, it's a scan or a legacy (non-Unicode) font —
the script prints a Tesseract OCR recipe to rescue it.
