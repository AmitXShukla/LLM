# code — runnable examples

One folder per step. Each folder is self-contained and has its own README.

```
code/
├── step-01-tiny-transformer/
├── step-04-sanskrit-tokenizer/
├── step-06-data-audit/
├── step-09-training/
└── step-11-adapt-base-model/
```

## Rules for code in this book

1. **It must run.** If it does not run on a clean machine, it does not belong
   here.
2. **Every folder gets a README** saying what it does, how to run it, what
   hardware it needs, and roughly how long it takes.
3. **No large files.** No data, no checkpoints, no model weights. Add a
   download script instead.
4. **Comment the surprising parts.** Do not comment `i = i + 1`. Do comment
   why the mask is applied where it is.
5. **Small over clever.** This is teaching code. A reader should be able to
   follow it top to bottom.

## Setting up

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
```

Pin your exact versions once things work. See Step 0.
