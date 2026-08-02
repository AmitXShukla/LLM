# Step 2 — Break it 🔬

The Step 1 tiny transformer with five deliberate breakages behind flags, so you
can run each experiment from [Step 2](../../book/02-break-it.md) without
hand-editing (and without forgetting to put the causal mask back).

## Run it
```bash
pip install torch regex
python ablations.py                              # correct baseline (masked, pre-norm)
python ablations.py --no-mask                    # 1: remove causal mask  → loss cheats
python ablations.py --post-norm                  # 2: normalize AFTER the block → shaky
python ablations.py --no-residual --n-layer 10   # 3: drop residuals → won't learn
python ablations.py --lr 3e-3                     # 4: LR 10× too high → spikes / NaN
python ablations.py --tiny                       # 5: overfit tiny data → train↓ val↑
```
Add `--smoke` for a fast run. Watch the **train vs val** loss it prints.

- **Hardware:** CPU is fine for the built-in sample; a GPU for real corpora.
- **For textbook-clear curves**, point at a real Sanskrit file and train longer:
  ```bash
  python ablations.py --no-mask --corpus ../step-01-tiny-transformer/data/sample_corpus.txt --iters 5000
  ```
  On a few hundred steps of the tiny built-in sample the effects are only
  *directional* — the masked baseline memorises the sample quickly. The mask
  "cheat" (loss crashing to ~0 while samples turn to noise) shows up clearly once
  the baseline can no longer just memorise, i.e. on a real corpus with full training.

Everything except the flagged lines is identical to
`code/step-01-tiny-transformer/train_sanskrit_gpt.py`.
