# Step 24 (ECG) — Heartbeat → arrhythmia ❤️

A 1D ResNet that classifies 12-lead ECG windows into rhythm classes. Runs on
**synthetic ECG** so you see the whole pipeline today; swap in PTB-XL or MIT-BIH
for real data.

## Run it
```bash
pip install torch numpy
python ecg_arrhythmia.py --smoke     # ~10s, a few steps
python ecg_arrhythmia.py             # a fuller run
```
- **Hardware:** CPU is fine for the toy data; a GPU for real datasets.
- **The lesson:** the dangerous class is rare, so it uses a **class-weighted loss**
  and reports **per-class recall**, never plain accuracy (which would look great
  while missing every arrhythmia).
