# Step 22 — Patient behaviour from video 🎥

Per-frame CNN features → a temporal Transformer over the sequence → a behaviour
class (fall, seizure, agitation). Video = vision + time.

## Run it
```bash
pip install torch torchvision
python video_behavior.py --smoke     # synthetic clips, offline, ~15s
```
- **Hardware:** CPU for the smoke run; GPU for real clips.
- **Privacy first 🕶️:** for real patient video, prefer a **pose-based** pipeline
  (extract skeletal keypoints, model the pose sequence) — accurate, cheaper, and
  it stores no faces.
