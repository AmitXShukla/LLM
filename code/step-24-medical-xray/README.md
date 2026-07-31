# Step 24 (X-ray) — Fracture / findings by transfer learning 🦴

Take an ImageNet-pretrained ResNet, replace the head, fine-tune on X-rays. Same
idea as LoRA on a language model: reuse a big encoder, adapt a small part.

## Run it
```bash
pip install torch torchvision
python xray_finetune.py --smoke              # synthetic images, fully offline
python xray_finetune.py --data ./data        # real ImageFolder (train/fracture, train/normal)
python xray_finetune.py --smoke --freeze     # linear probing (train only the head)
```
- **Hardware:** CPU works for the smoke run; GPU for real training.
- **Watch out:** the rare class is up-weighted; augment only in clinically valid
  ways (no horizontal flip on chest X-rays!); check Grad-CAM for shortcut learning.
