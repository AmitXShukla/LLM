"""
xray_finetune.py — X-ray -> fracture/finding, by transfer learning
==================================================================
Take a backbone pretrained on ImageNet, replace its head, and fine-tune it on
X-rays. This is the textbook medical-imaging recipe, and it's the SAME idea as
LoRA on a language model: reuse a big pretrained encoder, adapt a small part.

Two knobs you'll meet:
  * freeze_backbone=True  -> "linear probing": train only the new head. Fast,
    least data, safest against overfitting on small clinical sets.
  * freeze_backbone=False -> full fine-tune. Higher ceiling, needs more data.

Runs out of the box on SYNTHETIC images (--smoke). For real use, point
`--data` at an ImageFolder:  data/train/fracture/*.png , data/train/normal/*.png

    python xray_finetune.py --smoke          # random tensors, offline, ~15s
    python xray_finetune.py --data ./data    # real ImageFolder, ImageNet weights
"""

import argparse
import torch
import torch.nn as nn
import torchvision


def build_model(n_classes=2, pretrained=True, freeze_backbone=False):
    weights = "IMAGENET1K_V2" if pretrained else None
    m = torchvision.models.resnet50(weights=weights)
    if freeze_backbone:                       # linear probing
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_classes)   # new head: fracture / normal
    return m


def synthetic_batch(n=16, size=224, imbalance=0.2):
    """Fake X-rays: class 1 (fracture) is rare and has a faint bright streak."""
    x = torch.rand(n, 3, size, size) * 0.5
    y = (torch.rand(n) < imbalance).long()          # ~20% fractures
    for i in range(n):
        if y[i] == 1:
            r = torch.randint(20, size - 20, (1,)).item()
            x[i, :, r:r + 3, :] += 0.6              # a bright "fracture line"
    return x.clamp(0, 1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic data, offline")
    ap.add_argument("--data", default=None, help="path to an ImageFolder for real use")
    ap.add_argument("--freeze", action="store_true", help="linear probing (freeze backbone)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # In smoke mode we skip the pretrained download so it runs fully offline.
    model = build_model(n_classes=2, pretrained=not args.smoke,
                        freeze_backbone=args.freeze).to(device)

    # Up-weight the rare "fracture" class 3x so the model can't ignore it.
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    print(f"device={device}  trainable params={sum(p.numel() for p in trainable)/1e6:.2f}M  "
          f"mode={'linear-probe' if args.freeze else 'full fine-tune'}")

    if args.data:
        import torchvision.transforms as Tf
        from torch.utils.data import DataLoader
        tf = Tf.Compose([Tf.Resize((224, 224)), Tf.ToTensor(),
                         Tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # NOTE: RandomHorizontalFlip is OFF for chest X-rays — a flip puts the
        # heart on the wrong side. Only augment in clinically valid ways.
        ds = torchvision.datasets.ImageFolder(args.data, transform=tf)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        steps = [(xb, yb) for xb, yb in loader]
    else:
        steps = [synthetic_batch() for _ in range(20)]   # 20 fake batches

    model.train()
    for step, (xb, yb) in enumerate(steps):
        opt.zero_grad()
        loss = criterion(model(xb.to(device)), yb.to(device))
        loss.backward(); opt.step()
        if step % 5 == 0:
            print(f"  step {step:>3}: loss={loss.item():.3f}")

    print("\nEvaluate with AUROC/AUPRC per finding + sensitivity at a chosen")
    print("operating point, and ALWAYS check Grad-CAM saliency maps — models love")
    print("to 'cheat' by reading scanner markers instead of the anatomy.")


if __name__ == "__main__":
    main()
