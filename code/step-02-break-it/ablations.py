"""
ablations.py — Step 2: break your Step 1 model on purpose, one flag at a time.
=============================================================================
This is the Step 1 tiny transformer with five deliberate breakages wired behind
flags, so you can run each experiment without hand-editing (and without ever
forgetting to put the mask back).

    python ablations.py                 # the correct baseline (pre-norm, masked)
    python ablations.py --no-mask       # 1: remove the causal mask  -> loss cheats to ~0
    python ablations.py --post-norm     # 2: normalize AFTER the block -> shaky
    python ablations.py --no-residual --n-layer 10   # 3: drop residuals -> won't learn
    python ablations.py --lr 3e-3       # 4: LR 10x too high -> loss spikes / NaN
    python ablations.py --tiny          # 5: overfit tiny data -> train down, val UP

Add --smoke for a fast ~15s run. Watch the train vs val loss it prints.
Everything else is identical to code/step-01-tiny-transformer/train_sanskrit_gpt.py.
"""

import argparse
import regex
import torch
import torch.nn as nn
from torch.nn import functional as F

# A small public-domain Sanskrit sample so this runs with no download. Enough
# distinct verses that the masked baseline can't trivially memorise it — which is
# what makes the --no-mask "cheat" stand out. For textbook-clear curves, point at
# your real Step 1 corpus with --corpus path/to/sample_corpus.txt
VERSES = [
    "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥",
    "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय। सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते॥",
    "वासांसि जीर्णानि यथा विहाय नवानि गृह्णाति नरोऽपराणि। तथा शरीराणि विहाय जीर्णान्यन्यानि संयाति नवानि देही॥",
    "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज। अहं त्वा सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः॥",
    "विद्या ददाति विनयं विनयाद्याति पात्रताम्। पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम्॥",
    "उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः। न हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः॥",
    "अयं निजः परो वेति गणना लघुचेतसाम्। उदारचरितानां तु वसुधैव कुटुम्बकम्॥",
    "सत्यं ब्रूयात्प्रियं ब्रूयान्न ब्रूयात्सत्यमप्रियम्। प्रियं च नानृतं ब्रूयादेष धर्मः सनातनः॥",
    "काव्यशास्त्रविनोदेन कालो गच्छति धीमताम्। व्यसनेन च मूर्खाणां निद्रया कलहेन वा॥",
    "पुस्तकस्था तु या विद्या परहस्तगतं धनम्। कार्यकाले समुत्पन्ने न सा विद्या न तद्धनम्॥",
    "अलसस्य कुतो विद्या अविद्यस्य कुतो धनम्। अधनस्य कुतो मित्रम् अमित्रस्य कुतः सुखम्॥",
    "आलस्यं हि मनुष्याणां शरीरस्थो महान् रिपुः। नास्त्युद्यमसमो बन्धुः कृत्वा यं नावसीदति॥",
]
SAMPLE = (" ".join(VERSES) + " ") * 6  # a bit of repetition for enough windows


def graphemes(text):
    return regex.findall(r"\X", text)  # one akshara (grapheme cluster) at a time


class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)
        self.use_mask = cfg.use_mask  # <-- Experiment 1

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        if self.use_mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.sa = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)
        self.residual = cfg.residual  # <-- Experiment 3
        self.post_norm = cfg.post_norm  # <-- Experiment 2

    def forward(self, x):
        if self.post_norm:                       # Experiment 2: normalize AFTER
            x = self.ln1(x + self.sa(x))
            x = self.ln2(x + self.ff(x))
        elif self.residual:                      # Step 1 default: pre-norm + residual
            x = x + self.sa(self.ln1(x))
            x = x + self.ff(self.ln2(x))
        else:                                    # Experiment 3: no residual
            x = self.sa(self.ln1(x))
            x = self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss


def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(20)
        for k in range(20):
            xb, yb = get_batch(data, cfg, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mask", action="store_true", help="Exp 1: remove causal mask")
    ap.add_argument("--post-norm", action="store_true", help="Exp 2: normalize after")
    ap.add_argument("--no-residual", action="store_true", help="Exp 3: drop residuals")
    ap.add_argument("--lr", type=float, default=3e-4, help="Exp 4: try 3e-3")
    ap.add_argument("--tiny", action="store_true", help="Exp 5: overfit tiny data")
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--corpus", default=None, help="path to a real Sanskrit text file")
    ap.add_argument("--iters", type=int, default=None, help="override training iterations")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Cfg:
        block_size = 64
        n_embd = 128
        n_head = 4
        n_layer = args.n_layer
        dropout = 0.1
        batch_size = 32
        max_iters = args.iters if args.iters else (300 if args.smoke else 3000)
        learning_rate = args.lr
        use_mask = not args.no_mask
        post_norm = args.post_norm
        residual = not args.no_residual

    cfg = Cfg()
    text = open(args.corpus, encoding="utf-8").read() if args.corpus else SAMPLE
    units = sorted(set(graphemes(text)))
    stoi = {u: i for i, u in enumerate(units)}
    ids = torch.tensor([stoi[g] for g in graphemes(text)], dtype=torch.long)

    if args.tiny:                       # Experiment 5: memorise a tiny slice
        ids = ids[: max(cfg.block_size + 5, len(ids) // 12)]
    n = int(0.9 * len(ids))
    splits = {"train": ids[:n], "val": ids[n:]}

    torch.manual_seed(0)
    model = TinyGPT(cfg, len(units)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    label = (("no-mask " if args.no_mask else "") + ("post-norm " if args.post_norm else "")
             + ("no-residual " if args.no_residual else "") + ("tiny " if args.tiny else "")
             + (f"lr={cfg.learning_rate} " if cfg.learning_rate != 3e-4 else "")) or "baseline"
    print(f"[{label.strip()}]  vocab={len(units)}  n_layer={cfg.n_layer}  device={device}")

    every = 100 if args.smoke else 500
    for it in range(cfg.max_iters + 1):
        if it % every == 0:
            L = estimate(model, splits, cfg, device)
            print(f"iter {it:>5}: train {L['train']:.3f}  val {L['val']:.3f}")
        xb, yb = get_batch(splits["train"], cfg, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print("\nWrite ONE sentence about what you just saw, then run the next experiment.")


if __name__ == "__main__":
    main()
