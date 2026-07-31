"""
train_sanskrit_gpt.py
=====================
A tiny GPT, written from scratch, that learns to continue Sanskrit text one
*akshara* at a time. This is a deliberately small, readable re-implementation in
the spirit of Karpathy's nanoGPT — the goal is understanding, not benchmarks.

The whole pipeline:
    PDFs/txt --(prepare_data.py)--> corpus.txt --(this file)--> a model that
    babbles Sanskrit.

Everything here runs on a laptop CPU for the toy corpus, and flies on your DGX
Spark. Read TEACHING.md alongside this file — every class below maps to a
section there.

Usage:
    python train_sanskrit_gpt.py                 # real-ish run on corpus.txt
    python train_sanskrit_gpt.py --smoke         # 30-second sanity run
    python train_sanskrit_gpt.py --tokenizer codepoint   # see the naive way
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from devanagari_tokenizer import build_tokenizer


# ===========================================================================
# 0. Config. These defaults are a sensible weekend starting point. Scale them
#    UP on the DGX Spark (more layers, wider n_embd, longer block_size) once
#    your real corpus is in place.
# ===========================================================================
class Config:
    block_size = 128      # context length: how many aksharas the model sees at once
    n_embd     = 256      # width of the model (embedding dimension)
    n_head     = 4        # number of attention heads (n_embd must divide by this)
    n_layer    = 4        # number of transformer blocks stacked on top of each other
    dropout    = 0.1
    # training
    batch_size    = 32
    max_iters     = 3000
    eval_interval = 250
    eval_iters    = 50
    learning_rate = 3e-4


# ===========================================================================
# 1. Attention: the single idea that makes a transformer a transformer.
#    One head lets every position "look back" at earlier positions and pull in
#    information from the ones it finds relevant.
# ===========================================================================
class Head(nn.Module):
    def __init__(self, cfg: Config, head_size: int):
        super().__init__()
        # Each token emits a "query" (what am I looking for?), a "key" (what do I
        # offer?) and a "value" (what do I actually pass on if chosen?).
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # A lower-triangular mask so position t can only attend to <= t (causal).
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)        # (B, T, head_size)
        q = self.query(x)      # (B, T, head_size)
        # affinity between every query and every key, scaled to keep softmax sane
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        # mask out the future: a token may not peek at what comes after it
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)         # turn affinities into weights that sum to 1
        wei = self.dropout(wei)
        v = self.value(x)                    # (B, T, head_size)
        return wei @ v                       # weighted sum of values -> (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, then projected back to the model width.

    Different heads can specialise — one might track vowel signs, another might
    track word boundaries marked by dandas. We don't tell them what to do; they
    discover it.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """A little per-position MLP. Attention moves information *between* positions;
    this lets each position then *think* about what it gathered."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block = communicate (attention) then compute (MLP).
    The `x + ...` are residual connections; the LayerNorms stabilise training.
    Note we normalise BEFORE the sub-layer ("pre-norm"), which trains more nicely.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.sa  = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff  = FeedForward(cfg)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ===========================================================================
# 2. The full model: embeddings -> N blocks -> a prediction over the vocab.
# ===========================================================================
class SanskritGPT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        # what each akshara "means" (learned) ...
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        # ... and where it sits in the sequence (learned). Transformers have no
        # innate sense of order, so we hand them position explicitly.
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)   # project to a score per akshara

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                                  # (B, T, n_embd)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                         # (B, T, vocab)

        if targets is None:
            return logits, None
        # cross-entropy wants (N, vocab) vs (N,)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]     # never feed more than block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # focus on the last step
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ===========================================================================
# 3. Data plumbing + training loop.
# ===========================================================================
def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])   # targets are inputs shifted by 1
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--tokenizer", default="grapheme", choices=["grapheme", "codepoint"])
    ap.add_argument("--smoke", action="store_true", help="tiny/fast run to check wiring")
    ap.add_argument("--prompt", default="विद्या", help="seed text for the final sample")
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:                       # shrink everything for a quick sanity check
        cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer = 32, 64, 2, 2
        cfg.max_iters, cfg.eval_interval, cfg.eval_iters = 200, 50, 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    text = Path(args.corpus).read_text(encoding="utf-8")
    tok = build_tokenizer(text, args.tokenizer)
    print(f"device={device}  tokenizer={tok.name}  vocab_size={tok.vocab_size}  "
          f"corpus_tokens={len(tok.encode(text)):,}")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    model = SanskritGPT(cfg, tok.vocab_size).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, splits, cfg, device)
            print(f"iter {it:>5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch(splits["train"], cfg, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ---- sample some Sanskrit from the trained model ----
    print("\n----- sample -----")
    start = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    if start.numel() == 0:                       # prompt had no in-vocab tokens
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=200, temperature=0.8, top_k=20)
    print(tok.decode(out[0].tolist()))

    torch.save({"model": model.state_dict(), "vocab": tok.units, "cfg": vars(cfg)},
               "sanskrit_gpt.pt")
    print("\nsaved checkpoint -> sanskrit_gpt.pt")


if __name__ == "__main__":
    main()
