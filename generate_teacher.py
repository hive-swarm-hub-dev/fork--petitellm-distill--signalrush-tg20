"""
Self-distillation teacher trainer + top-k logit cacher.

Called by prepare.sh on first run. Trains a modestly larger GPT on the FineWeb
sp1024 shards, then iterates over those shards in deterministic order and
caches top-k teacher log-probs per token.

Output (written to data/teacher_logits/):
    teacher_topk_indices.bin      int16   shape=(num_seq, seq_len, top_k)
    teacher_topk_logprobs.bin     float16 same shape
    meta.json                     alignment metadata

Alignment: sequence i in the cache corresponds to the i-th seq_len-chunk
obtained by concatenating the listed shards in the recorded order.

Do not modify this file. Agents should only modify train.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- small GPT -----------------------------
# Shared between train.py and generate_teacher.py via textual fork; we
# intentionally keep this compact (<200 lines) so it fits in 16MB cleanly.


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        hidden = dim * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary(q, k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab: int, dim: int, num_layers: int, num_heads: int, mlp_mult: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, dim)
        self.rope = RotaryEmbedding(dim // num_heads)
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_mult) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        # Weight tying
        self.head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope(T, idx.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        return self.head(x)  # logits


# ----------------------------- data -----------------------------


def load_shard(path: str) -> np.ndarray:
    """FineWeb binary shards are uint16 tokens preceded by a 256*4-byte header."""
    header = np.fromfile(path, dtype=np.int32, count=256)
    if len(header) == 0 or header[0] != 20240520:
        raise ValueError(f"bad shard header in {path}")
    ntoks = int(header[2])
    toks = np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(ntoks,))
    return np.array(toks, dtype=np.uint16)


def concat_shards(paths: list[str]) -> np.ndarray:
    pieces = [load_shard(p) for p in paths]
    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


# ----------------------------- teacher training -----------------------------


def train_teacher(args, shard_paths: list[str], device: torch.device) -> nn.Module:
    """Train a larger student on the same FineWeb shards for ~`max_seconds`."""
    print(f"[teacher] training: layers={args.teacher_layers} dim={args.teacher_dim} "
          f"heads={args.teacher_heads} mlp_mult={args.teacher_mlp} seq_len={args.seq_len}")
    model = GPT(
        vocab=args.vocab,
        dim=args.teacher_dim,
        num_layers=args.teacher_layers,
        num_heads=args.teacher_heads,
        mlp_mult=args.teacher_mlp,
        seq_len=args.seq_len,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr, betas=(0.9, 0.95), weight_decay=0.0)

    tokens = concat_shards(shard_paths)
    n = tokens.size
    # Randomized training sampling: draw random windows.
    rng = np.random.default_rng(args.seed)
    batch_size = args.teacher_batch
    seq_len = args.seq_len

    start = time.time()
    step = 0
    model.train()
    while True:
        if time.time() - start > args.teacher_seconds:
            break
        if step >= args.teacher_max_iters:
            break
        starts = rng.integers(0, n - seq_len - 1, size=batch_size)
        batch = np.stack([tokens[s : s + seq_len + 1] for s in starts]).astype(np.int64)
        batch_t = torch.from_numpy(batch).to(device, non_blocking=True)
        x = batch_t[:, :-1]
        y = batch_t[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, args.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            elapsed = time.time() - start
            print(f"[teacher] step={step} loss={loss.item():.4f} elapsed={elapsed:.0f}s", flush=True)
    print(f"[teacher] done: {step} steps, {time.time()-start:.0f}s")
    return model


# ----------------------------- logit caching -----------------------------


def cache_logits(args, model: nn.Module, shard_paths: list[str], device: torch.device, out_dir: str):
    model.eval()
    tokens = concat_shards(shard_paths)
    seq_len = args.seq_len
    top_k = args.top_k
    # Number of complete (x,y) pairs per shard sequence we cache.
    # We cache for inputs of length seq_len; each produces seq_len output positions.
    num_sequences = args.num_cache_sequences
    max_possible = (tokens.size - 1) // seq_len
    if num_sequences > max_possible:
        num_sequences = max_possible
    print(f"[cache] num_sequences={num_sequences} seq_len={seq_len} top_k={top_k} vocab={args.vocab}")

    idx_path = os.path.join(out_dir, "teacher_topk_indices.bin")
    lp_path = os.path.join(out_dir, "teacher_topk_logprobs.bin")

    # Pre-allocate memmapped output files.
    indices_mm = np.memmap(idx_path, dtype=np.int16, mode="w+", shape=(num_sequences, seq_len, top_k))
    logprobs_mm = np.memmap(lp_path, dtype=np.float16, mode="w+", shape=(num_sequences, seq_len, top_k))

    batch_size = args.cache_batch
    with torch.inference_mode():
        for bstart in range(0, num_sequences, batch_size):
            bend = min(bstart + batch_size, num_sequences)
            seqs = []
            for i in range(bstart, bend):
                s = i * seq_len
                seqs.append(tokens[s : s + seq_len].astype(np.int64))
            x = torch.from_numpy(np.stack(seqs)).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)  # (B, T, V)
            logp = F.log_softmax(logits.float(), dim=-1)
            topv, topi = torch.topk(logp, k=top_k, dim=-1)  # (B, T, K)
            indices_mm[bstart:bend] = topi.to(torch.int16).cpu().numpy()
            logprobs_mm[bstart:bend] = topv.to(torch.float16).cpu().numpy()
            if bstart % (batch_size * 20) == 0:
                print(f"[cache] {bend}/{num_sequences}", flush=True)

    indices_mm.flush()
    logprobs_mm.flush()
    del indices_mm, logprobs_mm

    meta = {
        "seq_len": seq_len,
        "top_k": top_k,
        "num_sequences": num_sequences,
        "vocab_size": args.vocab,
        "shard_order": [os.path.basename(p) for p in shard_paths],
        "indices_dtype": "int16",
        "logprobs_dtype": "float16",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] wrote {idx_path}, {lp_path}, meta.json")


# ----------------------------- main -----------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/datasets/fineweb10B_sp1024")
    p.add_argument("--out-dir", default="data/teacher_logits")
    p.add_argument("--teacher-checkpoint", default=None,
                   help="Optional: path to a pre-trained teacher .pt; if set, skip training and only cache.")
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1337)
    # Teacher model defaults (modest — fits in ~4 min on 1xA100).
    p.add_argument("--teacher-layers", type=int, default=8)
    p.add_argument("--teacher-dim", type=int, default=512)
    p.add_argument("--teacher-heads", type=int, default=8)
    p.add_argument("--teacher-mlp", type=int, default=4)
    p.add_argument("--teacher-lr", type=float, default=3e-4)
    p.add_argument("--teacher-batch", type=int, default=16)
    p.add_argument("--teacher-seconds", type=float, default=240.0)
    p.add_argument("--teacher-max-iters", type=int, default=4000)
    # Cache defaults (int16 indices + fp16 logprobs; ~600MB at these settings).
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--num-cache-sequences", type=int, default=9766)  # ~10M tokens at seq_len=1024
    p.add_argument("--cache-batch", type=int, default=32)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for generate_teacher.py", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")

    shard_paths = sorted(glob.glob(os.path.join(args.data_dir, "fineweb_train_*.bin")))
    if not shard_paths:
        print(f"ERROR: no train shards in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.teacher_checkpoint and os.path.exists(args.teacher_checkpoint):
        print(f"[teacher] loading checkpoint {args.teacher_checkpoint}")
        model = GPT(
            vocab=args.vocab, dim=args.teacher_dim, num_layers=args.teacher_layers,
            num_heads=args.teacher_heads, mlp_mult=args.teacher_mlp, seq_len=args.seq_len,
        ).to(device)
        sd = torch.load(args.teacher_checkpoint, map_location=device)
        model.load_state_dict(sd)
    else:
        model = train_teacher(args, shard_paths, device)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "teacher_checkpoint.pt"))

    cache_logits(args, model, shard_paths, device, args.out_dir)
    print("[done]")


if __name__ == "__main__":
    main()
