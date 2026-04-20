"""
PetiteLLM: Distillation — student training script.

Baseline: small GPT student trained on FineWeb sp1024 with a mixed
hard-label + soft-label (teacher top-k KL) loss. Student and teacher share
the sp1024 vocabulary, so teacher top-k indices can be directly gathered
from student logits.

Outputs (at end of run):
    final_model.ptz                  zlib-compressed fp16 state dict
    stdout line: "final_int8_zlib_roundtrip_exact val_bpb:<float>"
    stdout line: "Total submission size: <bytes> bytes"

Agents should iterate on ARCHITECTURE, OPTIMIZER, DISTILL HPARAMS, QUANTIZATION,
etc. to minimize val_bpb.
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- hyperparameters -----------------------------


class HP:
    data_dir = os.environ.get("DATA_DIR", "data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "data/tokenizers/fineweb_1024_bpe.model")
    teacher_dir = os.environ.get("TEACHER_DIR", "data/teacher_logits")

    seed = int(os.environ.get("SEED", 1337))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Student architecture
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    model_dim = int(os.environ.get("MODEL_DIM", 448))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    seq_len = int(os.environ.get("SEQ_LEN", 1024))

    # Optimizer
    lr = float(os.environ.get("LR", 4e-4))          # AdamW lr (for embeds/LN/biases)
    muon_lr = float(os.environ.get("MUON_LR", 0.02))  # Muon lr (for hidden 2D weights)
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    lr_min_mult = float(os.environ.get("LR_MIN_MULT", 0.05))  # final lr = lr * lr_min_mult
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.1))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))

    # Distillation
    distill_alpha = float(os.environ.get("DISTILL_ALPHA", 0.5))
    distill_temperature = float(os.environ.get("DISTILL_T", 2.0))
    teacher_top_k = int(os.environ.get("TEACHER_TOP_K", 32))
    # Fraction of batches that use cached-teacher sequences (the rest sample
    # anywhere in the raw shards with hard-CE only). Teacher cache covers
    # only ~10M tokens but shards have ~200M — expanding the pool prevents
    # the student from memorizing the 9766 cached sequences.
    teacher_batch_frac = float(os.environ.get("TEACHER_BATCH_FRAC", 0.1))


# ----------------------------- model -----------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_mult, is_first=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_first = is_first
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
        # Value residual [Zhu et al. 2024]: non-first layers learn a blend
        # between their own v and layer-0's v. Init 0.5 = equal mix.
        if not is_first:
            self.lambda_v = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, cos, sin, v_first):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary(q, k, cos, sin)
        if self.is_first:
            v_first = v
        else:
            v = self.lambda_v * v_first + (1.0 - self.lambda_v) * v
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x, v_first


class GPT(nn.Module):
    def __init__(self, vocab, dim, num_layers, num_heads, mlp_mult, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, dim)
        self.rope = RotaryEmbedding(dim // num_heads)
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_mult, is_first=(i == 0))
            for i in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        # Untied head with zero init (nanoGPT speedrun): input embedding and
        # output projection play different roles; untying adds ~460K params and
        # zero init starts logits uniform so no token is preferred before training.
        self.head = nn.Linear(dim, vocab, bias=False)
        # GPT-2 style init: N(0, 0.02) for linears/embeds, proj layers scaled by sqrt(2L).
        self.apply(self._init_weights)
        proj_std = 0.02 / math.sqrt(2 * num_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, mean=0.0, std=proj_std)
            nn.init.normal_(blk.mlp[2].weight, mean=0.0, std=proj_std)
        # Zero-init the output head. CE with uniform predictions starts at
        # log(vocab) and gradients are pure signal (no random head to fight).
        nn.init.zeros_(self.head.weight)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope(T, idx.device, x.dtype)
        v_first = None
        for blk in self.blocks:
            x, v_first = blk(x, cos, sin, v_first)
        x = self.ln_f(x)
        return self.head(x)


# ----------------------------- data -----------------------------


def load_shard(path: str) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    if len(header) == 0 or header[0] != 20240520:
        raise ValueError(f"bad shard header in {path}")
    ntoks = int(header[2])
    toks = np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(ntoks,))
    return np.array(toks, dtype=np.uint16)


def concat_shards(paths):
    pieces = [load_shard(p) for p in paths]
    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


class DistillationDataLoader:
    """Yields (x, y, teacher_top_idx, teacher_top_lp) aligned to the teacher cache.

    Reads shards in the exact order recorded in meta.json. Sequence i in the
    concatenated token stream maps 1:1 to sequence i in the teacher cache,
    indexed by (i // batch_size) * batch_size + offset.
    """

    def __init__(self, data_dir: str, teacher_dir: str, seq_len: int, batch_size: int, seed: int, teacher_batch_frac: float = 0.5):
        with open(os.path.join(teacher_dir, "meta.json")) as f:
            meta = json.load(f)
        self.meta = meta
        if meta["seq_len"] != seq_len:
            raise ValueError(f"meta seq_len={meta['seq_len']} != requested {seq_len}")
        self.top_k = meta["top_k"]
        self.num_sequences = meta["num_sequences"]
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.teacher_batch_frac = teacher_batch_frac

        shard_paths = [os.path.join(data_dir, name) for name in meta["shard_order"]]
        for p in shard_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"shard missing: {p}")
        self.tokens = concat_shards(shard_paths)
        assert self.tokens.size >= self.num_sequences * seq_len + 1, \
            "token stream too short for cached sequences"
        # Total sequences available in the raw shards (200M tokens).
        self.total_raw_seqs = (self.tokens.size - 1) // seq_len

        self.indices = np.memmap(
            os.path.join(teacher_dir, "teacher_topk_indices.bin"),
            dtype=np.int16, mode="r",
            shape=(self.num_sequences, seq_len, self.top_k),
        )
        self.logprobs = np.memmap(
            os.path.join(teacher_dir, "teacher_topk_logprobs.bin"),
            dtype=np.float16, mode="r",
            shape=(self.num_sequences, seq_len, self.top_k),
        )
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, device):
        # Per-batch decision: cached+distill, or raw+hard-only.
        use_teacher = self.rng.random() < self.teacher_batch_frac
        if use_teacher:
            idxs = self.rng.integers(0, self.num_sequences, size=self.batch_size)
        else:
            idxs = self.rng.integers(0, self.total_raw_seqs, size=self.batch_size)
        # Build (x, y) from the concatenated token stream.
        x_list, y_list = [], []
        for i in idxs:
            s = int(i) * self.seq_len
            e = s + self.seq_len + 1
            chunk = self.tokens[s:e].astype(np.int64)
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])
        x = torch.from_numpy(np.stack(x_list)).to(device, non_blocking=True)
        y = torch.from_numpy(np.stack(y_list)).to(device, non_blocking=True)
        if use_teacher:
            ti = torch.from_numpy(np.stack([self.indices[i] for i in idxs]).astype(np.int64)).to(device, non_blocking=True)
            tl = torch.from_numpy(np.stack([self.logprobs[i] for i in idxs]).astype(np.float32)).to(device, non_blocking=True)
        else:
            ti, tl = None, None
        return x, y, ti, tl, use_teacher


def load_val_tokens(data_dir: str, seq_len: int) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(data_dir, "fineweb_val_*.bin")))
    if not paths:
        raise FileNotFoundError(f"no fineweb_val_*.bin in {data_dir}")
    tokens = np.concatenate([load_shard(p) for p in paths]).astype(np.int64)
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return torch.from_numpy(tokens[: usable + 1])


# ----------------------------- muon optimizer -----------------------------


def _zeropower_via_newtonschulz5_eager(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to approximate G(G^TG)^{-1/2}.

    From Keller Jordan's Muon optimizer (2024). Quintic polynomial coefficients
    (3.4445, -4.7750, 2.0315) converge fast for matrices with singular values
    bounded in [0, 1.2]. The Frobenius-norm division normalizes to that range.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


# torch.compile the inner NS function — nanoGPT speedrun does this to hide the
# NS iteration overhead (usually ~20% of per-step time on small Muon workloads).
try:
    zeropower_via_newtonschulz5 = torch.compile(_zeropower_via_newtonschulz5_eager, dynamic=False)
except Exception:
    zeropower_via_newtonschulz5 = _zeropower_via_newtonschulz5_eager


class Muon(torch.optim.Optimizer):
    """Muon: momentum+Newton-Schulz orthogonalization for 2D matrix params.

    Per Keller Jordan (2024), the current nanoGPT speedrun winner. Meant for
    hidden 2D linear layers (qkv, proj, mlp.*); use AdamW for embeddings,
    LayerNorms, and biases.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "mb" not in state:
                    state["mb"] = torch.zeros_like(g)
                buf = state["mb"]
                buf.mul_(momentum).add_(g)
                g_mom = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(g_mom, steps=ns_steps)
                # Muon's rectangular-matrix correction: scale by sqrt(fan_out/fan_in)
                # so that wider-than-tall matrices get proportionally bigger updates.
                scale = max(1.0, (p.size(0) / p.size(1)) ** 0.5)
                p.data.add_(update, alpha=-lr * scale)


# ----------------------------- loss -----------------------------


def distillation_loss(student_logits, y, teacher_idx, teacher_logp, T: float, alpha: float):
    """Hard-CE + soft-KL distillation.

    student_logits: (B, T, V)
    y:              (B, T)        int64 next-token targets
    teacher_idx:    (B, T, K)     int64 sp1024 token ids
    teacher_logp:   (B, T, K)     fp32 log-probs over the top-k positions
    T:              temperature (applied to both saved logprobs and student logits at top-k positions).
    alpha:          soft-loss weight (0 = pure hard, 1 = pure soft).

    NOTE: applying /T to cached logprobs then renormalizing over the top-k
    yields a proper distribution, but is not identical to softmax(logits/T)
    because the non-top-k mass is dropped before renormalization.
    """
    B, TT, V = student_logits.shape
    hard = F.cross_entropy(student_logits.view(-1, V), y.view(-1))
    if alpha <= 0.0:
        return hard, hard.detach(), torch.tensor(0.0, device=hard.device)

    # Gather student logits at teacher's top-k positions.
    student_at_top = student_logits.gather(dim=-1, index=teacher_idx)  # (B, T, K)
    s_logp = F.log_softmax(student_at_top / T, dim=-1)
    # Teacher distribution over top-k.
    t_logp_scaled = teacher_logp / T
    t_logp_norm = t_logp_scaled - torch.logsumexp(t_logp_scaled, dim=-1, keepdim=True)
    t_probs = t_logp_norm.exp()
    # Per-token KL: sum over top-k, mean over (B*T). batchmean on (B,T,K) divides
    # by B only → off by seq_len, producing huge loss values and unstable training.
    soft = (t_probs * (t_logp_norm - s_logp)).sum(dim=-1).mean() * (T * T)
    total = alpha * soft + (1.0 - alpha) * hard
    return total, hard.detach(), soft.detach()


# ----------------------------- bpb evaluation -----------------------------


def build_byte_lut(sp_model_path: str, vocab_size: int, device: torch.device):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    sp_v = int(sp.vocab_size())
    table = max(sp_v, vocab_size)
    base_bytes = np.zeros((table,), dtype=np.int16)
    has_space = np.zeros((table,), dtype=np.bool_)
    is_boundary = np.ones((table,), dtype=np.bool_)
    for tok in range(sp_v):
        if sp.is_control(tok) or sp.is_unknown(tok) or sp.is_unused(tok):
            continue
        is_boundary[tok] = False
        if sp.is_byte(tok):
            base_bytes[tok] = 1
            continue
        piece = sp.id_to_piece(tok)
        if piece.startswith("▁"):
            has_space[tok] = True
            piece = piece[1:]
        base_bytes[tok] = len(piece.encode("utf-8"))
    return (
        torch.from_numpy(base_bytes).to(device),
        torch.from_numpy(has_space).to(device),
        torch.from_numpy(is_boundary).to(device),
    )


@torch.inference_mode()
def eval_val_bpb(model, val_tokens: torch.Tensor, seq_len: int, device, base_bytes_lut, has_space_lut, is_boundary_lut, batch_seqs: int = 16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    n = val_tokens.numel()
    total_seqs = (n - 1) // seq_len
    for bstart in range(0, total_seqs, batch_seqs):
        bend = min(bstart + batch_seqs, total_seqs)
        seqs = []
        for i in range(bstart, bend):
            s = i * seq_len
            seqs.append(val_tokens[s : s + seq_len + 1])
        chunk = torch.stack(seqs).to(device, non_blocking=True)
        x = chunk[:, :-1]
        y = chunk[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum").float()
        total_loss += loss.item()
        total_tokens += y.numel()
        # Bytes-per-token via sp LUT.
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(torch.int64)
        tb = tb + (has_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(torch.int64)
        total_bytes += int(tb.sum().item())
    model.train()
    if total_tokens == 0 or total_bytes == 0:
        return float("inf"), float("inf")
    bits_per_tok = (total_loss / total_tokens) / math.log(2.0)
    toks_per_byte = total_tokens / total_bytes
    return total_loss / total_tokens, bits_per_tok * toks_per_byte


# ----------------------------- quantize + save -----------------------------


def _quantize_int8_rowwise(w: torch.Tensor):
    # w: (R, C). Symmetric per-row int8 quant, fp16 scale per row.
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float16)


def _dequantize_int8_rowwise(q: torch.Tensor, scale: torch.Tensor):
    # q: (R, C) int8, scale: (R,) fp16 → dequantized fp32
    return q.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)


def save_model_compressed(model: nn.Module, path: str) -> int:
    """Int8 per-row quant for 2D params; fp16 for 1D. zlib-compressed."""
    q_w, q_scale, fp16_w = {}, {}, {}
    for k, v in model.state_dict().items():
        vd = v.detach().cpu()
        if vd.dim() == 2 and vd.numel() >= 256:
            q, s = _quantize_int8_rowwise(vd.to(torch.float32))
            q_w[k] = q
            q_scale[k] = s
        else:
            fp16_w[k] = vd.to(torch.float16)
    blob_dict = {"q_w": q_w, "q_scale": q_scale, "fp16_w": fp16_w}
    buf = io.BytesIO()
    torch.save(blob_dict, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    with open(path, "wb") as f:
        f.write(blob)
    return os.path.getsize(path)


def load_model_compressed(path: str, model: nn.Module):
    with open(path, "rb") as f:
        blob = f.read()
    buf = io.BytesIO(zlib.decompress(blob))
    packed = torch.load(buf, map_location="cpu")
    reconstructed = {}
    for k, q in packed["q_w"].items():
        reconstructed[k] = _dequantize_int8_rowwise(q, packed["q_scale"][k])
    for k, v in packed["fp16_w"].items():
        reconstructed[k] = v.to(torch.float32)
    out = {}
    for k, v in model.state_dict().items():
        out[k] = reconstructed[k].to(dtype=v.dtype)
    model.load_state_dict(out, strict=True)


# ----------------------------- main -----------------------------


def main():
    random.seed(HP.seed); np.random.seed(HP.seed); torch.manual_seed(HP.seed)
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda")

    # Build model
    model = GPT(
        vocab=HP.vocab_size, dim=HP.model_dim, num_layers=HP.num_layers,
        num_heads=HP.num_heads, mlp_mult=HP.mlp_mult, seq_len=HP.seq_len,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"student params: {total_params/1e6:.2f}M  dim={HP.model_dim} layers={HP.num_layers}")

    # Data
    loader = DistillationDataLoader(
        data_dir=HP.data_dir, teacher_dir=HP.teacher_dir,
        seq_len=HP.seq_len, batch_size=HP.batch_size, seed=HP.seed,
        teacher_batch_frac=HP.teacher_batch_frac,
    )
    assert loader.top_k >= HP.teacher_top_k, (
        f"meta.json top_k={loader.top_k} < requested TEACHER_TOP_K={HP.teacher_top_k}"
    )
    top_k = HP.teacher_top_k

    val_tokens = load_val_tokens(HP.data_dir, HP.seq_len)
    base_lut, space_lut, boundary_lut = build_byte_lut(HP.tokenizer_path, HP.vocab_size, device)

    # Split params: Muon for hidden 2D linear weights, AdamW for everything else.
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Hidden 2D weights in the transformer blocks.
        if p.ndim == 2 and name.startswith("blocks."):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    print(f"optimizer split: muon={sum(p.numel() for p in muon_params)/1e6:.2f}M  "
          f"adamw={sum(p.numel() for p in adamw_params)/1e6:.2f}M", flush=True)

    opt_adamw = torch.optim.AdamW(
        adamw_params, lr=HP.lr, betas=(HP.beta1, HP.beta2), weight_decay=HP.weight_decay,
    )
    opt_muon = Muon(
        muon_params, lr=HP.muon_lr, momentum=HP.muon_momentum, nesterov=True, ns_steps=5,
    )

    adamw_lr_min = HP.lr * HP.lr_min_mult
    muon_lr_min = HP.muon_lr * HP.lr_min_mult

    def cosine_schedule(peak: float, floor: float, step: int, elapsed: float) -> float:
        if step < HP.warmup_steps:
            return peak * (step + 1) / HP.warmup_steps
        frac = max(0.0, min(1.0, elapsed / HP.max_wallclock_seconds))
        cos_w = 0.5 * (1.0 + math.cos(math.pi * frac))
        return floor + (peak - floor) * cos_w

    start = time.time()
    step = 0
    model.train()
    while True:
        elapsed = time.time() - start
        if elapsed >= HP.max_wallclock_seconds:
            break
        for g in opt_adamw.param_groups:
            g["lr"] = cosine_schedule(HP.lr, adamw_lr_min, step, elapsed)
        for g in opt_muon.param_groups:
            g["lr"] = cosine_schedule(HP.muon_lr, muon_lr_min, step, elapsed)

        x, y, ti, tl, use_teacher = loader.sample_batch(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            if use_teacher:
                ti = ti[..., :top_k]
                tl = tl[..., :top_k]
                loss, hard_l, soft_l = distillation_loss(
                    logits, y, ti, tl, T=HP.distill_temperature, alpha=HP.distill_alpha,
                )
            else:
                V = logits.size(-1)
                loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
                hard_l = loss.detach()
                soft_l = torch.tensor(0.0, device=loss.device)
        opt_adamw.zero_grad(set_to_none=True)
        opt_muon.zero_grad(set_to_none=True)
        loss.backward()
        opt_adamw.step()
        opt_muon.step()
        step += 1
        if step % 50 == 0:
            print(
                f"step={step} elapsed={elapsed:.0f}s loss={loss.item():.4f} "
                f"hard={hard_l.item():.4f} soft={soft_l.item():.4f} "
                f"adamw_lr={opt_adamw.param_groups[0]['lr']:.2e} "
                f"muon_lr={opt_muon.param_groups[0]['lr']:.2e}",
                flush=True,
            )

    print(f"training done: {step} steps in {time.time()-start:.0f}s")

    # Save + reload (compression round-trip) so val_bpb reflects the compressed model.
    artifact_path = "final_model.ptz"
    model_bytes = save_model_compressed(model, artifact_path)
    code_bytes = os.path.getsize(__file__)
    total_bytes = model_bytes + code_bytes
    print(f"final_model.ptz: {model_bytes} bytes")
    print(f"train.py:        {code_bytes} bytes")
    print(f"Total submission size: {total_bytes} bytes")

    load_model_compressed(artifact_path, model)
    val_loss, val_bpb = eval_val_bpb(
        model, val_tokens, HP.seq_len, device, base_lut, space_lut, boundary_lut,
    )
    # eval/eval.sh regex requires no `:` between the marker and `val_bpb:`.
    print(f"val_loss {val_loss:.8f}")
    print(f"final_int8_zlib_roundtrip_exact val_bpb:{val_bpb:.8f}")


if __name__ == "__main__":
    main()
