"""Exact full-context decode GQA for fixed, contiguous BF16 KV cache.

Inputs: Q [B,32,1,128], K/V [B,8,C,128], scalar CUDA position [1].
Output: newly allocated BF16 [B,32,1,128]. No input mutation. Cache slot
`position` must already contain this step's K/V; keys above it are ignored.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q, K, V, POS, OUT, PM, PL, PA,
             QS0: tl.constexpr, QS1: tl.constexpr, QS3: tl.constexpr,
             C: tl.constexpr, SCALE: tl.constexpr, SPLITS: tl.constexpr,
             BLOCKS: tl.constexpr, BN: tl.constexpr):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    b = bh // 32
    h = bh % 32
    kvh = h // 4
    d = tl.arange(0, 128)
    n_lane = tl.arange(0, BN)
    q = tl.load(Q + b * QS0 + h * QS1 + d * QS3).to(tl.float32)
    last = tl.load(POS)
    base = (b * 8 + kvh) * C * 128
    m = -float("inf")
    denom = 0.0
    acc = tl.full((128,), 0.0, tl.float32)
    for block in range(BLOCKS):
        n = split * BLOCKS * BN + block * BN + n_lane
        valid = (n < C) & (n <= last)
        if split * BLOCKS * BN + block * BN <= last:
            offsets = base + n[:, None] * 128 + d[None, :]
            k = tl.load(K + offsets, mask=valid[:, None], other=0).to(tl.float32)
            score = tl.sum(k * q[None, :], 1) * SCALE
            score = tl.where(valid, score, -float("inf"))
            new_m = tl.maximum(m, tl.max(score, 0))
            alpha = tl.exp(m - new_m)
            prob = tl.exp(score - new_m)
            v = tl.load(V + offsets, mask=valid[:, None], other=0).to(tl.float32)
            acc = acc * alpha + tl.sum(prob[:, None] * v, 0)
            denom = denom * alpha + tl.sum(prob, 0)
            m = new_m
    if SPLITS == 1:
        tl.store(OUT + bh * 128 + d, acc / denom)
    else:
        slot = bh * SPLITS + split
        tl.store(PM + slot, m)
        tl.store(PL + slot, denom)
        tl.store(PA + slot * 128 + d, acc)


@triton.jit
def _combine(OUT, PM, PL, PA, SPLITS: tl.constexpr, BS: tl.constexpr):
    bh = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, 128)
    valid = s < SPLITS
    m_i = tl.load(PM + bh * SPLITS + s, mask=valid, other=-float("inf"))
    l_i = tl.load(PL + bh * SPLITS + s, mask=valid, other=0)
    m = tl.max(m_i, 0)
    scale = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m))
    acc_i = tl.load(PA + (bh * SPLITS + s[:, None]) * 128 + d[None, :],
                    mask=valid[:, None], other=0)
    denominator = tl.sum(l_i * scale, 0)
    numerator = tl.sum(acc_i * scale[:, None], 0)
    tl.store(OUT + bh * 128 + d, numerator / denominator)


def gqa_decode(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
               position: torch.Tensor, scaling: float) -> torch.Tensor:
    """Run full, causal, grouped-query attention for one decode position."""
    if query.dtype != torch.bfloat16 or keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
        raise ValueError("Q/K/V must be BF16")
    if query.ndim != 4 or query.shape[1:] != (32, 1, 128):
        raise ValueError("Q must have shape [B,32,1,128]")
    batch = query.shape[0]
    capacity = keys.shape[2]
    if keys.shape != (batch, 8, capacity, 128) or values.shape != keys.shape:
        raise ValueError("K/V must have shape [B,8,C,128]")
    if not keys.is_contiguous() or not values.is_contiguous():
        raise ValueError("K/V must be contiguous")
    if position.shape != (1,) or position.dtype != torch.int64 or not position.is_cuda:
        raise ValueError("position must be one CUDA int64 value")
    if not (query.is_cuda and keys.is_cuda and values.is_cuda):
        raise ValueError("Q/K/V must be CUDA tensors")
    splits = 4 if batch == 1 and capacity >= 1024 else 1
    block_n = 32
    blocks = triton.cdiv(capacity, splits * block_n)
    out = torch.empty((batch, 32, 1, 128), device=query.device, dtype=query.dtype)
    if splits > 1:
        pm = torch.empty((batch * 32 * splits,), device=query.device, dtype=torch.float32)
        pl = torch.empty_like(pm)
        pa = torch.empty((batch * 32 * splits, 128), device=query.device, dtype=torch.float32)
    else:
        pm = pl = pa = out
    _partial[(batch * 32, splits)](
        query, keys, values, position, out, pm, pl, pa,
        query.stride(0), query.stride(1), query.stride(3),
        capacity, scaling, splits, blocks, block_n,
        num_warps=4,
    )
    if splits > 1:
        _combine[(batch * 32,)](
            out, pm, pl, pa, splits, triton.next_power_of_2(splits), num_warps=4
        )
    return out
