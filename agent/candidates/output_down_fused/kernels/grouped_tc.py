"""Decode-only Tensor Core GQA: four Q heads share one KV tile load.

Q [B,32,1,128], contiguous BF16 KV [B,8,C,128], device position [1].
Computes all valid cached keys. Split-K changes reduction order only.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_partial(Q, K, V, POS, OUT, PM, PL, PA,
                     QS0: tl.constexpr, QS1: tl.constexpr, QS3: tl.constexpr,
                     C: tl.constexpr, SCALE: tl.constexpr, SPLITS: tl.constexpr,
                     BLOCKS: tl.constexpr, SPAN: tl.constexpr, BN: tl.constexpr):
    group = tl.program_id(0)
    split = tl.program_id(1)
    b = group // 8
    kvh = group % 8
    rows = tl.arange(0, 16)
    d = tl.arange(0, 128)
    n_lane = tl.arange(0, BN)
    q_ptr = Q + b * QS0 + (kvh * 4 + rows[:, None]) * QS1 + d[None, :] * QS3
    q = tl.load(q_ptr, mask=rows[:, None] < 4, other=0)
    last = tl.load(POS)
    base = (b * 8 + kvh) * C * 128
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.full((16,), 0.0, tl.float32)
    accumulator = tl.full((16, 128), 0.0, tl.float32)
    end = tl.where(split == SPLITS - 1, C, (split + 1) * SPAN)
    for block in range(BLOCKS):
        start = split * SPAN + block * BN
        n = start + n_lane
        valid = (n < end) & (n <= last)
        if (start <= last) & (start < end):
            kv_offset = base + n[:, None] * 128 + d[None, :]
            k = tl.load(K + kv_offset, mask=valid[:, None], other=0)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SCALE
            scores = tl.where(valid[None, :], scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.exp(maximum - new_maximum)
            probabilities = tl.exp(scores - new_maximum[:, None])
            v = tl.load(V + kv_offset, mask=valid[:, None], other=0)
            pv = tl.dot(probabilities.to(tl.bfloat16), v, out_dtype=tl.float32)
            accumulator = accumulator * alpha[:, None] + pv
            denominator = denominator * alpha + tl.sum(probabilities, 1)
            maximum = new_maximum
    head = b * 32 + kvh * 4 + rows
    if SPLITS == 1:
        tl.store(OUT + head[:, None] * 128 + d[None, :],
                 accumulator / denominator[:, None], mask=rows[:, None] < 4)
    else:
        slot = head * SPLITS + split
        tl.store(PM + slot, maximum, mask=rows < 4)
        tl.store(PL + slot, denominator, mask=rows < 4)
        tl.store(PA + slot[:, None] * 128 + d[None, :], accumulator,
                 mask=rows[:, None] < 4)


@triton.jit
def _combine(OUT, PM, PL, PA, SPLITS: tl.constexpr, BS: tl.constexpr):
    head = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, 128)
    valid = s < SPLITS
    partial_max = tl.load(PM + head * SPLITS + s, mask=valid, other=-float("inf"))
    partial_denominator = tl.load(PL + head * SPLITS + s, mask=valid, other=0)
    maximum = tl.max(partial_max, 0)
    factor = tl.where(partial_max == -float("inf"), 0.0,
                      tl.exp(partial_max - maximum))
    partial_acc = tl.load(PA + (head * SPLITS + s[:, None]) * 128 + d[None, :],
                          mask=valid[:, None], other=0)
    numerator = tl.sum(partial_acc * factor[:, None], 0)
    denominator = tl.sum(partial_denominator * factor, 0)
    tl.store(OUT + head * 128 + d, numerator / denominator)


def grouped_tc_decode(query: torch.Tensor, keys: torch.Tensor,
                      values: torch.Tensor, position: torch.Tensor,
                      scaling: float, prompt_length: int) -> torch.Tensor:
    """Exact full-context GQA with deterministic shape-based split selection."""
    if query.dtype != torch.bfloat16 or query.ndim != 4 or query.shape[1:] != (32, 1, 128):
        raise ValueError("Q must be BF16 [B,32,1,128]")
    batch = query.shape[0]
    capacity = keys.shape[2]
    if (keys.shape != (batch, 8, capacity, 128) or values.shape != keys.shape
            or keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16
            or not keys.is_contiguous() or not values.is_contiguous()):
        raise ValueError("K/V must be contiguous BF16 [B,8,C,128]")
    if not (query.is_cuda and keys.is_cuda and values.is_cuda):
        raise ValueError("Q/K/V must be CUDA")
    if position.shape != (1,) or position.dtype != torch.int64 or not position.is_cuda:
        raise ValueError("position must be one CUDA int64 value")
    if not 1 <= prompt_length <= capacity:
        raise ValueError("invalid prompt length")
    block_n = 64
    target = triton.cdiv(128, batch * 8)
    splits = max(1, min(target, triton.cdiv(prompt_length, block_n)))
    span = triton.cdiv(prompt_length, splits)
    last_span = capacity - (splits - 1) * span
    blocks = triton.cdiv(max(span, last_span), block_n)
    out = torch.empty((batch, 32, 1, 128), device=query.device, dtype=query.dtype)
    if splits > 1:
        pm = torch.empty((batch * 32, splits), device=query.device, dtype=torch.float32)
        pl = torch.empty_like(pm)
        pa = torch.empty((batch * 32, splits, 128), device=query.device, dtype=torch.float32)
    else:
        pm = pl = pa = out
    _grouped_partial[(batch * 8, splits)](
        query, keys, values, position, out, pm, pl, pa,
        query.stride(0), query.stride(1), query.stride(3),
        capacity, scaling, splits, blocks, span, block_n, num_warps=4,
    )
    if splits > 1:
        _combine[(batch * 32,)](
            out, pm, pl, pa, splits, triton.next_power_of_2(splits), num_warps=4,
        )
    return out
