"""Four-token batched causal GQA over the full KV prefix with precise PV."""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_partial(Q, K, V, POS, OUT, PM, PL, PA,
                     QS0: tl.constexpr, QS1: tl.constexpr,
                     QS2: tl.constexpr, QS3: tl.constexpr,
                     PS: tl.constexpr,
                     C: tl.constexpr, SCALE: tl.constexpr, SPLITS: tl.constexpr,
                     BLOCKS: tl.constexpr, SPAN: tl.constexpr, BN: tl.constexpr):
    group = tl.program_id(0)
    split = tl.program_id(1)
    b = group // 8
    kvh = group % 8
    rows = tl.arange(0, 16)
    d = tl.arange(0, 128)
    n_lane = tl.arange(0, BN)
    q_head = kvh * 4 + rows // 4
    q_time = rows % 4
    q_ptr = (Q + b * QS0 + q_head[:, None] * QS1
             + q_time[:, None] * QS2 + d[None, :] * QS3)
    q = tl.load(q_ptr)
    last = tl.load(POS + b * PS)
    base = (b * 8 + kvh) * C * 128
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.full((16,), 0.0, tl.float32)
    accumulator = tl.full((16, 128), 0.0, tl.float32)
    end = tl.where(split == SPLITS - 1, C, (split + 1) * SPAN)
    for block in range(BLOCKS):
        start = split * SPAN + block * BN
        n = start + n_lane
        valid = (n < end) & (n <= last + 3)
        if (start <= last + 3) & (start < end):
            kv_offset = base + n[:, None] * 128 + d[None, :]
            k = tl.load(K + kv_offset, mask=valid[:, None], other=0)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SCALE
            causal = valid[None, :] & (n[None, :] <= last + q_time[:, None])
            scores = tl.where(causal, scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.where(new_maximum == -float("inf"), 0.0,
                             tl.exp(maximum - new_maximum))
            probabilities = tl.where(causal,
                                      tl.exp(scores - new_maximum[:, None]), 0.0)
            v = tl.load(V + kv_offset, mask=valid[:, None], other=0)
            # Two BF16 terms retain the probability's low bits for the PV MMA.
            p_hi = probabilities.to(tl.bfloat16)
            p_lo = (probabilities - p_hi.to(tl.float32)).to(tl.bfloat16)
            pv = tl.dot(p_hi, v, out_dtype=tl.float32)
            pv = tl.dot(p_lo, v, pv)
            accumulator = accumulator * alpha[:, None] + pv
            denominator = denominator * alpha + tl.sum(probabilities, 1)
            maximum = new_maximum
    head = (b * 32 + q_head) * 4 + q_time
    if SPLITS == 1:
        tl.store(OUT + head[:, None] * 128 + d[None, :],
                 accumulator / denominator[:, None])
    else:
        slot = head * SPLITS + split
        tl.store(PM + slot, maximum)
        tl.store(PL + slot, denominator)
        tl.store(PA + slot[:, None] * 128 + d[None, :], accumulator)


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


def grouped_tc_window(query: torch.Tensor, keys: torch.Tensor,
                      values: torch.Tensor, position: torch.Tensor,
                      scaling: float, prompt_length: int) -> torch.Tensor:
    """Full-prefix GQA for four query positions per sequence, B<=4."""
    if (query.dtype != torch.bfloat16 or query.shape[0] < 1
            or query.shape[0] > 4 or query.shape[1:] != (32, 4, 128)):
        raise ValueError("Q must be BF16 [B,32,4,128], B<=4")
    batch = query.shape[0]
    capacity = keys.shape[2]
    if (keys.shape != (batch, 8, capacity, 128) or values.shape != keys.shape
            or keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16
            or not keys.is_contiguous() or not values.is_contiguous()):
        raise ValueError("K/V must be contiguous BF16 [B,8,C,128]")
    if not (query.is_cuda and keys.is_cuda and values.is_cuda):
        raise ValueError("Q/K/V must be CUDA")
    if (position.shape != (batch,) or position.dtype != torch.int64
            or not position.is_cuda):
        raise ValueError("position must have one CUDA int64 value per sequence")
    if not 1 <= prompt_length <= capacity:
        raise ValueError("invalid prompt length")
    block_n = 64
    target = triton.cdiv(128, batch * 8)
    splits = max(1, min(target, triton.cdiv(prompt_length, block_n)))
    span = triton.cdiv(prompt_length, splits)
    last_span = capacity - (splits - 1) * span
    blocks = triton.cdiv(max(span, last_span), block_n)
    out = torch.empty((batch, 32, 4, 128), device=query.device, dtype=query.dtype)
    if splits > 1:
        pm = torch.empty((batch * 32 * 4, splits), device=query.device, dtype=torch.float32)
        pl = torch.empty_like(pm)
        pa = torch.empty((batch * 32 * 4, splits, 128), device=query.device, dtype=torch.float32)
    else:
        pm = pl = pa = out
    _grouped_partial[(batch * 8, splits)](
        query, keys, values, position, out, pm, pl, pa,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        position.stride(0),
        capacity, scaling, splits, blocks, span, block_n, num_warps=4,
    )
    if splits > 1:
        _combine[(batch * 32 * 4,)](
            out, pm, pl, pa, splits, triton.next_power_of_2(splits), num_warps=4,
        )
    return out
