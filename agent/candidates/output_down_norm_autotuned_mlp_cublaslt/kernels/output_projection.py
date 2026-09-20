"""Decode split-K down projection with residual add and next RMSNorm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _project(X, W, PART, B: tl.constexpr, K: tl.constexpr,
             SPAN: tl.constexpr, BLOCKS: tl.constexpr):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    row = tl.arange(0, 16)
    col = tile * 64 + tl.arange(0, 64)
    kk = tl.arange(0, 64)
    acc = tl.full((16, 64), 0.0, tl.float32)
    for block in range(BLOCKS):
        k = split * SPAN + block * 64 + kk
        x = tl.load(X + row[:, None] * K + k[None, :],
                    row[:, None] < B, 0)
        w = tl.load(W + col[:, None] * K + k[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    offset = (row[:, None] * 4 + split) * 2560 + col[None, :]
    tl.store(PART + offset, acc, row[:, None] < B)


@triton.jit
def _reduce_add_norm(PART, RESIDUAL, GAIN, OUT_RESIDUAL, OUT_NORM,
                     EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 4096)
    valid = col < 2560
    base = row * 4 * 2560 + col
    projection = (tl.load(PART + base, valid, 0)
                  + tl.load(PART + base + 2560, valid, 0)
                  + tl.load(PART + base + 5120, valid, 0)
                  + tl.load(PART + base + 7680, valid, 0)).to(tl.bfloat16)
    residual = tl.load(RESIDUAL + row * 2560 + col, valid, 0).to(tl.float32)
    result = (projection.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(OUT_RESIDUAL + row * 2560 + col, result, valid)
    value = result.to(tl.float32)
    variance = tl.sum(value * value, 0) / 2560.0
    normalized = (value * tl.rsqrt(variance + EPS)).to(tl.bfloat16)
    gain = tl.load(GAIN + col, valid, 0).to(tl.float32)
    tl.store(OUT_NORM + row * 2560 + col,
             (normalized.to(tl.float32) * gain).to(tl.bfloat16), valid)


def down_project_add_norm(x, weight, residual, gain, eps):
    """BF16 down projection plus residual and next RMSNorm, B 1..16."""
    batch, length, k = x.shape
    if (batch < 1 or batch > 16 or length != 1 or k != 9728
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or weight.shape != (2560, k) or weight.dtype != x.dtype
            or not weight.is_contiguous()
            or residual.shape != (batch, 1, 2560)
            or residual.dtype != x.dtype or not residual.is_contiguous()
            or gain.shape != (2560,) or gain.dtype != x.dtype
            or not gain.is_contiguous()):
        raise ValueError("invalid decode projection/add input")
    partial = torch.empty((batch, 4, 2560), device=x.device, dtype=torch.float32)
    out_residual = torch.empty_like(residual)
    out_norm = torch.empty_like(residual)
    span = k // 4
    _project[(40, 4)](
        x, weight, partial, batch, k, span, span // 64,
        num_warps=4,
    )
    _reduce_add_norm[(batch,)](
        partial, residual, gain, out_residual, out_norm, eps,
        num_warps=8, enable_fp_fusion=False,
    )
    return out_residual, out_norm
