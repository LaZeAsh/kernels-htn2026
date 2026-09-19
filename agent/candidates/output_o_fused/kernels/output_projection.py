"""Decode split-K BF16 output projection with residual add."""

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
def _reduce_add(PART, RESIDUAL, OUT):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * 256 + tl.arange(0, 256)
    base = row * 4 * 2560 + col
    projection = (tl.load(PART + base) + tl.load(PART + base + 2560)
                  + tl.load(PART + base + 5120)
                  + tl.load(PART + base + 7680)).to(tl.bfloat16)
    residual = tl.load(RESIDUAL + row * 2560 + col).to(tl.float32)
    result = (projection.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(OUT + row * 2560 + col, result)


def project_add(x, weight, residual):
    """BF16 [B,1,K] @ original [2560,K] weight.T plus BF16 residual.

    K is 4096 for o_proj or 9728 for down_proj. B is 1..16. The
    projection is rounded to BF16 before the residual addition.
    """
    batch, length, k = x.shape
    if (batch < 1 or batch > 16 or length != 1 or k not in (4096, 9728)
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or weight.shape != (2560, k) or weight.dtype != x.dtype
            or not weight.is_contiguous()
            or residual.shape != (batch, 1, 2560)
            or residual.dtype != x.dtype or not residual.is_contiguous()):
        raise ValueError("invalid decode projection/add input")
    partial = torch.empty((batch, 4, 2560), device=x.device, dtype=torch.float32)
    out = torch.empty_like(residual)
    span = k // 4
    _project[(40, 4)](
        x, weight, partial, batch, k, span, span // 64,
        num_warps=4,
    )
    _reduce_add[(batch, 10)](
        partial, residual, out, num_warps=4, enable_fp_fusion=False,
    )
    return out
